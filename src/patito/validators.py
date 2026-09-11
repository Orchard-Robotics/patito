"""Module for validating datastructures with respect to model specifications."""

from __future__ import annotations

import inspect
import itertools
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
    cast,
    overload,
)

import polars as pl
from pydantic.aliases import AliasGenerator
from typing_extensions import get_args

from patito._pydantic.dtypes import is_optional
from patito._pydantic.dtypes.utils import unwrap_optional
from patito.exceptions import (
    ColumnDTypeError,
    DataFrameValidationError,
    ErrorWrapper,
    MissingColumnsError,
    MissingValuesError,
    PrimaryKeyUniquenessError,
    RowValueError,
    SuperfluousColumnsError,
    UnvalidatedConstraintWarning,
)

try:
    import pandas as pd  # type: ignore

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

if TYPE_CHECKING:
    from polars._typing import ConcatMethod

    from patito import Model


def _horizontal_concat_method() -> str:
    """Return the name of the concat method which lines up one-row frames side by side.

    Polars 1.42.1 deprecated the padding behaviour of ``how="horizontal"`` in favour of
    an explicit ``how="horizontal_extend"``, reserving ``"horizontal"`` for a future
    variant which will demand equal heights. Every frame combined here is a single-row
    aggregation, so the two spellings behave identically; the explicit one is used
    wherever it exists in order to keep the deprecation warning quiet.
    """
    try:
        from polars._typing import ConcatMethod as _ConcatMethod
    except ImportError:  # pragma: no cover - a private module, which may yet move
        return "horizontal"

    if "horizontal_extend" in get_args(_ConcatMethod):
        return "horizontal_extend"
    return "horizontal"


_HORIZONTAL_CONCAT = _horizontal_concat_method()


VALID_POLARS_TYPES = {
    "enum": {pl.Categorical},
    "boolean": {pl.Boolean},
    "string": {pl.String, pl.Datetime, pl.Date},
    "number": {pl.Float32, pl.Float64},
    "integer": {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    },
}

Frame = TypeVar("Frame", bound=Union[pl.DataFrame, pl.LazyFrame])


def _column_names(frame: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Return the column names of an eager or lazy frame without collecting it."""
    return frame.collect_schema().names()


def _caller_stacklevel() -> int:
    """Return the stack level of the nearest frame outside of patito itself.

    Warnings are raised on behalf of whoever asked for the validation, so they should
    point at their code rather than at the internals of this package.
    """
    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frame = inspect.currentframe()
    for level in itertools.count():
        if frame is None:
            break
        if not os.path.abspath(frame.f_code.co_filename).startswith(package):
            return max(level, 1)
        frame = frame.f_back
    return 1


def _transform_frame(frame: Frame, schema: type[Model]) -> Frame:
    """Transform any properties of the frame according to the model.

    Currently only supports using AliasGenerator to transform column names to match a
    model.

    Args:
        frame: Polars DataFrame or LazyFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.

    Returns:
        The same kind of frame as the one given, with the model-mandated
        transformations applied.

    """
    # Check if an alias generator is present in model_config
    alias_gen = schema.model_config.get("alias_generator")
    if not alias_gen:
        return frame

    if isinstance(alias_gen, AliasGenerator):
        alias_func = alias_gen.validation_alias or alias_gen.alias
        assert (
            alias_func is not None
        ), "An AliasGenerator must contain a transforming function"
    else:  # alias_gen is a function
        alias_func = alias_gen

    renaming = {
        field_name: cast(str, alias_func(field_name))
        for field_name in _column_names(frame)
    }
    return cast(
        Frame,
        frame.rename({old: new for old, new in renaming.items() if old != new}),
    )


class _Checks:
    """Accumulator of validation errors, deferring those that depend on frame content.

    Structural checks, such as missing columns and dtype mismatches, are resolved
    immediately from the frame schema. Checks which depend on the *content* of the
    frame are instead registered as aggregation expressions producing a single row,
    all of which are evaluated in one go once every check has been registered. That
    way validation never materializes the frame itself, only the results of the
    aggregations, and a lazy frame can be validated without collecting it.
    """

    def __init__(
        self, schema_only: bool = False, streamable_only: bool = False
    ) -> None:
        """Construct an empty accumulator.

        Args:
            schema_only: If True, content-dependent checks are dropped rather than
                registered, restricting validation to what can be determined from the
                frame schema alone.
            streamable_only: If True, checks which need to see the frame in its
                entirety are dropped rather than registered, restricting validation to
                what can be determined from any one batch of rows on its own.

        """
        self._schema_only = schema_only
        self._streamable_only = streamable_only
        self._unvalidated: list[str] = []
        # Either an already detected error, or the alias of an aggregation paired with
        # the factory turning the result of that aggregation into an optional error.
        self._entries: list[
            Union[ErrorWrapper, tuple[str, Callable[[Any], Union[ErrorWrapper, None]]]]
        ] = []
        self._selections: dict[int, tuple[pl.LazyFrame, list[pl.Expr]]] = {}

    def add(self, error: ErrorWrapper) -> None:
        """Register an already detected error."""
        self._entries.append(error)

    def defer(
        self,
        frame: pl.LazyFrame,
        aggregation: pl.Expr,
        factory: Callable[[Any], ErrorWrapper | None],
        requires_full_frame: str | None = None,
    ) -> None:
        """Register a check to be performed once the given aggregation has been run.

        Args:
            frame: The lazy frame to evaluate the aggregation over.
            aggregation: An expression which must evaluate to exactly one value.
            factory: Callable converting that value into an error, or into ``None`` if
                the value turns out to be acceptable after all.
            requires_full_frame: Set to the location of the check for aggregations
                which cannot be decided from a batch of rows in isolation, and which
                must therefore be dropped when only streamable checks are wanted.

        """
        if self._schema_only:
            return
        if requires_full_frame is not None and self._streamable_only:
            self._unvalidated.append(requires_full_frame)
            return
        alias = f"__patito_check_{len(self._entries)}"
        _, aggregations = self._selections.setdefault(id(frame), (frame, []))
        aggregations.append(aggregation.alias(alias))
        self._entries.append((alias, factory))

    @property
    def unvalidated(self) -> list[str]:
        """The locations of the checks which have been dropped as unstreamable."""
        return self._unvalidated

    def _aggregations(self) -> pl.LazyFrame | None:
        """Combine every deferred aggregation into a single one-row lazy frame."""
        if not self._selections:
            return None
        selections = [
            frame.select(aggregations)
            for frame, aggregations in self._selections.values()
        ]
        if len(selections) == 1:
            return selections[0]
        return pl.concat(selections, how=cast("ConcatMethod", _HORIZONTAL_CONCAT))

    @staticmethod
    def _values(aggregations: pl.DataFrame) -> Mapping[str, Any]:
        """Extract the single row of aggregated values.

        Every registered check aggregates its frame down to exactly one row, which is
        what allows the results of checks over different frames to be lined up side by
        side. A check which neglected to aggregate would instead be padded out with
        nulls and silently misread from its first row, so the invariant is pinned down
        here rather than left to the concatenation to honour.
        """
        assert aggregations.height == 1, (
            "Validation checks must aggregate to exactly one row, but "
            f"{', '.join(aggregations.columns)} produced {aggregations.height}"
        )
        return aggregations.row(0, named=True)

    def _errors(self, values: Mapping[str, Any]) -> list[ErrorWrapper]:
        """Assemble the errors, in registration order, from the aggregated values."""
        errors: list[ErrorWrapper] = []
        for entry in self._entries:
            if isinstance(entry, ErrorWrapper):
                errors.append(entry)
                continue
            alias, factory = entry
            error = factory(values[alias])
            if error is not None:
                errors.append(error)
        return errors

    def resolve(self) -> list[ErrorWrapper]:
        """Evaluate all deferred checks and return the errors in registration order."""
        aggregations = self._aggregations()
        if aggregations is None:
            return self._errors({})
        return self._errors(self._values(aggregations.collect()))

    def attach(self, frame: pl.LazyFrame, schema: type[Model]) -> pl.LazyFrame:
        """Attach the checks to the given frame, to be resolved when it is collected.

        The aggregations are added to the query plan as a branch which yields no rows
        of its own, but which raises if the data turns out to be invalid. Polars'
        common subplan elimination lets that branch share the scan of the data itself,
        so validating this way costs one pass over the data rather than two.

        Args:
            frame: The lazy frame to attach the checks to.
            schema: Patito model to report validation errors against.

        Returns:
            A lazy frame yielding the same data as the given one, which raises a
            ``DataFrameValidationError`` upon collection if the data is invalid.

        """
        if not self._entries:
            return frame

        frame_schema = frame.collect_schema()

        def _validate(aggregations: pl.DataFrame) -> pl.DataFrame:
            errors = self._errors(self._values(aggregations))
            if errors:
                raise DataFrameValidationError(errors=errors, model=schema)
            return pl.DataFrame(schema=frame_schema)

        aggregations = self._aggregations()
        if aggregations is None:
            # Only structural errors have been registered, but they should still be
            # raised upon collection, so a trivial aggregation is used to carry them
            aggregations = frame.select(pl.len())

        guard = aggregations.map_batches(
            _validate, schema=frame_schema, validate_output_schema=False
        )
        return pl.concat([guard, frame], how="vertical")


def _attach_streaming_checks(
    frame: pl.LazyFrame,
    schema: type[Model],
    columns: Sequence[str] | None = None,
    allow_missing_columns: bool = False,
    allow_superfluous_columns: bool = False,
) -> pl.LazyFrame:
    """Attach the checks to the given frame, to be applied batch by batch.

    Every check which can be decided from a batch of rows on its own is applied to each
    batch as it flows through the query, raising as soon as a batch is found to be
    invalid. Validation therefore costs no additional pass over the data, holds no more
    than a batch in memory, and does not need to reach the end of the data in order to
    reject it.

    Checks which need to see the frame in its entirety, namely the uniqueness ones, are
    dropped, as a duplicate may be spread across any two batches. Doing so is reported
    through an ``UnvalidatedConstraintWarning``.

    Args:
        frame: The lazy frame to attach the checks to.
        schema: Patito model which specifies how the dataframe should be structured.
        columns: If specified, only validate the given columns.
        allow_missing_columns: If True, missing columns will not be considered an error.
        allow_superfluous_columns: If True, additional columns will not be considered an error.

    Returns:
        A lazy frame yielding the same data as the given one, which raises a
        ``DataFrameValidationError`` upon collection if any batch is invalid.

    """

    def _check(batch: pl.DataFrame) -> _Checks:
        checks = _Checks(streamable_only=True)
        _register_checks(
            checks=checks,
            dataframe=batch.lazy(),
            schema=schema,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
        )
        return checks

    # Registering the checks reads nothing but the frame schema, so they can be
    # registered up front purely in order to report the ones that had to be dropped
    unvalidated = _check(pl.DataFrame(schema=frame.collect_schema())).unvalidated
    if unvalidated:
        warnings.warn(
            f"Uniqueness of {', '.join(unvalidated)} is not validated when streaming, "
            "as duplicates may be spread across any two batches of rows. Validate "
            "without 'streaming' in order to check it.",
            UnvalidatedConstraintWarning,
            stacklevel=_caller_stacklevel(),
        )

    def _validate_batch(batch: pl.DataFrame) -> pl.DataFrame:
        # A fresh accumulator per batch, as batches are handed to us concurrently
        errors = _check(batch).resolve()
        if errors:
            raise DataFrameValidationError(errors=errors, model=schema)
        return batch

    return frame.map_batches(
        _validate_batch,
        schema=frame.collect_schema(),
        validate_output_schema=False,
        streamable=True,
    )


def _missing_values_message(num_missing_values: int, in_lists: bool = False) -> str:
    return (
        f"{num_missing_values} missing "
        f"{'value' if num_missing_values == 1 else 'values'}"
        f"{' in lists' if in_lists else ''}"
    )


def _enum_properties(
    props: dict[str, Any], schema: type[Model]
) -> dict[str, Any] | None:
    """Return the sub-properties declaring the permissible values, if there are any."""
    if "enum" in props:
        return props
    if "items" in props and "enum" in props["items"]:
        return _enum_properties(props["items"], schema)
    for item in props.get("anyOf", []):
        if "enum" in item:
            return _enum_properties(item, schema)
        if (
            "$ref" in item
        ):  # If the item is a reference to another definition pass it as the properties
            return _enum_properties(
                schema.model_json_schema()["$defs"][item["$ref"]], schema
            )
    return None


def _register_checks(  # noqa: C901
    checks: _Checks,
    dataframe: pl.LazyFrame,
    schema: type[Model],
    columns: Sequence[str] | None = None,
    allow_missing_columns: bool = False,
    allow_superfluous_columns: bool = False,
    loc_prefix: str = "",
) -> None:
    """Register every check mandated by the schema onto the given accumulator.

    Args:
        checks: Accumulator which the checks are registered onto.
        dataframe: Polars LazyFrame to be validated.
        schema: Patito model which specifies how the dataframe should be structured.
        columns: If specified, only validate the given columns. Missing columns will
            check if any specified columns are missing from the inputted dataframe,
            and superfluous columns will check if any columns not specified in the
            schema are present in the columns list.
        allow_missing_columns: If True, missing columns will not be considered an error.
        allow_superfluous_columns: If True, additional columns will not be considered an error.
        loc_prefix: Prefix prepended to the location of every registered error, used to
            namespace the errors of nested struct fields.

    The errors eventually produced by the accumulator are patito.exception.ErrorWrapper
    instances, the specific validation error of which can be retrieved from their "exc"
    attribute:

        MissingColumnsError: If there are any missing columns.
        SuperfluousColumnsError: If there are additional, non-specified columns.
        MissingValuesError: If there are nulls in a non-optional column.
        ColumnDTypeError: If any column has the wrong dtype.
        NotImplementedError: If validation has not been implement for the given
            type.

    """
    frame_schema = dataframe.collect_schema()
    frame_columns = frame_schema.names()

    schema_subset = columns or schema.columns
    column_subset = columns or frame_columns
    # Columns which are both requested for validation and actually present. Columns
    # which are requested but absent are reported as missing rather than inspected.
    checked_columns = set(column_subset).intersection(frame_columns)

    if not allow_missing_columns:
        # Check if any columns are missing
        for missing_column in set(schema_subset) - set(frame_columns):
            col_info = schema.column_infos.get(missing_column)
            if col_info is not None and col_info.allow_missing:
                continue

            checks.add(
                ErrorWrapper(
                    MissingColumnsError("Missing column"),
                    loc=f"{loc_prefix}{missing_column}",
                )
            )

    if not (allow_superfluous_columns or schema.model_config.get("extra") == "allow"):
        # Check if any additional columns are included
        for superfluous_column in set(column_subset) - set(schema.columns):
            checks.add(
                ErrorWrapper(
                    SuperfluousColumnsError("Superfluous column"),
                    loc=f"{loc_prefix}{superfluous_column}",
                )
            )

    # Check if any non-optional columns have null values
    for column in schema.non_nullable_columns.intersection(checked_columns):

        def _missing_values(
            num_missing_values: int, column: str = column
        ) -> ErrorWrapper | None:
            if not num_missing_values:
                return None
            return ErrorWrapper(
                MissingValuesError(_missing_values_message(num_missing_values)),
                loc=f"{loc_prefix}{column}",
            )

        checks.defer(dataframe, pl.col(column).null_count(), _missing_values)

    # check for primary keys uniqueness
    if schema.primary_key_columns and set(schema.primary_key_columns).issubset(
        frame_columns
    ):
        primary_key_columns = list(schema.primary_key_columns)

        def _duplicated_primary_keys(num_duplicated: int) -> ErrorWrapper | None:
            if not num_duplicated:
                return None
            # Only the offending rows are materialized, and only in order for them to
            # be displayed as part of the error message
            duplicated = (
                dataframe.select(primary_key_columns)
                .filter(pl.struct(pl.all()).is_duplicated())
                .collect()
            )
            return ErrorWrapper(
                PrimaryKeyUniquenessError(f"Primary key is not unique \n{duplicated}"),
                loc=f"{loc_prefix}{', '.join(primary_key_columns)}",
            )

        checks.defer(
            dataframe,
            pl.struct(primary_key_columns).is_duplicated().sum(),
            _duplicated_primary_keys,
            # Duplicates may be spread arbitrarily far apart across the frame
            requires_full_frame=f"{loc_prefix}{', '.join(primary_key_columns)}",
        )

    # Check if any lists contain null items, which is only permissible if the item
    # annotation is itself optional
    for column, dtype in schema.dtypes.items():
        if column not in checked_columns:
            continue
        if not isinstance(dtype, pl.List):
            continue
        if not isinstance(frame_schema[column], pl.List):
            # The column does not contain lists at all, which is reported as a dtype
            # error rather than inspected any further
            continue

        annotation = schema.model_fields[column].annotation  # type: ignore[unreachable]

        # Retrieve the annotation of the list itself,
        # dewrapping any potential Optional[...]
        list_type = unwrap_optional(annotation)

        # Check if the list items themselves should be considered nullable
        item_type = get_args(list_type)[0]
        if is_optional(item_type):
            continue

        def _missing_list_values(
            num_missing_values: int | None, column: str = column
        ) -> ErrorWrapper | None:
            if not num_missing_values:
                return None
            return ErrorWrapper(
                MissingValuesError(
                    _missing_values_message(num_missing_values, in_lists=True)
                ),
                loc=f"{loc_prefix}{column}",
            )

        # The number of nulls contained in the lists of a column is the number of items
        # which do not survive dropping the nulls of those same lists. Rows which do not
        # contain a list at all evaluate to null, and are thus ignored by the sum.
        list_column = pl.col(column)
        checks.defer(
            dataframe,
            (list_column.list.len() - list_column.list.drop_nulls().list.len()).sum(),
            _missing_list_values,
        )

    # Check if any column has a wrong dtype
    valid_dtypes = schema.valid_dtypes
    for column_name, column_properties in schema._schema_properties.items():
        column_info = schema.column_infos[column_name]
        if column_name not in checked_columns:
            continue

        polars_type = frame_schema[column_name]
        if polars_type not in [
            pl.Struct,
            pl.List(pl.Struct),
        ]:  # defer struct validation for recursive call to _register_checks later
            if polars_type not in valid_dtypes[column_name]:
                checks.add(
                    ErrorWrapper(
                        ColumnDTypeError(
                            f"Polars dtype {polars_type} does not match model field type."
                        ),
                        loc=f"{loc_prefix}{column_name}",
                    )
                )

        # Test for when only specific values are accepted
        _register_enum_check(
            checks=checks,
            dataframe=dataframe,
            dtype=polars_type,
            column_name=column_name,
            props=column_properties,
            schema=schema,
            loc_prefix=loc_prefix,
        )

        if column_info.unique:

            def _duplicates(
                num_duplicated: int | None, column_name: str = column_name
            ) -> ErrorWrapper | None:
                # Coalescing to 0 in the case of dataframe of height 0
                if not (num_duplicated or 0):
                    return None
                return ErrorWrapper(
                    RowValueError(f"{num_duplicated} rows with duplicated values."),
                    loc=f"{loc_prefix}{column_name}",
                )

            checks.defer(
                dataframe,
                pl.col(column_name).is_duplicated().sum(),
                _duplicates,
                # Duplicates may be spread arbitrarily far apart across the frame
                requires_full_frame=f"{loc_prefix}{column_name}",
            )

        # Intercept struct columns, and process errors separately
        if schema.dtypes[column_name] == pl.Struct and isinstance(
            polars_type, pl.Struct
        ):
            nested_schema = schema.model_fields[column_name].annotation
            assert nested_schema is not None

            nested_frame = dataframe
            # Additional unpack required if structs column is optional
            if is_optional(nested_schema):
                nested_schema = unwrap_optional(nested_schema)

                # An optional struct means that we allow the struct entry to be
                # null. It is the inner model that is responsible for determining
                # whether its fields are optional or not. Since the struct is optional,
                # we need to filter out any null rows as the inner model may disallow
                # nulls on a particular field

                # NB As of Polars 1.1, struct_col.is_null() cannot return True
                # The following code has been added to accomodate this

                col_struct = pl.col(column_name).struct
                only_non_null_expr = ~pl.all_horizontal(
                    [col_struct.field(f.name).is_null() for f in polars_type.fields]
                )
                nested_frame = nested_frame.filter(only_non_null_expr)

            _register_checks(
                checks=checks,
                dataframe=nested_frame.select(column_name).unnest(column_name),
                schema=nested_schema,
                loc_prefix=f"{loc_prefix}{column_name}.",
            )

            # No need to do any more checks
            continue

        # Intercept list of structs columns, and process errors separately
        elif schema.dtypes[column_name] == pl.List(pl.Struct) and isinstance(
            polars_type, pl.List
        ):
            list_annotation = schema.model_fields[column_name].annotation
            assert list_annotation is not None

            nested_frame = dataframe
            # Handle Optional[list[pl.Struct]]
            if is_optional(list_annotation):
                list_annotation = unwrap_optional(list_annotation)

                nested_frame = nested_frame.filter(pl.col(column_name).is_not_null())

            # Unpack list schema
            nested_schema = list_annotation.__args__[0]

            nested_frame = (
                nested_frame.select(column_name)
                .filter(pl.col(column_name).list.len() > 0)
                .explode(column_name)
                .unnest(column_name)
            )

            # Handle list[Optional[pl.Struct]]
            if is_optional(nested_schema):
                nested_schema = unwrap_optional(nested_schema)

                nested_frame = nested_frame.filter(pl.all().is_not_null())

            _register_checks(
                checks=checks,
                dataframe=nested_frame,
                schema=nested_schema,
                loc_prefix=f"{loc_prefix}{column_name}.",
            )

            # No need to do any more checks
            continue

        # Check for bounded value fields
        col = pl.col(column_name)
        filters = {
            "maximum": lambda v, col=col: col <= v,
            "exclusiveMaximum": lambda v, col=col: col < v,
            "minimum": lambda v, col=col: col >= v,
            "exclusiveMinimum": lambda v, col=col: col > v,
            "multipleOf": lambda v, col=col: (col == 0) | ((col % v) == 0),
            "const": lambda v, col=col: col == v,
            "pattern": lambda v, col=col: col.str.contains(v),
            "minLength": lambda v, col=col: col.str.len_chars() >= v,
            "maxLength": lambda v, col=col: col.str.len_chars() <= v,
        }

        # Remove string checks for non-string types
        string_only = {"pattern", "minLength", "maxLength"}
        if polars_type != pl.String:
            filters = {k: v for k, v in filters.items() if k not in string_only}

        if "anyOf" in column_properties:
            bounds = [
                check(x[key])
                for key, check in filters.items()
                for x in column_properties["anyOf"]
                if key in x
            ]
        else:
            bounds = []
        bounds += [
            check(column_properties[key])
            for key, check in filters.items()
            if key in column_properties
        ]
        if bounds:

            def _out_of_bounds(
                n_invalid_rows: int | None, column_name: str = column_name
            ) -> ErrorWrapper | None:
                if not n_invalid_rows:
                    return None
                return ErrorWrapper(
                    RowValueError(
                        f"{n_invalid_rows} row{'' if n_invalid_rows == 1 else 's'} "
                        "with out of bound values."
                    ),
                    loc=f"{loc_prefix}{column_name}",
                )

            # Count the failing rows of each bound. Nulls evaluate to null on a boolean
            # check, and are ignored, as we only want the failures (false).
            checks.defer(
                dataframe,
                pl.sum_horizontal([(~bound).sum() for bound in bounds]),
                _out_of_bounds,
            )

        if column_info.constraints is not None:
            custom_constraints = column_info.constraints
            if isinstance(custom_constraints, pl.Expr):
                custom_constraints = [custom_constraints]
            constraints = pl.any_horizontal(
                [constraint.not_() for constraint in custom_constraints]
            )

            constrained_frame = dataframe
            if "_" in constraints.meta.root_names():
                # An underscore is an alias for the current field
                constrained_frame = dataframe.with_columns(
                    pl.col(column_name).alias("_")
                )

            def _unmatched_constraints(
                num_illegal_rows: int | None, column_name: str = column_name
            ) -> ErrorWrapper | None:
                if not num_illegal_rows:
                    return None
                return ErrorWrapper(
                    RowValueError(
                        f"{num_illegal_rows} "
                        f"row{'' if num_illegal_rows == 1 else 's'} "
                        "does not match custom constraints."
                    ),
                    loc=f"{loc_prefix}{column_name}",
                )

            checks.defer(constrained_frame, constraints.sum(), _unmatched_constraints)


def _register_enum_check(
    checks: _Checks,
    dataframe: pl.LazyFrame,
    dtype: pl.DataType,
    column_name: str,
    props: dict[str, Any],
    schema: type[Model],
    loc_prefix: str = "",
) -> None:
    """Register a check of the values of a column against its permissible values."""
    enum_props = _enum_properties(props, schema)
    if enum_props is None:
        return

    permissible_values = set(enum_props["enum"])
    if column_name in schema.nullable_columns:
        permissible_values.add(None)

    def _impermissible_values(actual_values: list[Any]) -> ErrorWrapper | None:
        impermissible_values = set(actual_values) - permissible_values
        if not impermissible_values:
            return None
        return ErrorWrapper(
            RowValueError(f"Rows with invalid values: {impermissible_values}."),
            loc=f"{loc_prefix}{column_name}",
        )

    column = pl.col(column_name)
    if isinstance(dtype, pl.List):
        column = column.explode()
    # Imploding the distinct values keeps this a single-row aggregation, so that it can
    # be evaluated alongside every other check
    checks.defer(dataframe, column.unique().implode(), _impermissible_values)


@overload
def validate(
    dataframe: pl.LazyFrame,
    schema: type[Model],
    columns: Sequence[str] | None = ...,
    allow_missing_columns: bool = ...,
    allow_superfluous_columns: bool = ...,
    drop_superfluous_columns: bool = ...,
    schema_only: bool = ...,
    on_collect: bool = ...,
    streaming: bool = ...,
) -> pl.LazyFrame: ...


@overload
def validate(
    dataframe: pd.DataFrame | pl.DataFrame,
    schema: type[Model],
    columns: Sequence[str] | None = ...,
    allow_missing_columns: bool = ...,
    allow_superfluous_columns: bool = ...,
    drop_superfluous_columns: bool = ...,
    schema_only: bool = ...,
    on_collect: bool = ...,
    streaming: bool = ...,
) -> pl.DataFrame: ...


def validate(
    dataframe: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    schema: type[Model],
    columns: Sequence[str] | None = None,
    allow_missing_columns: bool = False,
    allow_superfluous_columns: bool = False,
    drop_superfluous_columns: bool = False,
    schema_only: bool = False,
    on_collect: bool = False,
    streaming: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Validate the given dataframe.

    Args:
        dataframe: Polars DataFrame or LazyFrame to be validated. A LazyFrame is never
            collected, only the aggregations required to check its content are.
        schema: Patito model which specifies how the dataframe should be structured.
        columns: Optional list of columns to validate. If not provided, all columns
            of the dataframe will be validated.
        allow_missing_columns: If True, missing columns will not be considered an error.
        allow_superfluous_columns: If True, additional columns will not be considered an error.
        drop_superfluous_columns: If True, drop any columns not specified in the schema before validation.
        schema_only: If True, only validate what can be determined from the schema of
            the frame, namely the presence and dtypes of its columns. The content of
            the frame is never read, making validation free for a lazy frame.
        on_collect: If True, attach the content checks to the returned lazy frame
            instead of performing them right away, so that any violation is raised
            when that frame is collected. Requires a LazyFrame.
        streaming: If True, attach the content checks to the returned lazy frame batch
            by batch, so that each batch is checked as it flows through the query and
            the first invalid one raises. Validation then costs no additional pass over
            the data and holds no more than a batch in memory, but the uniqueness
            checks have to be dropped, as a duplicate may be spread across any two
            batches. Requires a LazyFrame.

    Returns:
        The validated data, as a LazyFrame if a LazyFrame was given, and as a DataFrame
        otherwise.

    Raises:
        DataFrameValidationError: If the given dataframe does not match the given
            schema. If ``on_collect`` or ``streaming`` is set, this is instead raised
            when the returned lazy frame is collected.
        ValueError: If the given combination of arguments is contradictory.

    """
    if drop_superfluous_columns and columns:
        raise ValueError(
            "Cannot specify both 'columns' and 'drop_superfluous_columns'."
        )

    deferred = [
        name
        for name, requested in (("on_collect", on_collect), ("streaming", streaming))
        if requested
    ]

    if schema_only and deferred:
        raise ValueError(
            f"Cannot specify both 'schema_only' and {deferred[0]!r}, as schema "
            "validation does not read the content of the frame and is therefore "
            "never deferred."
        )

    if len(deferred) > 1:
        raise ValueError(
            "Cannot specify both 'on_collect' and 'streaming', as they are two "
            "different ways of deferring the very same checks."
        )

    if deferred and not isinstance(dataframe, pl.LazyFrame):
        raise ValueError(
            f"{deferred[0]!r} requires a polars LazyFrame, as there is no collection "
            "to defer the validation to otherwise."
        )

    frame: pl.DataFrame | pl.LazyFrame
    if isinstance(dataframe, pl.LazyFrame):
        frame = dataframe
    elif _PANDAS_AVAILABLE and isinstance(dataframe, pd.DataFrame):
        frame = pl.from_pandas(dataframe)
    else:
        frame = cast(pl.DataFrame, dataframe).clone()

    frame = _transform_frame(frame, schema)

    if drop_superfluous_columns:
        # NOTE: dropping rather than selecting to get the correct error messages
        to_drop = set(_column_names(frame)) - set(schema.columns)
        frame = frame.drop(to_drop)

    if streaming:
        return _attach_streaming_checks(
            frame=cast(pl.LazyFrame, frame),
            schema=schema,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
        )

    checks = _Checks(schema_only=schema_only)
    _register_checks(
        checks=checks,
        dataframe=frame.lazy(),
        schema=schema,
        columns=columns,
        allow_missing_columns=allow_missing_columns,
        allow_superfluous_columns=allow_superfluous_columns,
    )

    if on_collect:
        return checks.attach(cast(pl.LazyFrame, frame), schema)

    errors = checks.resolve()
    if errors:
        raise DataFrameValidationError(errors=errors, model=schema)

    return frame
