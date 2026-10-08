from __future__ import annotations

import ast
import inspect
import io
import json
from typing import Annotated, Any, Optional, Union

import polars as pl
import polars.datatypes
from polars.datatypes import DataType, DataTypeClass
from polars.exceptions import ComputeError
from pydantic import BaseModel, BeforeValidator, field_serializer

_DTYPES: dict[str, DataTypeClass] = {
    name: obj
    for name, obj in vars(polars.datatypes).items()
    if inspect.isclass(obj) and issubclass(obj, DataType)
}


def _parse_dtype_node(node: ast.expr) -> Any:
    """Evaluate the subset of Python syntax produced by ``str(dtype)``.

    Only polars dtype names, calls to them, and literals are accepted, so that
    deserializing untrusted json can never execute arbitrary code.
    """
    if isinstance(node, ast.Name):
        if node.id in _DTYPES:
            return _DTYPES[node.id]
        raise ValueError(f"Unknown polars dtype {node.id!r}.")
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_parse_dtype_node(arg) for arg in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("Unpacking is not supported in dtypes.")
            kwargs[kw.arg] = _parse_dtype_node(kw.value)
        return _parse_dtype_node(node.func)(*args, **kwargs)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        return [_parse_dtype_node(e) for e in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_parse_dtype_node(e) for e in node.elts)
    elif isinstance(node, ast.Dict):
        items = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise ValueError("Unpacking is not supported in dtypes.")
            items[_parse_dtype_node(key)] = _parse_dtype_node(value)
        return items
    raise ValueError(f"Unsupported syntax in dtype: {ast.dump(node)}")


def dtype_deserializer(dtype: str | DataTypeClass | DataType | None):
    """Deserialize a dtype from json."""
    if isinstance(dtype, DataTypeClass) or isinstance(dtype, DataType):
        return dtype
    else:
        if dtype == "null" or dtype is None:
            return None
        else:
            try:
                tree = ast.parse(dtype, mode="eval")
            except SyntaxError as e:
                raise ValueError(f"{dtype!r} is not a valid polars dtype.") from e
            return _parse_dtype_node(tree.body)


def expr_deserializer(
    expr: str | pl.Expr | list[pl.Expr] | None,
) -> pl.Expr | list[pl.Expr] | None:
    """Deserialize a polars expression or list thereof from json.

    This is applied both during deserialization and validation.
    """
    if expr is None:
        return None
    elif isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, list):
        return expr
    elif isinstance(expr, str):
        if expr == "null":
            return None
        # can be either a list of expr or expr
        elif expr[0] == "[":
            return [
                pl.Expr.deserialize(io.StringIO(e), format="json")
                for e in json.loads(expr)
            ]
        else:
            return pl.Expr.deserialize(io.StringIO(expr), format="json")
    else:
        raise ValueError(f"{expr} can not be deserialized.")


def expr_or_col_name_deserializer(expr: str | pl.Expr | None) -> pl.Expr | str | None:
    """Deserialize a polars expression or column name from json.

    This is applied both during deserialization and validation.
    """
    if expr is None:
        return None
    elif isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, list):
        return expr
    elif isinstance(expr, str):
        # Default behaviour
        if expr == "null":
            return None
        else:
            try:
                return pl.Expr.deserialize(io.StringIO(expr), format="json")
            except ComputeError:
                try:
                    # Column name is being deserialized
                    return json.loads(expr)
                except json.JSONDecodeError:
                    # Column name has been passed literally
                    # to ColumnInfo(derived_from="foo")
                    return expr
    else:
        raise ValueError(f"{expr} can not be deserialized.")


class ColumnInfo(BaseModel, arbitrary_types_allowed=True):
    """patito-side model for storing column metadata.

    Args:
        allow_missing (bool): Column may be missing.
        constraints (Union[polars.Expression, List[polars.Expression]): A single
            constraint or list of constraints, expressed as a polars expression objects.
            All rows must satisfy the given constraint. You can refer to the given column
            with ``pt.field``, which will automatically be replaced with
            ``polars.col(<field_name>)`` before evaluation.
        derived_from (Union[str, polars.Expr]): used to mark fields that are meant to be derived from other fields. Users can specify a polars expression that will be called to derive the column value when `pt.DataFrame.derive` is called.
        dtype (polars.datatype.DataType): The given dataframe column must have the given
            polars dtype, for instance ``polars.UInt64`` or ``pl.Float32``.
        unique (bool): All row values must be unique.
        primary_key (bool): The field is part of the primary keys of the Model.

    """

    allow_missing: Optional[bool] = None
    dtype: Annotated[
        Optional[Union[DataTypeClass, DataType]],
        BeforeValidator(dtype_deserializer),
    ] = None
    constraints: Annotated[
        Optional[Union[pl.Expr, list[pl.Expr]]],
        BeforeValidator(expr_deserializer),
    ] = None
    derived_from: Annotated[
        Optional[Union[str, pl.Expr]],
        BeforeValidator(expr_or_col_name_deserializer),
    ] = None
    unique: Optional[bool] = None
    primary_key: bool = False

    def __repr__(self) -> str:
        """Print only Field attributes whose values are not default (mainly None)."""
        not_default_field = {
            field: getattr(self, field)
            for field in self.model_fields
            if getattr(self, field) is not self.model_fields[field].default
        }

        string = ""
        for field, value in not_default_field.items():
            string += f"{field}={value}, "
        if string:
            # remove trailing comma and space
            string = string[:-2]
        return f"ColumnInfo({string})"

    @field_serializer("constraints", "derived_from")
    def expr_serializer(self, expr: None | pl.Expr | list[pl.Expr]):
        """Converts polars expr to json."""
        if expr is None:
            return "null"
        elif isinstance(expr, str):
            return json.dumps(expr)
        elif isinstance(expr, list):
            return json.dumps([e.meta.serialize(format="json") for e in expr])
        else:
            return expr.meta.serialize(format="json")

    @field_serializer("dtype")
    def dtype_serializer(self, dtype: DataTypeClass | DataType | None) -> str:
        """Converts polars dtype to json."""
        if dtype is None:
            return "null"
        else:
            return str(dtype)
