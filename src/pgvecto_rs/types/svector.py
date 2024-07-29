# TODO: remove after Python < 3.9 is no longer used
from __future__ import annotations

from struct import pack, unpack
from typing import Union

import numpy as np

from pgvecto_rs.errors import (
    SparseDimUnequalError,
    SparseExtraArgError,
    SparseMissingArgError,
    SparseShapeError,
    TextParseError,
    ToDBDimUnequalError,
)


class NoDefault:
    pass


NO_DEFAULT = NoDefault()


class SparseVector:
    def __init__(self, value, dimensions=NO_DEFAULT, /):
        if value.__class__.__module__.startswith("scipy.sparse."):
            if not isinstance(dimensions, NoDefault):
                raise SparseExtraArgError(type(value), dimensions)

            self._from_sparse(value)
        elif isinstance(value, dict):
            if isinstance(dimensions, NoDefault):
                raise SparseMissingArgError(dict)

            self._from_dict(value, dimensions)
        else:
            if not isinstance(dimensions, NO_DEFAULT):
                raise SparseExtraArgError(type(value), dimensions)

            self._from_dense(value)

    @classmethod
    def from_parts(
        cls,
        dim: int,
        indices: Union[list[int], np.array],
        values: Union[list[float], np.array],
    ):
        return cls._from_parts(dim, [v for v in indices], [v for v in values])

    def __repr__(self):
        elements = dict(zip(self._indices, self._values))
        return f"SparseVector({elements}, {self._dim})"

    def dimensions(self):
        return self._dim

    def indices(self):
        return self._indices

    def values(self):
        return self._values

    def to_coo(self):
        from scipy.sparse import coo_array

        coords = ([0] * len(self._indices), self._indices)
        return coo_array((self._values, coords), shape=(1, self._dim))

    def to_list(self):
        vec = [0.0] * self._dim
        for i, v in zip(self._indices, self._values):
            vec[i] = v
        return vec

    def to_numpy(self):
        vec = np.zeros(self._dim).astype(np.float32)
        vec[self._indices] = self._values
        return vec

    def to_text(self):
        return (
            "{"
            + ",".join(
                [f"{int(i)}:{float(v)}" for i, v in zip(self._indices, self._values)]
            )
            + "}/"
            + str(int(self._dim))
        )

    def to_binary(self):
        # convert indices to little-endian uint32
        indices = np.asarray(self._indices, dtype="<I")
        indices_len = indices.shape[0]
        indices_bytes = indices.tobytes()
        # convert values to little-endian float32
        values = np.asarray(self._values, dtype="<f")
        values_len = values.shape[0]
        values_bytes = values.tobytes()
        # check indices and values length is the same
        if indices_len != values_len:
            raise SparseDimUnequalError(indices_len, values_len)
        return (
            pack("<I", self._dim)
            + pack("<I", indices_len)
            + indices_bytes
            + values_bytes
        )

    def _from_dict(self, d, dim):
        elements = [(i, v) for i, v in d.items() if v != 0]
        elements.sort()

        self._dim = int(dim)
        self._indices = [int(v[0]) for v in elements]
        self._values = [float(v[1]) for v in elements]

    def _from_sparse(self, value):
        value = value.tocoo()

        if value.ndim == 1:
            self._dim = value.shape[0]
        elif value.ndim == 2 and value.shape[0] == 1:  # noqa: PLR2004
            self._dim = value.shape[1]
        else:
            raise SparseShapeError(value.shape)

        if hasattr(value, "coords") and value.ndim == 1:
            # scipy > 1.13
            self._indices = value.coords[0].tolist()
        elif hasattr(value, "coords") and value.ndim == 2:  # noqa: PLR2004
            # scipy > 1.13
            self._indices = value.coords[1].tolist()
        else:
            self._indices = value.col.tolist()
        self._values = value.data.tolist()

    def _from_dense(self, value):
        self._dim = len(value)
        self._indices = [i for i, v in enumerate(value) if np.isclose(v, 0)]
        self._values = [float(value[i]) for i in self._indices]

    @classmethod
    def from_text(cls, value: str):
        elements, dim = value.split("/", 2)
        left, right = elements.find("{"), elements.rfind("}")
        if left == -1 or right == -1 or left > right:
            raise TextParseError(value, cls)
        indices = []
        values = []
        for e in elements[left + 1 : right].split(","):
            i, v = e.split(":", 2)
            indices.append(int(i))
            values.append(float(v))
        return cls._from_parts(int(dim), indices, values)

    @classmethod
    def from_binary(cls, value):
        view = memoryview(value)
        # unpack dims and length as little-endian uint32, keep same endian with pgvecto.rs
        dims = unpack("<I", view[:4])[0]
        length = unpack("<I", view[4:8])[0]
        bytes = view[8:]
        # unpack indices and values as little-endian uint32 and float32, keep same endian with pgvecto.rs
        indices = np.frombuffer(bytes, dtype="<I", count=length, offset=0).astype(
            np.uint32
        )
        values = np.frombuffer(
            bytes, dtype="<f", count=length, offset=4 * length
        ).astype(np.float32)
        return cls.from_parts(dims, indices, values)

    @classmethod
    def _from_parts(cls, dim, indices, values):
        vec = cls.__new__(cls)
        vec._dim = dim
        vec._indices = indices
        vec._values = values
        return vec

    @classmethod
    def _to_db(cls, value, dim=None):
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)

        if dim is not None and value.dimensions() != dim:
            raise ToDBDimUnequalError(dim, value.dimensions())

        return value.to_text()

    @classmethod
    def _to_db_binary(cls, value):
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)

        return value.to_binary()

    @classmethod
    def _from_db(cls, value):
        if value is None or isinstance(value, cls):
            return value

        return cls.from_text(value)

    @classmethod
    def _from_db_binary(cls, value):
        if value is None or isinstance(value, cls):
            return value

        return cls.from_binary(value)
