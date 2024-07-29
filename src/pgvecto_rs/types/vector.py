from struct import pack, unpack

import numpy as np

from pgvecto_rs.errors import NDArrayDimensionError, ToDBDimUnequalError


class Vector:
    def __init__(self, value):
        # asarray still copies if same dtype
        if not isinstance(value, np.ndarray) or value.dtype != "<f4":
            value = np.asarray(value, dtype="<f4")

        if value.ndim != 1:
            raise NDArrayDimensionError(value.ndim)

        self._value = value

    def __repr__(self):
        return f"Vector({self.to_list()})"

    def dimensions(self):
        return len(self._value)

    def to_list(self):
        return self._value.tolist()

    def to_numpy(self):
        return self._value

    def to_text(self):
        return "[" + ",".join([str(float(v)) for v in self._value]) + "]"

    def to_binary(self):
        # pack to little-endian uint16, keep same endian with pgvecto.rs
        dims: bytes = pack("<H", self._value.shape[0])
        return dims + self._value.tobytes()

    @classmethod
    def from_text(cls, value):
        left, right = value.find("["), value.rfind("]")
        if left == -1 or right == -1 or left > right:
            raise ValueError
        return cls([float(v) for v in value[left + 1 : right].split(",")])

    @classmethod
    def from_binary(cls, value):
        view = memoryview(value)
        dim = unpack("<H", view[:2])[0]
        # start reading buffer from 3th byte (first 2 bytes are for dimension info)
        return cls(
            np.frombuffer(view, dtype="<f", count=dim, offset=2).astype(np.float32)
        )

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
