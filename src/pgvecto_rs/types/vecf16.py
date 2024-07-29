from struct import unpack

import numpy as np

from pgvecto_rs.errors import NDArrayDimensionError
from pgvecto_rs.types.vector import Vector


class Float16Vector(Vector):
    def __init__(self, value):
        # asarray still copies if same dtype
        if not isinstance(value, np.ndarray) or value.dtype != "<f2":
            value = np.asarray(value, dtype="<f2")

        if value.ndim != 1:
            raise NDArrayDimensionError(value.ndim)

        self._value = value

    def __repr__(self):
        return f"Float16Vector({self.to_list()})"

    @classmethod
    def from_binary(cls, value):
        dim = unpack("<H", value[:2])[0]
        # start reading buffer from 3th byte (first 2 bytes are for dimension info)
        return cls(
            np.frombuffer(value, dtype="<f2", count=dim, offset=2).astype(np.float16)
        )
