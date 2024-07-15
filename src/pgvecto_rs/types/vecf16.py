from struct import pack, unpack

import numpy as np

from pgvecto_rs.errors import NDArrayDimensionError


class Float16Vector:
    def __init__(self, value):
        # asarray still copies if same dtype
        if not isinstance(value, np.ndarray) or value.dtype != "<f2":
            value = np.asarray(value, dtype="<f2")

        if value.ndim != 1:
            raise NDArrayDimensionError(value.ndim)

        self._value = value

    def __repr__(self):
        return f"Float16Vector({self.to_list()})"

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
        return cls([float(v) for v in value[1:-1].split(",")])

    @classmethod
    def from_binary(cls, value):
        dim = unpack("<H", value[:2])[0]
        # start reading buffer from 3th byte (first 2 bytes are for dimension info)
        return cls(
            np.frombuffer(value, dtype="<f2", count=dim, offset=2).astype(np.float16)
        )

    @classmethod
    def _to_db(cls, value, dim=None):
        if value is None:
            return value

        if not isinstance(value, cls):
            value = cls(value)

        if dim is not None and value.dimensions() != dim:
            raise ValueError(
                "expected %d dimensions, not %d" % (dim, value.dimensions())
            )

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
