import math
from struct import pack, unpack

import numpy as np

from pgvecto_rs.errors import NDArrayDimensionError, TextParseError, ToDBDimUnequalError


class BinaryVector:
    def __init__(self, value):
        if not isinstance(value, np.ndarray) or value.dtype != bool:
            value = np.asarray(value, dtype=bool)

        if value.ndim != 1:
            raise NDArrayDimensionError(value.ndim)

        self._value = value

    def __repr__(self):
        return f"BinaryVector({self.to_list()})"

    def dimensions(self):
        return len(self._value)

    def to_list(self):
        return self._value.tolist()

    def to_numpy(self):
        return self._value

    def to_text(self):
        return "[" + ",".join([str(int(v)) for v in self._value]) + "]"

    def to_binary(self):
        # pack to little-endian uint16, keep same endian with pgvecto.rs
        dims: bytes = pack("<H", self._value.shape[0])
        pad_width = (64 - self._value.shape[0] % 64) % 64
        padded = np.pad(self._value, (0, pad_width), "constant")
        data = np.packbits(padded, bitorder="little").view(np.uint64)

        return dims + data.tobytes()

    @classmethod
    def from_text(cls, value):
        left, right = value.find("["), value.rfind("]")
        if left == -1 or right == -1 or left > right:
            raise TextParseError(value, cls)
        return cls([int(v) for v in value[left + 1 : right].split(",")])

    @classmethod
    def from_binary(cls, value):
        view = memoryview(value)
        # start reading buffer from 3th byte (first 2 bytes are for dimension info)
        dim = unpack("<H", view[:2])[0]
        length = math.ceil(dim / 64)
        data = np.frombuffer(view, dtype="<u8", count=length, offset=2).view(np.uint8)
        return cls(np.unpackbits(data, bitorder="little", count=dim))

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
