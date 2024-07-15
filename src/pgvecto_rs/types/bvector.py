import math
from struct import pack, unpack

import numpy as np

from pgvecto_rs.errors import NDArrayDimensionError


def packbits_u64(arr: np.ndarray):
    packed_size = math.ceil(len(arr) / 64)
    packed_arr = np.zeros(packed_size, dtype=np.uint64)
    for i, x in enumerate(arr):
        if x:
            # |= is forbidden for uint64: https://github.com/numpy/numpy/issues/22624
            packed_arr[i // 64] += 1 << (i % 64)
    return packed_arr


def unpackbits_u64(dim: int, packed_arr: np.ndarray):
    unpacked_arr = np.zeros(dim, dtype=bool)
    for i, x in enumerate(packed_arr):
        for j in range(64):
            unpacked_arr[i * 64 + j] = (x >> j) & 1
    return unpacked_arr


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
        data = packbits_u64(self._value).astype(dtype="<u8")
        return dims + data.tobytes()

    @classmethod
    def from_text(cls, value):
        return cls([int(v) for v in value[1:-1].split(",")])

    @classmethod
    def from_binary(cls, value):
        dim = unpack("<H", value[:2])[0]
        data = np.frombuffer(value, dtype="<u8", count=dim // 64, offset=2)

        # start reading buffer from 3th byte (first 2 bytes are for dimension info)
        return cls(unpackbits_u64(dim, data))

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
