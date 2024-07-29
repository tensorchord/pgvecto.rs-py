from typing import Tuple


class PGVectoRsError(ValueError):
    pass


class NDArrayDimensionError(PGVectoRsError):
    def __init__(self, dim: int) -> None:
        super().__init__(f"ndarray must be 1D for vector, got {dim}D")


class SparseExtraArgError(PGVectoRsError):
    def __init__(self, dtype: type, dim: int) -> None:
        super().__init__(
            f"sparse array don't need dimension for {dtype} input, but got argument dim={dim}D"
        )


class SparseMissingArgError(PGVectoRsError):
    def __init__(self, dtype: type) -> None:
        super().__init__(
            f"sparse array need dimension for {dtype} input, but got argument dim=None"
        )


class SparseShapeError(PGVectoRsError):
    def __init__(self, shape: Tuple[int, int]) -> None:
        super().__init__(f"sparse array must be (n,) or (1, n) for vector, got {shape}")


class VectorDimensionError(PGVectoRsError):
    def __init__(self, dim: int) -> None:
        super().__init__(f"vector dimension must be > 0 and < 65536, got {dim}")


class SparseDimensionError(PGVectoRsError):
    def __init__(self, dim: int) -> None:
        super().__init__(
            f"sparse vector dimension must be > 0 and < 1048576, got {dim}"
        )


class SparseDimUnequalError(PGVectoRsError):
    def __init__(self, indices_len: int, values_len: int) -> None:
        super().__init__(
            f"sparse vector expected indices length {indices_len} to match values length {values_len}"
        )


class ToDBDimUnequalError(PGVectoRsError):
    def __init__(
        self,
        arg_dim: int,
        value_dim: int,
    ) -> None:
        super().__init__(f"expected {arg_dim} dimensions, not {value_dim}")


class TypeNotFoundError(PGVectoRsError):
    def __init__(self, vtype: str) -> None:
        super().__init__(f"{vtype} type not found in the database")


class TextParseError(PGVectoRsError):
    def __init__(self, payload: str, dtype: type) -> None:
        super().__init__(f"failed to parse text of '{payload}' as a {dtype}")
