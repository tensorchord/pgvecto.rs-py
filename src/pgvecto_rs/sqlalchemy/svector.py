from sqlalchemy import types

from pgvecto_rs.errors import SparseDimensionError
from pgvecto_rs.types import SparseVector


class SVECTOR(types.UserDefinedType):
    cache_ok = True

    def __init__(self, dim):
        if dim < 0 or dim > 1048575:  # noqa: PLR2004
            raise SparseDimensionError(dim)
        self.dim = dim

    def get_col_spec(self, **kw):
        if self.dim is None or self.dim == 0:
            return "SVECTOR"
        return f"SVECTOR({self.dim})"

    def bind_processor(self, dialect):
        def _processor(value):
            return SparseVector._to_db(value, self.dim)

        return _processor

    def result_processor(self, dialect, coltype):
        def _processor(value):
            return SparseVector._from_db(value)

        return _processor

    class comparator_factory(types.UserDefinedType.Comparator):  # noqa: N801
        def l2_distance(self, other):
            return self.op("<->", return_type=types.Float)(other)

        def max_inner_product(self, other):
            return self.op("<#>", return_type=types.Float)(other)

        def cosine_distance(self, other):
            return self.op("<=>", return_type=types.Float)(other)
