from sqlalchemy import types

from pgvecto_rs.errors import VectorDimensionError
from pgvecto_rs.types import BinaryVector


class BVECTOR(types.UserDefinedType):
    cache_ok = True

    def __init__(self, dim):
        if dim < 0 or dim > 65535:  # noqa: PLR2004
            raise VectorDimensionError(dim)
        self.dim = dim

    def get_col_spec(self, **kw):
        if self.dim is None or self.dim == 0:
            return "BVECTOR"
        return f"BVECTOR({self.dim})"

    def bind_processor(self, dialect):
        def _processor(value):
            return BinaryVector._to_db(value, self.dim)

        return _processor

    def result_processor(self, dialect, coltype):
        def _processor(value):
            return BinaryVector._from_db(value)

        return _processor

    class comparator_factory(types.UserDefinedType.Comparator):  # noqa: N801
        def l2_distance(self, other):
            return self.op("<->", return_type=types.Float)(other)

        def max_inner_product(self, other):
            return self.op("<#>", return_type=types.Float)(other)

        def cosine_distance(self, other):
            return self.op("<=>", return_type=types.Float)(other)

        def jaccard_distance(self, other):
            return self.op("<~>", return_type=types.Float)(other)
