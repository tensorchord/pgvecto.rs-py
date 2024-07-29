from django.db.models import FloatField, Func, Value

from pgvecto_rs.types import BinaryVector, Float16Vector, SparseVector, Vector


class DistanceBase(Func):
    output_field = FloatField()

    def __init__(self, expression, vector, **extra):
        if not hasattr(vector, "resolve_expression"):
            if isinstance(vector, Float16Vector):
                vector = Value(Float16Vector._to_db(vector))
            elif isinstance(vector, SparseVector):
                vector = Value(SparseVector._to_db(vector))
            elif isinstance(vector, BinaryVector):
                vector = Value(BinaryVector._to_db(vector))
            else:
                vector = Value(Vector._to_db(vector))
        super().__init__(expression, vector, **extra)


class L2Distance(DistanceBase):
    function = ""
    arg_joiner = " <-> "


class MaxInnerProduct(DistanceBase):
    function = ""
    arg_joiner = " <#> "


class CosineDistance(DistanceBase):
    function = ""
    arg_joiner = " <=> "


class JaccardDistance(DistanceBase):
    function = ""
    arg_joiner = " <~> "
