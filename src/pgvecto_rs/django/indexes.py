from typing import Optional

from django.contrib.postgres.indexes import PostgresIndex

from pgvecto_rs.errors import IndexOptionTypeError
from pgvecto_rs.types import Flat, Hnsw, IndexOption, Ivf, Quantization
from pgvecto_rs.types.index import QuantizationRatio, QuantizationType


class IndexBase(PostgresIndex):
    suffix = "vectors"

    def __init__(
        self,
        *expressions,
        threads: Optional[int] = None,
        quantization_type: Optional[QuantizationType] = None,
        quantization_ratio: Optional[QuantizationRatio] = None,
        **kwargs,
    ):
        self.threads = threads
        self.quantization_type = quantization_type
        self.quantization_ratio = quantization_ratio
        super().__init__(*expressions, **kwargs)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs["threads"] = self.threads
        kwargs["quantization_type"] = self.quantization_type
        kwargs["quantization_ratio"] = self.quantization_ratio
        return path, args, kwargs

    def with_option(self, option: IndexOption):
        self.threads = option.threads
        quantization = option.index.quantization

        self.quantization_type = quantization.type if quantization else None
        self.quantization_ratio = quantization.ratio if quantization else None
        return self


class HnswIndex(IndexBase):
    def __init__(  # noqa: PLR0913
        self,
        *expressions,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        threads: Optional[int] = None,
        quantization_type: Optional[QuantizationType] = None,
        quantization_ratio: Optional[QuantizationRatio] = None,
        **kwargs,
    ):
        self.m = m
        self.ef_construction = ef_construction
        super().__init__(
            *expressions,
            threads=threads,
            quantization_type=quantization_type,
            quantization_ratio=quantization_ratio,
            **kwargs,
        )

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs["m"] = self.m
        kwargs["ef_construction"] = self.ef_construction
        return path, args, kwargs

    def get_with_params(self):
        quant = (
            Quantization(typ=self.quantization_type, ratio=self.quantization_ratio)
            if self.quantization_type
            else None
        )
        option = IndexOption(
            index=Hnsw(
                m=self.m, ef_construction=self.ef_construction, quantization=quant
            ),
            threads=self.threads,
        )
        return [f"options = $${option.dumps()}$$"]

    def with_option(self, option: IndexOption):
        """
        Fill a partially initialized HnswIndex object with option, override arguments:
        - m
        - ef_construction
        - threads
        - quantization_type
        - quantization_ratio
        """
        if not isinstance(option.index, Hnsw):
            raise IndexOptionTypeError(Hnsw, type(option.index))
        self.m = option.index.m
        self.ef_construction = option.index.ef_construction
        super().with_option(option)
        return self


class IvfIndex(IndexBase):
    def __init__(
        self,
        *expressions,
        nlist: Optional[int] = None,
        threads: Optional[int] = None,
        quantization_type: Optional[QuantizationType] = None,
        quantization_ratio: Optional[QuantizationRatio] = None,
        **kwargs,
    ):
        self.nlist = nlist
        super().__init__(
            *expressions,
            threads=threads,
            quantization_type=quantization_type,
            quantization_ratio=quantization_ratio,
            **kwargs,
        )

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs["nlist"] = self.nlist
        return path, args, kwargs

    def get_with_params(self):
        quant = (
            Quantization(typ=self.quantization_type, ratio=self.quantization_ratio)
            if self.quantization_type
            else None
        )
        option = IndexOption(
            index=Ivf(nlist=self.nlist, quantization=quant),
            threads=self.threads,
        )
        return [f"options = $${option.dumps()}$$"]

    def with_option(self, option: IndexOption):
        """
        Fill a partially initialized IvfIndex object with option, override arguments:
        - nlist
        - threads
        - quantization_type
        - quantization_ratio
        """
        if not isinstance(option.index, Ivf):
            raise IndexOptionTypeError(Ivf, type(option.index))
        self.nlist = option.index.nlist
        super().with_option(option)
        return self


class FlatIndex(IndexBase):
    def __init__(
        self,
        *expressions,
        threads: Optional[int] = None,
        quantization_type: Optional[QuantizationType] = None,
        quantization_ratio: Optional[QuantizationRatio] = None,
        **kwargs,
    ):
        super().__init__(
            *expressions,
            threads=threads,
            quantization_type=quantization_type,
            quantization_ratio=quantization_ratio,
            **kwargs,
        )

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        return path, args, kwargs

    def get_with_params(self):
        quant = (
            Quantization(typ=self.quantization_type, ratio=self.quantization_ratio)
            if self.quantization_type
            else None
        )
        option = IndexOption(
            index=Flat(quantization=quant),
            threads=self.threads,
        )
        return [f"options = $${option.dumps()}$$"]

    def with_option(self, option: IndexOption):
        """
        Fill a partially initialized FlatIndex object with option, override arguments:
        - threads
        - quantization_type
        - quantization_ratio
        """
        if not isinstance(option.index, Flat):
            raise IndexOptionTypeError(Flat, type(option.index))
        super().with_option(option)
        return self
