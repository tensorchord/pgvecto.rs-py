import os
from typing import Any, Literal, Optional, Union

import toml


class Quantization:
    QuantizationType = Literal["trivial", "scalar", "product"]
    QuantizationRatio = Optional[Literal["x4", "x8", "x16", "x32", "x64"]]

    def __init__(
        self, typ: QuantizationType = "trivial", ratio: QuantizationRatio = None
    ) -> None:
        self.type = typ
        self.ratio = ratio

    def dump(self) -> dict:
        if self.type == "trivial":
            return {"quantization": {"trivial": {}}}
        elif self.type == "scalar":
            return {"quantization": {"scalar": {}}}
        else:
            return {"quantization": {"product": {"ratio": self.ratio}}}


class Flat:
    def __init__(self, quantization: Optional[Quantization] = None):
        self.quantization = quantization

    def dump(self) -> dict:
        child: dict[str, Any] = {}
        if self.quantization is not None:
            child.update(self.quantization.dump())
        return {"flat": child}


class Hnsw:
    def __init__(
        self,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        quantization: Optional[Quantization] = None,
    ):
        self.m = m
        self.ef_construction = ef_construction
        self.quantization = quantization

    def dump(self) -> dict:
        child: dict[str, Any] = {}
        if self.quantization is not None:
            child.update(self.quantization.dump())
        if self.m is not None:
            child.update({"m": self.m})
        if self.ef_construction is not None:
            child.update({"ef_construction": self.ef_construction})
        return {"hnsw": child}


class Ivf:
    def __init__(
        self, nlist: Optional[int] = None, quantization: Optional[Quantization] = None
    ):
        self.nlist = nlist
        self.quantization = quantization

    def dump(self) -> dict:
        child: dict[str, Any] = {}
        if self.quantization is not None:
            child.update(self.quantization.dump())
        if self.nlist is not None:
            child.update({"nlist": self.nlist})
        return {"ivf": child}


class IndexOption:
    def __init__(
        self,
        index: Union[Hnsw, Ivf, Flat],
        threads: Optional[int] = None,
        quantization: Optional[Quantization] = None,
    ):
        self.index = index
        self.threads = threads or os.cpu_count() or 1
        self.quantization = quantization

    def dump(self) -> dict:
        return {
            "indexing": self.index.dump(),
            "optimizing": {"optimizing_threads": self.threads},
        }

    def dumps(self) -> str:
        return toml.dumps(self.dump())