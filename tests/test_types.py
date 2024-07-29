# TODO: remove after Python < 3.9 is no longer used
from __future__ import annotations

import pytest

from pgvecto_rs.types import IndexOption, SparseVector, Vector
from tests import EQUAL_SPARSE_VECTORS, EQUAL_VECTORS, INDEX_OPTION_DUMPS


def test_vector_equal():
    data: list[Vector] = []
    for raw in EQUAL_VECTORS:
        if isinstance(raw, Vector):
            data.append(raw.to_text())
        else:
            data.append(Vector(*raw).to_text())
    unique = set(data)
    assert len(unique) == 1


def test_sparse_equal():
    data: list[SparseVector] = []
    for raw in EQUAL_SPARSE_VECTORS:
        data.append(raw.to_text())
    unique = set(data)
    assert len(unique) == 1


@pytest.mark.parametrize(("inp", "out"), INDEX_OPTION_DUMPS)
def test_index_option_dump(inp: IndexOption, out: str):
    assert inp.dumps() == out
