from typing import Callable, List

import numpy as np
import pytest

from pgvecto_rs.sdk import Filter, PGVectoRs, Record, filters
from tests import (
    COSINE_DIS_OP,
    L2_DIS_OP,
    MAX_INNER_PROD_OP,
    URL,
    VECTORS,
    cosine_distance,
    l2_distance,
    max_inner_product,
)

URL = URL.replace("postgresql", "postgresql+psycopg")
MockTexts = {
    "text0": VECTORS[0],
    "text1": VECTORS[1],
    "text2": VECTORS[2],
}


class MockEmbedder:
    def embed(self, text: str) -> np.ndarray:
        if isinstance(MockTexts[text], list):
            return np.array(MockTexts[text], dtype=np.float32)
        return MockTexts[text]


@pytest.fixture(scope="module")
def client():
    client = PGVectoRs(db_url=URL, collection_name="empty", dimension=3, recreate=True)
    records1 = [Record.from_text(t, v, {"src": "src1"}) for t, v in MockTexts.items()]
    records2 = [Record.from_text(t, v, {"src": "src2"}) for t, v in MockTexts.items()]
    client.insert(records1)
    client.insert(records2)
    return client


filter_src1 = filters.meta_contains({"src": "src1"})
filter_src2: Filter = lambda r: r.meta.contains({"src": "src2"})


@pytest.mark.parametrize("filter", [filter_src1, filter_src2])
@pytest.mark.parametrize(
    ("dis_op", "dis_oprand", "assert_func"),
    zip(
        ["<->", "<#>", "<=>"],
        [L2_DIS_OP, MAX_INNER_PROD_OP, COSINE_DIS_OP],
        [l2_distance, max_inner_product, cosine_distance],
    ),
)
def test_search_filter_and_op(
    client: PGVectoRs,
    filter: Filter,
    dis_op: str,
    dis_oprand: List[float],
    assert_func: List[Callable],
):
    for rec, dis in client.search(dis_oprand, dis_op, top_k=99, filter=filter):
        expect = assert_func(dis_oprand, rec.embedding.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


@pytest.mark.parametrize(
    ("dis_op", "dis_oprand", "assert_func"),
    zip(
        ["<->", "<#>", "<=>"],
        [L2_DIS_OP, MAX_INNER_PROD_OP, COSINE_DIS_OP],
        [l2_distance, max_inner_product, cosine_distance],
    ),
)
def test_search_order_and_limit(
    client: PGVectoRs,
    dis_op: str,
    dis_oprand: List[float],
    assert_func: List[Callable],
):
    for rec, dis in client.search(dis_oprand, dis_op, top_k=4):
        expect = assert_func(dis_oprand, rec.embedding.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)
