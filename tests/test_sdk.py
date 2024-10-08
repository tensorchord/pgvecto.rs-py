import time
from typing import Callable, List

import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from pgvecto_rs.sdk import Filter, PGVectoRs, Record, filters
from pgvecto_rs.sdk.record import Column, Unique
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


def test_unique_text_table(
    client: PGVectoRs,
):
    unique_client = PGVectoRs(
        db_url=URL,
        collection_name="unique_text",
        dimension=3,
        recreate=True,
        constraints=[Unique(columns=[Column.TEXT])],
    )
    it = iter(MockTexts.items())
    text1, vector1 = next(it)
    _, vector2 = next(it)
    records_ok = [Record.from_text(t, v, {"src": "src1"}) for t, v in MockTexts.items()]
    records_fail = [
        Record.from_text(text1, vector1, {"src": "src1"}),
        Record.from_text(text1, vector2, {"src": "src2"}),
    ]
    unique_client.insert(records_ok)
    unique_client.delete_all()
    with pytest.raises(IntegrityError):
        unique_client.insert(records_fail)


def test_unique_meta_table(
    client: PGVectoRs,
):
    unique_client = PGVectoRs(
        db_url=URL,
        collection_name="unique_meta",
        dimension=3,
        recreate=True,
        constraints=[Unique(columns=[Column.META])],
    )
    it = iter(MockTexts.items())
    text1, vector1 = next(it)
    text2, vector2 = next(it)
    records_ok = [
        Record.from_text(text1, vector1, {"src": "src1"}),
        Record.from_text(text2, vector2, {"src": "src2"}),
    ]
    records_fail = [
        Record.from_text(text1, vector1, {"src": "src1"}),
        Record.from_text(text2, vector2, {"src": "src1"}),
    ]
    unique_client.insert(records_ok)
    unique_client.delete_all()
    with pytest.raises(IntegrityError):
        unique_client.insert(records_fail)


def test_unique_text_meta_table(
    client: PGVectoRs,
):
    unique_client = PGVectoRs(
        db_url=URL,
        collection_name="unique_both",
        dimension=3,
        recreate=True,
        constraints=[Unique(columns=[Column.TEXT, Column.META])],
    )
    it = iter(MockTexts.items())
    text1, vector1 = next(it)
    text2, vector2 = next(it)
    records_ok = [
        Record.from_text(text1, vector1, {"src": "src1"}),
        Record.from_text(text2, vector2, {"src": "src1"}),
    ]
    records_fail = [
        Record.from_text(text1, vector1, {"src": "src1"}),
        Record.from_text(text1, vector2, {"src": "src1"}),
    ]
    unique_client.insert(records_ok)
    unique_client.delete_all()
    with pytest.raises(IntegrityError):
        unique_client.insert(records_fail)


COUNT = 1000


def test_count_table(
    client: PGVectoRs,
):
    count_client = PGVectoRs(
        db_url=URL,
        collection_name="count",
        dimension=3,
        recreate=True,
    )
    it = iter(MockTexts.items())
    text1, vector1 = next(it)
    records = [Record.from_text(text1, vector1, {"src": "src1"}) for _ in range(COUNT)]
    count_client.insert(records)

    rows = count_client.row_count(estimate=False)
    assert rows == COUNT

    rows = count_client.row_count(estimate=False, filter=filter_src2)
    assert rows == 0

    for _ in range(90):
        estimate_rows = count_client.row_count(estimate=True)
        if estimate_rows == COUNT:
            return
        time.sleep(1)
    raise AssertionError
