import numpy as np
import pytest
from sqlalchemy import Index, Integer, create_engine, delete, insert, select, text
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from pgvecto_rs.sqlalchemy import BVECTOR, SVECTOR, VECF16, VECTOR
from tests import (
    BINARY_VECTORS,
    COSINE_DIS_OP,
    FILTER_VALUE,
    FLOAT16_OP,
    FLOAT16_VECTORS,
    INDEX_OPTIONS,
    INVALID_VECTORS,
    JACCARD_DIS_OP,
    L2_DIS_OP,
    MAX_INNER_PROD_OP,
    SPARSE_OP,
    SPARSE_VECTORS,
    URL,
    VECTORS,
    cosine_distance,
    jaccard_distance,
    l2_distance,
    max_inner_product,
)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "tb_test_item"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    embedding: Mapped[np.ndarray] = mapped_column(VECTOR(3))
    sparse_embedding = mapped_column(SVECTOR(3), nullable=True)
    float16_embedding = mapped_column(VECF16(3), nullable=True)
    binary_embedding = mapped_column(BVECTOR(3), nullable=True)


@pytest.fixture(scope="module")
def session():
    """Connect to the test db pointed by the URL. Can check more details
    in `tests/__init__.py`
    """
    engine = create_engine(URL.replace("postgresql", "postgresql+psycopg"))

    # ensure that we have installed pgvector.rs extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vectors"))
        conn.execute(text("DROP TABLE IF EXISTS tb_test_item"))
        conn.commit()

    with Session(engine) as session:
        Document.metadata.create_all(engine)
        create_items(session)
        try:
            yield session
        finally:
            session.rollback()
            Document.metadata.drop_all(engine)


def create_items(session: Session):
    data = [
        insert(Document).values(
            id=i,
            embedding=v,
            sparse_embedding=sv,
            float16_embedding=f16v,
            binary_embedding=bv,
        )
        for i, (v, sv, f16v, bv) in enumerate(
            zip(VECTORS, SPARSE_VECTORS, FLOAT16_VECTORS, BINARY_VECTORS)
        )
    ]
    for stat in data:
        session.execute(stat)
    session.commit()
    for row in session.scalars(select(Document)):
        assert np.allclose(row.embedding.to_numpy(), VECTORS[row.id], atol=1e-10)


# =================================
# Prefix functional tests
# =================================


@pytest.mark.parametrize(("index_name", "index_option"), INDEX_OPTIONS.items())
def test_create_index(session: Session, index_name: str, index_option: str):
    index = Index(
        index_name,
        Document.embedding,
        postgresql_using="vectors",
        postgresql_with={"options": f"$${index_option}$$"},
        postgresql_ops={"embedding": "vector_l2_ops"},
    )
    index.create(session.bind)
    session.commit()


@pytest.mark.parametrize(("i", "e"), enumerate(INVALID_VECTORS))
def test_invalid_insert(session: Session, i: int, e: np.array):
    try:
        session.execute(insert(Document).values(id=i, embedding=e))
    except StatementError:
        pass
    else:
        raise AssertionError(  # noqa: TRY003
            f"failed to raise invalid value error for {i}th vector {e}",
        )
    finally:
        session.rollback()


# =================================
# Semantic search tests
# =================================


def test_l2_distance(session: Session):
    for row in session.execute(
        select(
            Document.embedding,
            Document.embedding.l2_distance(L2_DIS_OP),
        ),
    ):
        (emb, dis) = row
        expect = l2_distance(np.array(L2_DIS_OP), emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_max_inner_product(session: Session):
    for row in session.execute(
        select(
            Document.embedding,
            Document.embedding.max_inner_product(MAX_INNER_PROD_OP),
        ),
    ):
        (emb, dis) = row
        expect = max_inner_product(np.array(MAX_INNER_PROD_OP), emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_cosine_distance(session: Session):
    for row in session.execute(
        select(
            Document.embedding,
            Document.embedding.cosine_distance(COSINE_DIS_OP),
        ),
    ):
        (emb, dis) = row
        expect = cosine_distance(np.array(COSINE_DIS_OP), emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_binary_jaccard_distance(session):
    for row in session.execute(
        select(
            Document.binary_embedding,
            Document.binary_embedding.jaccard_distance(JACCARD_DIS_OP),
        ),
    ):
        (emb, dis) = row
        expect = jaccard_distance(JACCARD_DIS_OP, emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_float16_vector(session):
    for row in session.execute(
        select(
            Document.float16_embedding,
            Document.float16_embedding.l2_distance(FLOAT16_OP),
        ),
    ):
        (emb, dis) = row
        expect = l2_distance(FLOAT16_OP, emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-2)


def test_sparse_vector(session):
    for row in session.execute(
        select(
            Document.sparse_embedding,
            Document.sparse_embedding.l2_distance(SPARSE_OP),
        ),
    ):
        (emb, dis) = row
        expect = l2_distance(SPARSE_OP.to_numpy(), emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_filter(session):
    for row in session.execute(
        select(
            Document.embedding.l2_distance(L2_DIS_OP),
        ).filter(Document.embedding.l2_distance(L2_DIS_OP) < FILTER_VALUE),
    ):
        (dis,) = row
        assert dis < FILTER_VALUE


# =================================
# Suffix functional tests
# =================================


def test_clean(session: Session):
    session.execute(delete(Document).where(Document.embedding == VECTORS[0]))
    session.commit()
    res = session.execute(select(Document))
    assert len(list(res)) == len(VECTORS) - 1
