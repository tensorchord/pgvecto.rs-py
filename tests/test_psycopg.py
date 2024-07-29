import numpy as np
import psycopg
import pytest
from psycopg import Connection, sql

from pgvecto_rs.psycopg import register_vector
from pgvecto_rs.types import BinaryVector, Float16Vector
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


@pytest.fixture()
def session():
    with psycopg.connect(URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vectors;")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS tb_test_item;")
        conn.execute(
            "CREATE TABLE tb_test_item (id bigserial PRIMARY KEY, \
                embedding vector(3) NOT NULL, sparse_embedding svector(3), \
                    float16_embedding vecf16(3), binary_embedding bvector(3));",
        )
        conn.commit()
        try:
            yield conn
        finally:
            conn.execute("DROP TABLE IF EXISTS tb_test_item;")
            conn.commit()


@pytest.mark.parametrize(("index_name", "index_option"), INDEX_OPTIONS.items())
def test_create_index(session: Connection, index_name: str, index_option: str):
    create_items(session)
    stat = sql.SQL(
        "CREATE INDEX {} ON tb_test_item USING vectors (embedding vector_l2_ops) WITH (options={});",
    ).format(sql.Identifier(index_name), index_option)

    session.execute(stat)
    session.commit()


def test_invalid_insert(session: Connection):
    for i, e in enumerate(INVALID_VECTORS):
        try:
            session.execute("INSERT INTO tb_test_item (embedding) VALUES (%s);", (e,))
        except Exception:
            session.rollback()
        else:
            session.rollback()
            raise AssertionError(
                "failed to raise invalid value error for {}th vector {}".format(i, e),
            )


# =================================
# Semetic search tests
# =================================


def test_copy(session: Connection):
    with session.cursor() as cursor, cursor.copy(
        "COPY tb_test_item (embedding, sparse_embedding, float16_embedding, binary_embedding) \
            FROM STDIN (FORMAT BINARY)"
    ) as copy:
        for e in zip(VECTORS, SPARSE_VECTORS, FLOAT16_VECTORS, BINARY_VECTORS):
            copy.write_row(e)

    session.commit()
    cur = session.execute("SELECT embedding FROM tb_test_item;", binary=True)
    rows = cur.fetchall()
    # query dense
    assert len(rows) == len(VECTORS)
    for i, (e,) in enumerate(rows):
        assert np.allclose(e.to_numpy(), VECTORS[i], atol=1e-10)
    # query sparse
    cur = session.execute("SELECT * FROM tb_test_item;", binary=True)
    rows = cur.fetchall()
    assert len(rows) == len(VECTORS)
    assert str(rows[0][1].to_list()) == "[1.0, 2.0, 3.0]"
    assert str(rows[1][1].to_list()) == "[7.0, 7.0, 7.0]"
    assert str(rows[3][1].to_list()) == "[1.0, 1.0, 1.0]"
    session.execute("Delete FROM tb_test_item;")
    session.commit()


def create_items(session: Connection):
    with session.cursor() as cur:
        data = zip(VECTORS, SPARSE_VECTORS, FLOAT16_VECTORS, BINARY_VECTORS)
        cur.executemany(
            "INSERT INTO tb_test_item (embedding, sparse_embedding, float16_embedding, binary_embedding) VALUES (%s, %s, %s, %s);",
            [e for e in data],
        )
        cur.execute("SELECT * FROM tb_test_item;")
        session.commit()
        rows = cur.fetchall()
        assert len(rows) == len(VECTORS)
        for i, e in enumerate(rows):
            assert np.allclose(e[1].to_numpy(), VECTORS[i], atol=1e-10)


def test_l2_distance(session: Connection):
    create_items(session)
    cur = session.execute(
        "SELECT embedding, embedding <-> %s FROM tb_test_item;",
        (L2_DIS_OP,),
    )
    for emb, dis in cur.fetchall():
        expect = l2_distance(np.array(L2_DIS_OP), emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_max_inner_product(session: Connection):
    create_items(session)
    cur = session.execute(
        "SELECT embedding, embedding <#> %s FROM tb_test_item;",
        (MAX_INNER_PROD_OP,),
    )
    for emb, dis in cur.fetchall():
        expect = max_inner_product(np.array(MAX_INNER_PROD_OP), emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_cosine_distance(session: Connection):
    create_items(session)
    cur = session.execute(
        "SELECT embedding, embedding <=> %s FROM tb_test_item;", (COSINE_DIS_OP,)
    )
    for emb, dis in cur.fetchall():
        expect = cosine_distance(np.array(COSINE_DIS_OP), emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_binary_jaccard_distance(session: Connection):
    create_items(session)
    cur = session.execute(
        "SELECT binary_embedding, binary_embedding <~> %s FROM tb_test_item;",
        (BinaryVector(JACCARD_DIS_OP),),
    )
    for emb, dis in cur.fetchall():
        expect = jaccard_distance(JACCARD_DIS_OP, emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_float16_vector(session):
    create_items(session)
    cur = session.execute(
        "SELECT float16_embedding, float16_embedding <-> %s FROM tb_test_item;",
        (Float16Vector(FLOAT16_OP),),
    )
    for emb, dis in cur.fetchall():
        expect = l2_distance(FLOAT16_OP, emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-2)


def test_sparse_vector(session):
    create_items(session)
    cur = session.execute(
        "SELECT sparse_embedding, sparse_embedding <-> %s FROM tb_test_item;",
        (SPARSE_OP,),
    )
    for emb, dis in cur.fetchall():
        expect = l2_distance(SPARSE_OP.to_numpy(), emb.to_numpy())
        assert np.allclose(expect, dis, atol=1e-10)


def test_filter(session):
    create_items(session)
    cur = session.execute(
        "SELECT embedding <-> %s FROM tb_test_item WHERE embedding <-> %s < %s;",
        (L2_DIS_OP, L2_DIS_OP, FILTER_VALUE),
    )
    for (dis,) in cur.fetchall():
        assert dis < FILTER_VALUE


# =================================
# Suffix functional tests
# =================================


def test_delete(session: Connection):
    create_items(session)
    session.execute("DELETE FROM tb_test_item WHERE embedding = %s;", (VECTORS[0],))
    session.commit()
    cur = session.execute("SELECT * FROM tb_test_item;")
    assert len(cur.fetchall()) == len(VECTORS) - 1
