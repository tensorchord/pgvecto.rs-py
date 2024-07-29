import os

import numpy as np
import psycopg
from scipy.sparse import coo_array

from pgvecto_rs.psycopg import register_vector
from pgvecto_rs.types import Hnsw, IndexOption, SparseVector

URL = "postgresql://{username}:{password}@{host}:{port}/{db_name}".format(
    port=os.getenv("DB_PORT", "5432"),
    host=os.getenv("DB_HOST", "localhost"),
    username=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS", "mysecretpassword"),
    db_name=os.getenv("DB_NAME", "postgres"),
)


# =================================
# Dense Vector Example
# =================================


# Connect to the DB and init things
with psycopg.connect(URL) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vectors;")
    register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS documents;")
    conn.execute(
        "CREATE TABLE documents (id SERIAL PRIMARY KEY, embedding vector(3) NOT NULL);",
    )
    conn.commit()
    try:
        embeddings = [
            np.array([1, 2, 3]),
            np.array([1.0, 2.0, 4.0]),
            np.array([1, 3, 4]),
        ]

        with (
            conn.cursor() as cursor,
            cursor.copy(
                "COPY documents (embedding) FROM STDIN (FORMAT BINARY)"
            ) as copy,
        ):
            # write row by row
            for e in embeddings:
                copy.write_row([e])
            copy.write_row([[1, 3, 5]])
        # Create index for the vectors
        conn.execute(
            "CREATE INDEX embedding_idx ON documents USING \
                vectors (embedding vector_l2_ops) WITH (options=$${}$$);".format(
                IndexOption(index=Hnsw(), threads=1).dumps()
            ),
        )
        conn.commit()

        # Select the rows using binary format
        cur = conn.execute(
            "SELECT * FROM documents;",
            binary=True,
        )
        for row in cur.fetchall():
            print(row[0], ": ", row[1].to_numpy())

        # The output will be:
        # 1 :  [1. 2. 3.]
        # 2 :  [1. 2. 4.]
        # 3 :  [1. 3. 4.]
        # 4 :  [1. 3. 5.]
    finally:
        # Drop the table
        conn.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()


# =================================
# Sparse Vector Example
# =================================

# Connect to the DB and init things
with psycopg.connect(URL) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vectors;")
    register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS documents;")
    conn.execute(
        "CREATE TABLE documents (id SERIAL PRIMARY KEY, embedding svector(60) NOT NULL);",
    )
    conn.commit()
    try:
        with (
            conn.cursor() as cursor,
            cursor.copy(
                "COPY documents (embedding) FROM STDIN (FORMAT BINARY)"
            ) as copy,
        ):
            copy.write_row([SparseVector({0: 2, 1: 4, 2: 6}, 60)])
            copy.write_row(
                [
                    SparseVector(
                        coo_array(
                            (np.array([2.0, 3.0]), np.array([[1, 2]])),
                            shape=(60,),
                        )
                    )
                ]
            )
            copy.write_row([SparseVector.from_parts(60, [0, 2], [1.0, 3.0])])
        conn.pgconn.flush()
        # Create index for the vectors
        conn.execute(
            "CREATE INDEX embedding_idx ON documents USING \
                vectors (embedding svector_l2_ops) WITH (options=$${}$$);".format(
                IndexOption(index=Hnsw(), threads=1).dumps()
            ),
        )
        conn.commit()

        # Select the rows using binary format
        cur = conn.execute(
            "SELECT * FROM documents;",
            binary=True,
        )
        for row in cur.fetchall():
            print(row[0], ": ", row[1])

        # The output will be:
        # 1 :  SparseVector({0: 2.0, 1: 4.0, 2: 6.0}, 60)
        # 2 :  SparseVector({1: 2.0, 2: 3.0}, 60)
        # 3 :  SparseVector({0: 1.0, 2: 3.0}, 60)
    finally:
        # Drop the table
        conn.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()
