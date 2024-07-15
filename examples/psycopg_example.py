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
        "CREATE TABLE documents (id SERIAL PRIMARY KEY, text TEXT NOT NULL, embedding vector(3) NOT NULL);",
    )
    conn.commit()
    try:
        # Insert 3 rows into the table
        conn.execute(
            "INSERT INTO documents (text, embedding) VALUES (%s, %s);",
            ("hello world", [1, 2, 3]),
        )
        conn.execute(
            "INSERT INTO documents (text, embedding) VALUES (%s, %s);",
            ("hello postgres", [1.0, 2.0, 4.0]),
        )
        conn.execute(
            "INSERT INTO documents (text, embedding) VALUES (%s, %s);",
            ("hello pgvecto.rs", np.array([1, 3, 4])),
        )
        # Create index for the vectors
        conn.execute(
            "CREATE INDEX embedding_idx ON documents USING \
                vectors (embedding vector_l2_ops) WITH (options=$${}$$);".format(
                IndexOption(index=Hnsw(), threads=1).dumps()
            ),
        )
        conn.commit()

        # Select the row "hello pgvecto.rs"
        cur = conn.execute(
            "SELECT * FROM documents WHERE text = %s;",
            ("hello pgvecto.rs",),
        )
        target = cur.fetchone()[2]

        # Select all the rows and sort them
        # by the l2_distance to "hello pgvecto.rs"
        cur = conn.execute(
            "SELECT text, embedding, embedding <-> %s AS distance FROM documents ORDER BY distance;",
            (target,),
        )
        for row in cur.fetchall():
            print(row)
        # The output will be:
        # ```
        # ('hello pgvecto.rs', array([1., 3., 4.], dtype=float32), 0.0)
        # ('hello postgres', array([1., 2., 4.], dtype=float32), 1.0)
        # ('hello world', array([1., 2., 3.], dtype=float32), 2.0)
        # ```
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
        "CREATE TABLE documents (id SERIAL PRIMARY KEY, text TEXT NOT NULL, embedding svector(60) NOT NULL);",
    )
    conn.commit()
    try:
        # Insert 3 rows into the table
        conn.execute(
            "INSERT INTO documents (text, embedding) VALUES (%s, %s);",
            ("hello world", SparseVector({0: 2, 1: 4, 2: 6}, 60)),
        )
        conn.execute(
            "INSERT INTO documents (text, embedding) VALUES (%s, %s);",
            (
                "hello postgres",
                SparseVector(
                    coo_array(
                        (np.array([2.0, 3.0]), (np.array([0, 0]), np.array([1, 2]))),
                        shape=(1, 60),
                    )
                ),
            ),
        )
        conn.execute(
            "INSERT INTO documents (text, embedding) VALUES (%s, %s);",
            ("hello pgvecto.rs", SparseVector.from_parts(60, [0, 2], [1.0, 3.0])),
        )
        # Create index for the vectors
        conn.execute(
            "CREATE INDEX embedding_idx ON documents USING \
                vectors (embedding svector_l2_ops) WITH (options=$${}$$);".format(
                IndexOption(index=Hnsw(), threads=1).dumps()
            ),
        )
        conn.commit()

        # Select the row "hello pgvecto.rs"
        cur = conn.execute(
            "SELECT * FROM documents WHERE text = %s;",
            ("hello pgvecto.rs",),
        )
        target = cur.fetchone()[2]

        # Select all the rows and sort them
        # by the l2_distance to "hello pgvecto.rs"
        cur = conn.execute(
            "SELECT text, embedding, embedding <-> %s AS distance FROM documents ORDER BY distance;",
            (target,),
        )
        for row in cur.fetchall():
            print(row)
        # The output will be:
        # ```
        # ('hello pgvecto.rs', SparseVector({0: 1.0, 2: 3.0}, 60), 0.0)
        # ('hello postgres', SparseVector({1: 2.0, 2: 3.0}, 60), 5.0)
        # ('hello world', SparseVector({0: 2.0, 1: 4.0, 2: 6.0}, 60), 26.0)
        # ```
    finally:
        # Drop the table
        conn.execute("DROP TABLE IF EXISTS documents;")
        conn.commit()
