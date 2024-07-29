import os

import numpy as np
from scipy.sparse import coo_array
from sqlalchemy import Index, Integer, String, create_engine, insert, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from pgvecto_rs.sqlalchemy import SVECTOR, VECTOR
from pgvecto_rs.types import Hnsw, IndexOption, SparseVector

URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
    port=os.getenv("DB_PORT", "5432"),
    host=os.getenv("DB_HOST", "localhost"),
    username=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS", "mysecretpassword"),
    db_name=os.getenv("DB_NAME", "postgres"),
)


# =================================
# Dense Vector Example
# =================================


# Define the ORM model
class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(String)
    embedding: Mapped[np.ndarray] = mapped_column(VECTOR(3))

    def __repr__(self) -> str:
        return f"{self.text}: {self.embedding}"


# Connect to the DB and create the table
engine = create_engine(URL)
Document.metadata.drop_all(engine)
Document.metadata.create_all(engine)

with Session(engine) as session:
    # Insert 3 rows into the table
    t1 = insert(Document).values(text="hello world", embedding=[1, 2, 3])
    t2 = insert(Document).values(text="hello postgres", embedding=[1.0, 2.0, 4.0])
    t3 = insert(Document).values(text="hello pgvecto.rs", embedding=np.array([1, 3, 4]))

    for t in [t1, t2, t3]:
        session.execute(t)
    session.commit()

    # Create index for the vectors
    index = Index(
        "embedding_idx",
        Document.embedding,
        postgresql_using="vectors",
        postgresql_with={
            "options": f"$${IndexOption(index=Hnsw(), threads=1).dumps()}$$"
        },
        postgresql_ops={"embedding": "vector_l2_ops"},
    )
    index.create(session.bind)

    # Select the row "hello pgvecto.rs"
    stmt = select(Document).where(Document.text == "hello pgvecto.rs")
    target = session.scalar(stmt)

    # Select all the rows and sort them
    # by the l2_distance to "hello pgvecto.rs"
    stmt = select(
        Document.text,
        Document.embedding,
        Document.embedding.l2_distance(target.embedding).label(
            "distance",
        ),
    ).order_by("distance")
    for text, emb, dis in session.execute(stmt):
        print((text, emb.to_numpy(), dis))

    # The output will be:
    # ```
    # ('hello pgvecto.rs', array([1., 3., 4.], dtype=float32), 0.0)
    # ('hello postgres', array([1., 2., 4.], dtype=float32), 1.0)
    # ('hello world', array([1., 2., 3.], dtype=float32), 2.0)
    # ```

# Drop the table
Document.metadata.drop_all(engine)


# =================================
# Sparse Vector Example
# =================================


# Define the ORM model
class SparseBase(DeclarativeBase):
    pass


class DocumentSparse(SparseBase):
    __tablename__ = "documents_sparse"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(String)
    embedding: Mapped[np.ndarray] = mapped_column(SVECTOR(60))

    def __repr__(self) -> str:
        return f"{self.text}: {self.embedding}"


# Connect to the DB and create the table
engine = create_engine(URL)
DocumentSparse.metadata.drop_all(engine)
DocumentSparse.metadata.create_all(engine)

with Session(engine) as session:
    # Insert 3 rows into the table
    t1 = insert(DocumentSparse).values(
        text="hello world", embedding=SparseVector({0: 2, 1: 4, 2: 6}, 60)
    )
    t2 = insert(DocumentSparse).values(
        text="hello postgres",
        embedding=SparseVector(
            coo_array(
                (np.array([2.0, 3.0]), np.array([[1, 2]])),
                shape=(60,),
            )
        ),
    )
    t3 = insert(DocumentSparse).values(
        text="hello pgvecto.rs",
        embedding=SparseVector.from_parts(60, [0, 2], [1.0, 3.0]),
    )
    for t in [t1, t2, t3]:
        session.execute(t)
    session.commit()

    # Create index for the vectors
    index = Index(
        "embedding_idx",
        DocumentSparse.embedding,
        postgresql_using="vectors",
        postgresql_with={
            "options": f"$${IndexOption(index=Hnsw(), threads=1).dumps()}$$"
        },
        postgresql_ops={"embedding": "svector_l2_ops"},
    )
    index.create(session.bind)

    # Select the row "hello pgvecto.rs"
    stmt = select(DocumentSparse).where(DocumentSparse.text == "hello pgvecto.rs")
    target = session.scalar(stmt)

    # Select all the rows and sort them
    # by the l2_distance to "hello pgvecto.rs"
    stmt = select(
        DocumentSparse.text,
        DocumentSparse.embedding,
        DocumentSparse.embedding.l2_distance(target.embedding).label(
            "distance",
        ),
    ).order_by("distance")
    for doc in session.execute(stmt):
        print(doc)

    # The output will be:
    # ```
    # ('hello pgvecto.rs', SparseVector({0: 1.0, 2: 3.0}, 60), 0.0)
    # ('hello postgres', SparseVector({1: 2.0, 2: 3.0}, 60), 5.0)
    # ('hello world', SparseVector({0: 2.0, 1: 4.0, 2: 6.0}, 60), 26.0)
    # ```

# Drop the table
DocumentSparse.metadata.drop_all(engine)
