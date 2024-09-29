from typing import List, Literal, Optional, Tuple, Type, Union
from uuid import UUID

from numpy import ndarray
from sqlalchemy import (
    BIGINT,
    Column,
    ColumnElement,
    Float,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql.pg_catalog import pg_class
from sqlalchemy.engine import Engine
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm.session import Session
from sqlalchemy.types import String

from pgvecto_rs.errors import CountRowsEstimateCondError
from pgvecto_rs.sdk.filters import Filter
from pgvecto_rs.sdk.record import Record, RecordORM, RecordORMType, Unique
from pgvecto_rs.sqlalchemy import VECTOR


def table_factory(collection_name, dimension, table_args, base_class=RecordORM):
    def __init__(self, **kwargs):  # noqa: N807
        base_class.__init__(self, **kwargs)

    newclass = type(
        collection_name,
        (base_class,),
        {
            "__init__": __init__,
            "__tablename__": f"collection_{collection_name}",
            "__table_args__": table_args,
            "id": mapped_column(
                postgresql.UUID(as_uuid=True),
                primary_key=True,
            ),
            "text": mapped_column(String),
            "meta": mapped_column(postgresql.JSONB),
            "embedding": mapped_column(VECTOR(dimension)),
        },
    )
    return newclass


class PGVectoRs:
    _engine: Engine
    _table: Type[RecordORM]
    dimension: int

    def __init__(  # noqa: PLR0913
        self,
        db_url: str,
        collection_name: str,
        dimension: int,
        recreate: bool = False,
        constraints: Union[List[Unique], None] = None,
    ) -> None:
        """Connect to an existing table or create a new empty one.
        If the `recreate=True`, the table will be dropped if it exists.

        Args:
        ----
            db_url (str): url to the database.
            collection_name (str): name of the collection. A prefix `collection_` is added to actual table name.
            dimension (int): dimension of the embeddings.
            recreate (bool): drop the table if it exists. Defaults to False.
            constraints (List[Unique]): add constraints to columns, e.g. UNIQUE constraint
        """
        if constraints is None or len(constraints) == 0:
            table_args = {"extend_existing": True}
        else:
            table_args = (
                *[col.make() for col in constraints],
                {"extend_existing": True},
            )

        self._engine = create_engine(db_url)
        self._table = table_factory(collection_name, dimension, table_args)
        with Session(self._engine) as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vectors"))
            if recreate:
                session.execute(
                    text(f"DROP TABLE IF EXISTS {self._table.__tablename__}")
                )
            session.commit()
        self._table.__table__.create(self._engine, checkfirst=True)
        self.dimension = dimension

    def insert(self, records: List[Record]) -> None:
        with Session(self._engine) as session:
            for record in records:
                session.execute(
                    insert(self._table).values(
                        id=record.id,
                        text=record.text,
                        meta=record.meta,
                        embedding=record.embedding,
                    ),
                )
            session.commit()

    def search(
        self,
        embedding: Union[ndarray, List[float]],
        distance_op: Literal["<->", "<=>", "<#>"] = "<->",
        top_k: int = 4,
        filter: Optional[Filter] = None,
    ) -> List[Tuple[Record, float]]:
        """Search for the nearest records.

        Args:
        ----
            embedding : Target embedding.
            distance_op : Distance op.
            top_k : Max records to return. Defaults to 4.
            filter : Read our document. Defaults to None.
            order_by_dis : Order by distance. Defaults to True.

        Returns:
        -------
            List of records and corresponding distances.

        """
        with Session(self._engine) as session:
            stmt = (
                select(
                    self._table,
                    self._table.embedding.op(distance_op, return_type=Float)(
                        embedding,
                    ).label("distance"),
                )
                .limit(top_k)
                .order_by("distance")
            )
            if filter is not None:
                stmt = stmt.where(filter(self._table))
            res = session.execute(stmt)
            return [(Record.from_orm(row[0]), row[1]) for row in res]

    # ================ Stat ==================
    def row_count(self, estimate: bool = True, filter: Optional[Filter] = None) -> int:
        if estimate and filter is not None:
            raise CountRowsEstimateCondError()
        if estimate:
            stmt = (
                select(func.cast(Column("reltuples", Float), BIGINT).label("rows"))
                .select_from(pg_class)
                .where(
                    Column("oid", Float)
                    == func.cast(self._table.__tablename__, postgresql.REGCLASS)
                )
            )
            with Session(self._engine) as session:
                result = session.execute(stmt).fetchone()
        else:
            stmt = select(func.count("*").label("rows")).select_from(self._table)
            if filter is not None:
                stmt = stmt.where(filter(self._table))
            with Session(self._engine) as session:
                result = session.execute(stmt).fetchone()
        return result[0]

    # ================ Delete ================
    def delete(self, filter: Filter) -> None:
        with Session(self._engine) as session:
            session.execute(delete(self._table).where(filter(self._table)))
            session.commit()

    def delete_all(self) -> None:
        with Session(self._engine) as session:
            session.execute(delete(self._table))
            session.commit()

    def delete_by_ids(self, ids: List[UUID]) -> None:
        def filter(record: RecordORMType) -> ColumnElement[bool]:
            return record.id.in_(ids)

        with Session(self._engine) as session:
            session.execute(delete(self._table).where(filter(self._table)))
            session.commit()

    # ================ Drop ================
    def drop(self) -> None:
        """Drop the table which the client is connected to."""
        self._table.__table__.drop(self._engine)
