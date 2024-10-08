from enum import IntEnum
from functools import reduce
from typing import List, Optional, Type, Union
from uuid import UUID, uuid4

from numpy import array, float32, ndarray
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped


class Column(IntEnum):
    TEXT = 1
    META = 2
    EMBEDDING = 4


class Unique:
    def __init__(self, columns: List[Column]):
        self.value = reduce(lambda x, y: x | y, columns)

    def make(self) -> UniqueConstraint:
        ans: List[UniqueConstraint] = []
        if self.value & Column.TEXT:
            ans.append("text")
        if self.value & Column.META:
            ans.append("meta")
        if self.value & Column.EMBEDDING:
            ans.append("embedding")
        return UniqueConstraint(*ans)


class RecordORM(DeclarativeBase):
    __tablename__: str
    id: Mapped[UUID]
    text: Mapped[str]
    meta: Mapped[dict]
    embedding: Mapped[ndarray]


RecordORMType = Type[RecordORM]


class Record:
    id: UUID
    text: str
    meta: dict
    embedding: ndarray

    def __init__(self, id: UUID, text: str, meta: dict, embedding: ndarray):
        self.id = id
        self.text = text
        self.meta = meta
        self.embedding = embedding

    def __repr__(self) -> str:
        return f"""============= Record =============
[id]       : {self.id}
[text]     : {self.text}
[meta]     : {self.meta}
[embedding]: {self.embedding}
========== End of Record ========="""

    @classmethod
    def from_orm(cls, orm: RecordORM):
        return cls(orm.id, orm.text, orm.meta, orm.embedding)

    @classmethod
    def from_text(
        cls,
        text: str,
        embedding: Union[ndarray, List[float]],
        meta: Optional[dict] = None,
    ):
        if isinstance(embedding, list):
            embedding = array(embedding, dtype=float32)
        return cls(uuid4(), text, meta or {}, embedding)
