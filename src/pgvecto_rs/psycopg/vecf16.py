from psycopg.adapt import Dumper, Loader
from psycopg.pq import Format

from pgvecto_rs.errors import TypeNotFoundError
from pgvecto_rs.types import Float16Vector


class Float16VectorDumper(Dumper):
    format = Format.TEXT

    def dump(self, obj):
        return Float16Vector._to_db(obj).encode("utf8")


class Float16VectorBinaryDumper(Float16VectorDumper):
    format = Format.BINARY

    def dump(self, obj):
        return Float16Vector._to_db_binary(obj)


class Float16VectorLoader(Loader):
    format = Format.TEXT

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return Float16Vector._from_db(data.decode("utf8"))


class Float16VectorBinaryLoader(Float16VectorLoader):
    format = Format.BINARY

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return Float16Vector._from_db_binary(data)


def register_vecf16_info(context, info):
    if info is None:
        raise TypeNotFoundError("vecf16")
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = type("", (Float16VectorDumper,), {"oid": info.oid})
    binary_dumper = type("", (Float16VectorBinaryDumper,), {"oid": info.oid})

    adapters = context.adapters
    adapters.register_dumper(Float16Vector, text_dumper)
    adapters.register_dumper(Float16Vector, binary_dumper)
    adapters.register_loader(info.oid, Float16VectorLoader)
    adapters.register_loader(info.oid, Float16VectorBinaryLoader)
