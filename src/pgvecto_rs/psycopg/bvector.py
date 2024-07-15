from psycopg.adapt import Dumper, Loader
from psycopg.pq import Format

from pgvecto_rs.errors import TypeNotFoundError
from pgvecto_rs.types import BinaryVector


class BinaryVectorDumper(Dumper):
    format = Format.TEXT

    def dump(self, obj):
        return BinaryVector._to_db(obj).encode("utf8")


class BinaryVectorBinaryDumper(BinaryVectorDumper):
    format = Format.BINARY

    def dump(self, obj):
        return BinaryVector._to_db_binary(obj)


class BinaryVectorLoader(Loader):
    format = Format.TEXT

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return BinaryVector._from_db(data.decode("utf8"))


class BinaryVectorBinaryLoader(BinaryVectorLoader):
    format = Format.BINARY

    def load(self, data):
        if isinstance(data, memoryview):
            data = bytes(data)
        return BinaryVector._from_db_binary(data)


def register_bvector_info(context, info):
    if info is None:
        raise TypeNotFoundError("bvector")
    info.register(context)

    # add oid to anonymous class for set_types
    text_dumper = type("", (BinaryVectorDumper,), {"oid": info.oid})
    binary_dumper = type("", (BinaryVectorBinaryDumper,), {"oid": info.oid})

    adapters = context.adapters
    adapters.register_dumper(BinaryVector, text_dumper)
    adapters.register_dumper(BinaryVector, binary_dumper)
    adapters.register_loader(info.oid, BinaryVectorLoader)
    adapters.register_loader(info.oid, BinaryVectorBinaryLoader)
