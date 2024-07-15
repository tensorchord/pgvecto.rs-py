from psycopg.types import TypeInfo

from .bvector import register_bvector_info
from .svector import register_svector_info
from .vecf16 import register_vecf16_info
from .vector import register_vector_info


def register_vector(context):
    info = TypeInfo.fetch(context, "vector")
    register_vector_info(context, info)

    info = TypeInfo.fetch(context, "bvector")
    if info is not None:
        register_bvector_info(context, info)

    info = TypeInfo.fetch(context, "vecf16")
    if info is not None:
        register_vecf16_info(context, info)

    info = TypeInfo.fetch(context, "svector")
    if info is not None:
        register_svector_info(context, info)


async def register_vector_async(context):
    info = await TypeInfo.fetch(context, "vector")
    register_vector_info(context, info)

    info = await TypeInfo.fetch(context, "bvector")
    if info is not None:
        register_bvector_info(context, info)

    info = await TypeInfo.fetch(context, "vecf16")
    if info is not None:
        register_vecf16_info(context, info)

    info = await TypeInfo.fetch(context, "svector")
    if info is not None:
        register_svector_info(context, info)
