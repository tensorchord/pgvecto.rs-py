import os

import numpy as np
import scipy
from scipy.sparse import coo_array

from pgvecto_rs.types import (
    BinaryVector,
    Flat,
    Float16Vector,
    Hnsw,
    IndexOption,
    Ivf,
    Quantization,
    SparseVector,
    Vector,
)

PORT = os.getenv("DB_PORT", "5432")
HOST = os.getenv("DB_HOST", "localhost")
USER = os.getenv("DB_USER", "postgres")
PASS = os.getenv("DB_PASS", "mysecretpassword")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Run tests with shell:
#   DB_HOST=localhost DB_USER=postgres DB_PASS=password DB_NAME=postgres python3 -m pytest bindings/python/tests/
URL = f"postgresql://{USER}:{PASS}@{HOST}:{PORT}/{DB_NAME}"
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": DB_NAME,
        "USER": USER,
        "PASSWORD": PASS,
        "HOST": HOST,
        "PORT": PORT,
    }
}
# ==== test_type_equal ====

COO_COMPACT_1 = (
    coo_array(
        (
            np.array(
                [2, 4, 6],
            ),
            np.array([[1, 3, 5]]),
        ),
        shape=(6,),
    )
    if scipy.__version__ >= "1.13"
    else coo_array(
        (
            np.array(
                [2, 4, 6],
            ),
            (np.array([0, 0, 0]), np.array([1, 3, 5])),
        ),
        shape=(1, 6),
    )
)

EQUAL_SPARSE_VECTORS = [
    SparseVector({1: 2, 3: 4, 5: 6}, 6),
    SparseVector(
        coo_array(
            (
                np.array(
                    [2, 4, 6],
                ),
                (np.array([0, 0, 0]), np.array([1, 3, 5])),
            ),
            shape=(1, 6),
        )
    ),
    SparseVector(COO_COMPACT_1),
    SparseVector.from_parts(6, [1, 3, 5], [2, 4, 6]),
]

EQUAL_VECTORS = [
    ([1.0, -1.0, 0.0],),
    (np.array([1.0, -1.0, 0.0]),),
    Vector([1.0, -1.0, 0.0]),
    Vector(np.array([1.0, -1.0, 0.0])),
]

INDEX_OPTION_DUMPS = [
    (
        IndexOption(
            index=Flat(),
            threads=1,
        ),
        "[optimizing]\noptimizing_threads = 1\n\n[indexing.flat]\n",
    ),
    (
        IndexOption(
            index=Hnsw(),
            threads=8,
        ),
        "[optimizing]\noptimizing_threads = 8\n\n[indexing.hnsw]\n",
    ),
    (
        IndexOption(
            index=Hnsw(m=1),
        ),
        "[indexing.hnsw]\nm = 1\n",
    ),
    (
        IndexOption(index=Ivf(quantization=Quantization(typ="trivial"))),
        "[indexing.ivf.quantization.trivial]\n",
    ),
    (
        IndexOption(
            index=Hnsw(
                m=1,
                ef_construction=2,
                quantization=Quantization(typ="product", ratio="x4"),
            ),
        ),
        '[indexing.hnsw]\nm = 1\nef_construction = 2\n\n[indexing.hnsw.quantization.product]\nratio = "x4"\n',
    ),
]
# ==== test_create_index ====

INDEX_OPTIONS = {
    "flat": IndexOption(
        index=Flat(),
        threads=1,
    ).dumps(),
    "hnsw": IndexOption(
        index=Hnsw(),
        threads=1,
    ).dumps(),
}

# ==== test_invalid_insert ====
INVALID_VECTORS = [
    [[1, 2], [3, 4], [5, 6]],
    ["123.", "123", "a"],
    np.zeros(shape=(1, 2)),
]

# =================================
# Semetic search tests
# =================================
VECTORS = [
    [1, 2, 3],
    [7, 7, 7],
    [0.0, -45, 2.34],
    np.ones(shape=(3)),
]

COO_COMPACT_2 = (
    coo_array((np.array([2.0, 3.0]), np.array([[1, 2]])), shape=(3,))
    if scipy.__version__ >= "1.13"
    else coo_array(
        (np.array([2.0, 3.0]), (np.array([0, 0]), np.array([1, 2]))), shape=(1, 3)
    )
)

SPARSE_VECTORS = [
    SparseVector({0: 2, 1: 4, 2: 6}, 3),
    SparseVector.from_parts(3, [0, 2], [1.0, 3.0]),
    SparseVector({0: 1.0, 1: 2.0, 2: 3.0}, 3),
    SparseVector(COO_COMPACT_2),
]
FLOAT16_VECTORS = [
    Float16Vector([1, 2, 3]),
    Float16Vector([0, 0, 0]),
    Float16Vector(np.array([-2.0, 2.1, 3.1], dtype=np.float16)),
    Float16Vector(np.ones(shape=(3))),
]
BINARY_VECTORS = [
    BinaryVector([False, False, True]),
    BinaryVector([0, 0, 1]),
    BinaryVector(np.ones(shape=(3), dtype=np.int8)),
    BinaryVector(np.array([True, True, False])),
]

# Operator tests
L2_DIS_OP = [0, 0, 0]
MAX_INNER_PROD_OP = [1, 2, 4]
COSINE_DIS_OP = [3, 2, 1]
JACCARD_DIS_OP = np.array([True, True, False])

# Vector type tests
SPARSE_OP = SparseVector({0: 3.0, 1: 2.0, 2: 1.0}, 3)
FLOAT16_OP = np.array([0.1, 0.2, 0.4], dtype=np.float16)

FILTER_VALUE = 4.0


def l2_distance(left: np.array, right: np.array):
    return np.linalg.norm(left - right) ** 2


def max_inner_product(left: np.array, right: np.array):
    return -np.dot(left, right)


def cosine_distance(left: np.array, right: np.array):
    return 1 - np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))


def jaccard_distance(left: np.array, right: np.array):
    return 1 - np.double(np.bitwise_and(left, right).sum()) / np.double(
        np.bitwise_or(left, right).sum()
    )


# ==== test_delete ====

__all__ = [
    "URL",
    "TOML_SETTINGS",
    "INVALID_VECTORS",
    "VECTORS",
    "EXPECTED_SQRT_EUCLID_DIS",
    "EXPECTED_NEG_DOT_PROD_DIS",
    "EXPECTED_NEG_COS_DIS",
    "LEN_AFT_DEL",
]
