# PGVecto.rs support for Python

<p align=center>
<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://pdm.fming.dev"><img src="https://img.shields.io/badge/pdm-managed-blueviolet" alt="trackgit-views" /></a>
<a href="https://pypi.org/project/pgvecto_rs/"><img src="https://img.shields.io/pypi/v/pgvecto_rs" alt="trackgit-views" /></a>
<a href="https://pypi.org/project/pgvecto_rs/"><img src="https://img.shields.io/pypi/dm/pgvecto_rs.svg?label=Pypi%20downloads" alt="trackgit-views" /></a>
</p>

[PGVecto.rs](https://github.com/tensorchord/pgvecto.rs) Python library, supports Django, SQLAlchemy, and Psycopg 3.

|                                                        | [Vector](https://docs.pgvecto.rs/usage/indexing.html) | [Sparse Vector](https://docs.pgvecto.rs/reference/vector-types/svector.html) | [Half-Precision Vector](https://docs.pgvecto.rs/reference/vector-types/vecf16.html) | [Binary Vector](https://docs.pgvecto.rs/reference/vector-types/bvector.html) |
| ------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) | ✅Insert                                               | ✅Insert                                                      | ✅Insert                                                      | ✅Insert                                                      |
| [Psycopg3](https://github.com/psycopg/psycopg)         | ✅Insert ✅Copy                                         | ✅Insert ✅Copy                                                | ✅Insert ✅Copy                                                | ✅Insert ✅Copy                                                |
| [Django](https://github.com/django/django)             | ✅Insert                                               | ✅Insert                                                      | ✅Insert                                                      | ✅Insert                                                      |

## Usage

Install from PyPI:
```bash
pip install pgvecto_rs
```

And use it with your database library:
- [SQLAlchemy](#sqlalchemy)
- [Psycopg3](#psycopg3)
- [Django](#django)

Or as a standalone SDK:
- [usage of SDK](#sdk)

### Requirements

To initialize a pgvecto.rs instance, you can run our official image by [Quick start](https://github.com/tensorchord/pgvecto.rs?tab=readme-ov-file#quick-start):

You can get the latest tags from the [Release page](https://github.com/tensorchord/pgvecto.rs/releases). For example, it might be:

```bash
docker run \
  --name pgvecto-rs-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d tensorchord/pgvecto-rs:pg16-v0.3.0
```

### SQLAlchemy

Install dependencies:
```bash
pip install "pgvecto_rs[sqlalchemy]"
```

Initialize a connection
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

URL = "postgresql://postgres:mysecretpassword@localhost:5432/postgres"
engine = create_engine(URL)
with Session(engine) as session:
    pass
```

Enable the extension
```python
from sqlalchemy import text

session.execute(text('CREATE EXTENSION IF NOT EXISTS vectors'))
```

Create a model
```python
from pgvecto_rs.sqlalchemy import Vector

class Item(Base):
    embedding = mapped_column(Vector(3))
```

All supported types are shown in this table

| Native types | Types for SQLAlchemy | Correspond to pgvector-python |
| ------------ | -------------------- | ----------------------------- |
| vector       | VECTOR               | VECTOR                        |
| svector      | SVECTOR              | SPARSEVEC                     |
| vecf16       | VECF16               | HALFVEC                       |
| bvector      | BVECTOR              | BIT                           |

Insert a vector
```python
from sqlalchemy import insert

stmt = insert(Item).values(embedding=[1, 2, 3])
session.execute(stmt)
session.commit()
```

Add an approximate index
```python
from sqlalchemy import Index
from pgvecto_rs.types import IndexOption, Hnsw, Ivf

index = Index(
    "emb_idx_1",
    Item.embedding,
    postgresql_using="vectors",
    postgresql_with={
        "options": f"$${IndexOption(index=Ivf(), threads=1).dumps()}$$"
    },
    postgresql_ops={"embedding": "vector_l2_ops"},
)
# or
index = Index(
    "emb_idx_2",
    Item.embedding,
    postgresql_using="vectors",
    postgresql_with={
        "options": f"$${IndexOption(index=Hnsw()).dumps()}$$"
    },
    postgresql_ops={"embedding": "vector_l2_ops"},
)
# Apply changes
index.create(session.bind)
```

Get the nearest neighbors to a vector
```python
from sqlalchemy import select

session.scalars(select(Item.embedding).order_by(Item.embedding.l2_distance(target.embedding)))
```

Also supports `max_inner_product`, `cosine_distance` and `jaccard_distance(for BVECTOR)`

Get items within a certain distance
```python
session.scalars(select(Item).filter(Item.embedding.l2_distance([3, 1, 2]) < 5))
```

See [examples/sqlalchemy_example.py](examples/sqlalchemy_example.py) and [tests/test_sqlalchemy.py](tests/test_sqlalchemy.py) for more examples

### Psycopg3

Install dependencies:
```bash
pip install "pgvecto_rs[psycopg3]"
```

Initialize a connection
```python
import psycopg

URL = "postgresql://postgres:mysecretpassword@localhost:5432/postgres"
with psycopg.connect(URL) as conn:
    pass
```

Enable the extension and register vector types
```python
from pgvecto_rs.psycopg import register_vector

conn.execute('CREATE EXTENSION IF NOT EXISTS vectors')
register_vector(conn)
# or asynchronously
# await register_vector_async(conn)
```

Create a table
```python
conn.execute('CREATE TABLE items (embedding vector(3))')
```

Insert or copy vectors into table
```python
conn.execute('INSERT INTO items (embedding) VALUES (%s)', ([1, 2, 3],))
# or faster, copy it
with conn.cursor() as cursor, cursor.copy(
    "COPY items (embedding) FROM STDIN (FORMAT BINARY)"
) as copy:
    copy.write_row([np.array([1, 2, 3])])
```

Add an approximate index
```python
from pgvecto_rs.types import IndexOption, Hnsw, Ivf

conn.execute(
    "CREATE INDEX emb_idx_1 ON items USING \
        vectors (embedding vector_l2_ops) WITH (options=$${}$$);".format(
        IndexOption(index=Hnsw(), threads=1).dumps()
    ),
)
# or
conn.execute(
    "CREATE INDEX emb_idx_2 ON items USING \
        vectors (embedding vector_l2_ops) WITH (options=$${}$$);".format(
        IndexOption(index=Ivf()).dumps()
    ),
)
# Apply all changes
conn.commit()
```

Get the nearest neighbors to a vector
```python
conn.execute('SELECT * FROM items ORDER BY embedding <-> %s LIMIT 5', (embedding,)).fetchall()
```

Get the distance
```python
conn.execute('SELECT embedding <-> %s FROM items \
    ORDER BY embedding <-> %s', (embedding, embedding)).fetchall()
```

Get items within a certain distance
```python
conn.execute('SELECT * FROM items WHERE embedding <-> %s < 1.0 \
    ORDER BY embedding <-> %s', (embedding, embedding)).fetchall()
```

See [examples/psycopg_example.py](examples/psycopg_example.py) and [tests/test_psycopg.py](tests/test_psycopg.py) for more examples

### Django

Install dependencies:

```bash
pip install "pgvecto_rs[django]"
```

Create a migration to enable the extension

```python
from pgvecto_rs.django import VectorExtension

class Migration(migrations.Migration):
    operations = [
        VectorExtension()
    ]
```

Add a vector field to your model

```python
from pgvecto_rs.django import VectorField

class Document(models.Model):
    embedding = VectorField(dimensions=3)
```

All supported types are shown in this table

| Native types | Types for Django   | Correspond to pgvector-python |
| ------------ | ------------------ | ----------------------------- |
| vector       | VectorField        | VectorField                   |
| svector      | SparseVectorField  | SparseVectorField             |
| vecf16       | Float16VectorField | HalfVectorField               |
| bvector      | BinaryVectorField  | BitField                      |

Insert a vector
```python
Item(embedding=[1, 2, 3]).save()
```

Add an approximate index
```python
from django.db import models
from pgvecto_rs.django import HnswIndex, IvfIndex
from pgvecto_rs.types import IndexOption, Hnsw


class Item(models.Model):
    class Meta:
        indexes = [
            HnswIndex(
                name="emb_idx_1",
                fields=["embedding"],
                opclasses=["vector_l2_ops"],
                m=16,
                ef_construction=100,
                threads=1,
            )
            # or
            IvfIndex(
                name="emb_idx_2",
                fields=["embedding"],
                nlist=3,
                opclasses=["vector_l2_ops"],
            ),
        ]
```

Get the nearest neighbors to a vector
```python
from pgvecto_rs.django import L2Distance

Item.objects.order_by(L2Distance('embedding', [3, 1, 2]))[:5]
```

Also supports `MaxInnerProduct`, `CosineDistance` and `JaccardDistance(for BinaryVectorField)`

Get the distance
```python
Item.objects.annotate(distance=L2Distance('embedding', [3, 1, 2]))
```

Get items within a certain distance
```python
Item.objects.alias(distance=L2Distance('embedding', [3, 1, 2])).filter(distance__lt=5)
```

See [examples/django_example.py](examples/django_example.py) and [tests/test_django.py](tests/test_django.py) for more examples.

### SDK

Our SDK is designed to use the pgvecto.rs out-of-box. You can exploit the power of pgvecto.rs to do similarity search or retrieve with filters, without writing any SQL code.

Install dependencies:
```bash
pip install "pgvecto_rs[sdk]"
```

A minimal example:

```Python
from pgvecto_rs.sdk import PGVectoRs, Record

# Create a client
client = PGVectoRs(
    db_url="postgresql+psycopg://postgres:mysecretpassword@localhost:5432/postgres",
    table_name="example",
    dimension=3,
)

try:
    # Add some records
    client.add_records(
        [
            Record.from_text("hello 1", [1, 2, 3]),
            Record.from_text("hello 2", [1, 2, 4]),
        ]
    )

    # Search with default operator (sqrt_euclid).
    # The results is sorted by distance
    for rec, dis in client.search([1, 2, 5]):
        print(rec.text)
        print(dis)
finally:
    # Clean up (i.e. drop the table)
    client.drop()
```

Output:
```
hello 2
1.0
hello 1
4.0
```

See [examples/sdk_example.py](examples/sdk_example.py) and [tests/test_sdk.py](tests/test_sdk.py) for more examples.




## Development

This package is managed by [PDM](https://pdm.fming.dev).

Set up things:
```bash
pdm venv create
pdm use # select the venv inside the project path
pdm sync -d -G :all --no-isolation

# lock requirement
# need pdm >=2.17: https://pdm-project.org/latest/usage/lock-targets/#separate-lock-files-or-merge-into-one
pdm lock -d -G :all --python=">=3.9"
pdm lock -d -G :all --python="<3.9" --append
# install package to local
# `--no-isolation` is required for scipy
pdm install -d --no-isolation
```

Run lint:
```bash
pdm run format
pdm run fix
pdm run check
```

Run test in current environment:
```bash
pdm run test
```


## Test

[Tox](https://tox.wiki) is used to test the package locally.

Run test in all environment:
```bash
tox run
```

## Acknowledgement

We would like to express our gratitude to the creators and contributors of the [pgvector-python](https://github.com/pgvector/pgvector-python) repository for their valuable code and architecture, which greatly influenced the development of this repository.