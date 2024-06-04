# PGVecto.rs support for Python

<p align=center>
<a href="https://discord.gg/KqswhpVgdU"><img alt="discord invitation link" src="https://dcbadge.vercel.app/api/server/KqswhpVgdU?style=flat"></a>
<a href="https://twitter.com/TensorChord"><img src="https://img.shields.io/twitter/follow/tensorchord?style=social" alt="trackgit-views" /></a>
<a href="https://pdm.fming.dev"><img src="https://img.shields.io/badge/pdm-managed-blueviolet" alt="trackgit-views" /></a>
<a href="https://pypi.org/project/pgvecto_rs/"><img src="https://img.shields.io/pypi/dm/pgvecto_rs.svg?label=Pypi%20downloads" alt="trackgit-views" /></a>
</p>

## Usage

Install from PyPI:
```bash
pip install pgvecto_rs
```

See the [usage of SDK](#sdk)

Or use it as an extension of postgres clients:
- [SQLAlchemy](#sqlalchemy)
- [psycopg3](#psycopg3)

### Requirements

To initialize a pgvecto.rs instance, you can run our official image by [Quick start](https://github.com/tensorchord/pgvecto.rs?tab=readme-ov-file#quick-start):

You can get the latest tags from the [Release page](https://github.com/tensorchord/pgvecto.rs/releases). For example, it might be:

```bash
docker run \
  --name pgvecto-rs-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d tensorchord/pgvecto-rs:pg16-v0.2.1
```

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


### SQLAlchemy

Install dependencies:
```bash
pip install "pgvecto_rs[sqlalchemy]"
```

Then write your code. See [examples/sqlalchemy_example.py](examples/sqlalchemy_example.py) and [tests/test_sqlalchemy.py](tests/test_sqlalchemy.py) for example.

All the operators include:
- `squared_euclidean_distance`
- `negative_dot_product_distance`
- `negative_cosine_distance`

### psycopg3

Install dependencies:
```bash
pip install "pgvecto_rs[psycopg3]"
```

Then write your code. See [examples/psycopg_example.py](examples/psycopg_example.py) and [tests/test_psycopg.py](tests/test_psycopg.py) for example.

## Development

This package is managed by [PDM](https://pdm.fming.dev).

Set up things:
```bash
pdm venv create
pdm use # select the venv inside the project path
pdm sync
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
