import os

import django
import numpy as np
from django.conf import settings
from django.db import connection, migrations, models
from django.db.migrations.loader import MigrationLoader
from scipy.sparse import coo_array

from pgvecto_rs.django import (
    HnswIndex,
    L2Distance,
    SparseVectorField,
    VectorExtension,
    VectorField,
)
from pgvecto_rs.types import SparseVector

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DB_NAME", "postgres"),
        "USER": os.getenv("DB_USER", "postgres"),
        "PASSWORD": os.getenv("DB_PASS", "mysecretpassword"),
        "HOST": os.getenv("DB_HOST", "localhost"),
        "PORT": os.getenv("DB_PORT", "5432"),
    }
}
settings.configure(DATABASES=DATABASES)
django.setup()


# =================================
# Dense Vector Example
# =================================


class Documents(models.Model):
    id = models.BigAutoField(primary_key=True)
    text = models.TextField()
    embedding = VectorField(dim=3, null=True, blank=True)

    class Meta:
        app_label = "dense"
        indexes = (
            HnswIndex(
                name="embedding_idx",
                fields=["embedding"],
                opclasses=["vector_l2_ops"],
                threads=1,
            ),
        )


class Migration(migrations.Migration):
    initial = True

    dependencies = ()

    operations = (
        VectorExtension(),
        migrations.CreateModel(
            name="documents",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "text",
                    models.TextField(null=True, blank=True),
                ),
                (
                    "embedding",
                    VectorField(dim=3, null=True, blank=True),
                ),
            ],
        ),
        migrations.AddIndex(
            model_name="documents",
            index=HnswIndex(
                name="embedding_idx",
                fields=["embedding"],
                opclasses=["vector_l2_ops"],
                threads=1,
            ),
        ),
    )


with connection.cursor() as cursor:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vectors")
    cursor.execute("DROP TABLE IF EXISTS dense_documents, sparse_documents")

    # Connect to the DB and create the table
    migration = Migration("initial", "dense")
    loader = MigrationLoader(connection, replace_migrations=False)
    loader.graph.add_node(("dense", migration.name), migration)
    sql_statements = loader.collect_sql([(migration, False)])
    cursor.execute("\n".join(sql_statements))

    # Insert 3 rows into the table
    Documents(text="hello world", embedding=[1, 2, 3]).save()
    Documents(text="hello postgres", embedding=[1.0, 2.0, 4.0]).save()
    Documents(text="hello pgvecto.rs", embedding=np.array([1, 3, 4])).save()

    # Select the row "hello pgvecto.rs"
    target = Documents.objects.filter(text="hello pgvecto.rs")[0]
    distance = L2Distance("embedding", target.embedding)
    docs = Documents.objects.annotate(distance=distance).order_by(distance)
    for doc in docs:
        print((doc.text, doc.embedding.to_numpy(), doc.distance))
    # The output will be:
    # ```
    # ('hello pgvecto.rs', array([1., 3., 4.], dtype=float32), 0.0)
    # ('hello postgres', array([1., 2., 4.], dtype=float32), 1.0)
    # ('hello world', array([1., 2., 3.], dtype=float32), 2.0)
    # ```
    cursor.execute("DROP TABLE IF EXISTS dense_documents")


# =================================
# Sparse Vector Example
# =================================


class Documents(models.Model):
    id = models.BigAutoField(primary_key=True)
    text = models.TextField()
    embedding = SparseVectorField(dim=60, null=True, blank=True)

    class Meta:
        app_label = "sparse"
        indexes = (
            HnswIndex(
                name="embedding_idx",
                fields=["embedding"],
                opclasses=["svector_l2_ops"],
                threads=1,
            ),
        )


class Migration(migrations.Migration):
    initial = True

    dependencies = ()

    operations = (
        VectorExtension(),
        migrations.CreateModel(
            name="documents",
            fields=(
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "text",
                    models.TextField(null=True, blank=True),
                ),
                (
                    "embedding",
                    SparseVectorField(dim=60, null=True, blank=True),
                ),
            ),
        ),
        migrations.AddIndex(
            model_name="documents",
            index=HnswIndex(
                name="embedding_idx",
                fields=["embedding"],
                opclasses=["svector_l2_ops"],
                threads=1,
            ),
        ),
    )


with connection.cursor() as cursor:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vectors")
    cursor.execute("DROP TABLE IF EXISTS dense_documents, sparse_documents")

    # Connect to the DB and create the table
    migration = Migration("initial", "sparse")
    loader = MigrationLoader(connection, replace_migrations=False)
    loader.graph.add_node(("sparse", migration.name), migration)
    sql_statements = loader.collect_sql([(migration, False)])
    cursor.execute("\n".join(sql_statements))

    # Insert 3 rows into the table
    Documents(text="hello world", embedding=SparseVector({0: 2, 1: 4, 2: 6}, 60)).save()
    Documents(
        text="hello postgres",
        embedding=SparseVector(
            coo_array(
                (np.array([2.0, 3.0]), np.array([[1, 2]])),
                shape=(60,),
            )
        ),
    ).save()
    Documents(
        text="hello pgvecto.rs",
        embedding=SparseVector.from_parts(60, [0, 2], [1.0, 3.0]),
    ).save()

    # Select the row "hello pgvecto.rs"
    target = Documents.objects.filter(text="hello pgvecto.rs")[0]
    distance = L2Distance("embedding", target.embedding)
    docs = Documents.objects.annotate(distance=distance).order_by(distance)
    for doc in docs:
        print((doc.text, doc.embedding, doc.distance))
    # The output will be:
    # ```
    # ('hello pgvecto.rs', SparseVector({0: 1.0, 2: 3.0}, 60), 0.0)
    # ('hello postgres', SparseVector({1: 2.0, 2: 3.0}, 60), 5.0)
    # ('hello world', SparseVector({0: 2.0, 1: 4.0, 2: 6.0}, 60), 26.0)
    # ```
    cursor.execute("DROP TABLE IF EXISTS sparse_documents")
