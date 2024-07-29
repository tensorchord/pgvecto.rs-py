import django
import numpy as np
import pytest
from django.conf import settings
from django.db import connection, migrations, models
from django.db.backends.utils import CursorWrapper
from django.forms import ModelForm

from pgvecto_rs.django import (
    BinaryVectorField,
    CosineDistance,
    FlatIndex,
    Float16VectorField,
    HnswIndex,
    JaccardDistance,
    L2Distance,
    MaxInnerProduct,
    SparseVectorField,
    VectorExtension,
    VectorField,
)
from pgvecto_rs.types import BinaryVector
from tests import (
    BINARY_VECTORS,
    COSINE_DIS_OP,
    DATABASES,
    FLOAT16_VECTORS,
    JACCARD_DIS_OP,
    L2_DIS_OP,
    MAX_INNER_PROD_OP,
    SPARSE_VECTORS,
    VECTORS,
    cosine_distance,
    jaccard_distance,
    l2_distance,
    max_inner_product,
)

settings.configure(DATABASES=DATABASES)
django.setup()


class Item(models.Model):
    embedding = VectorField(dim=3, null=True, blank=True)
    float16_embedding = Float16VectorField(dim=3, null=True, blank=True)
    binary_embedding = BinaryVectorField(dim=3, null=True, blank=True)
    sparse_embedding = SparseVectorField(dim=3, null=True, blank=True)

    class Meta:
        app_label = "django_app"
        indexes = (
            FlatIndex(
                name="emb_idx_1",
                fields=["embedding"],
                opclasses=["vector_l2_ops"],
                threads=1,
            ),
            HnswIndex(
                name="emb_idx_2",
                fields=["embedding"],
                m=16,
                ef_construction=100,
                threads=1,
                opclasses=["vector_l2_ops"],
            ),
        )


class Migration(migrations.Migration):
    initial = True

    dependencies = ()

    operations = (
        VectorExtension(),
        migrations.CreateModel(
            name="Item",
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
                    "embedding",
                    VectorField(dim=3, null=True, blank=True),
                ),
                (
                    "float16_embedding",
                    Float16VectorField(dim=3, null=True, blank=True),
                ),
                (
                    "binary_embedding",
                    BinaryVectorField(dim=3, null=True, blank=True),
                ),
                (
                    "sparse_embedding",
                    SparseVectorField(dim=3, null=True, blank=True),
                ),
            ),
        ),
        migrations.AddIndex(
            model_name="item",
            index=FlatIndex(
                name="emb_idx_1",
                fields=["embedding"],
                opclasses=["vector_l2_ops"],
                threads=1,
            ),
        ),
        migrations.AddIndex(
            model_name="item",
            index=HnswIndex(
                name="emb_idx_2",
                fields=["embedding"],
                m=16,
                ef_construction=100,
                opclasses=["vector_l2_ops"],
            ),
        ),
    )


@pytest.fixture()
def session():
    """Connect to the test db pointed by the URL. Can check more details
    in `tests/__init__.py`
    """
    from django.db.migrations.loader import MigrationLoader

    migration = Migration("initial", "django_app")
    loader = MigrationLoader(connection, replace_migrations=False)
    loader.graph.add_node(("django_app", migration.name), migration)
    sql_statements = loader.collect_sql([(migration, False)])

    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vectors")
        cursor.execute("DROP TABLE IF EXISTS django_app_item")
        cursor.execute("\n".join(sql_statements))
        create_items()
        yield cursor
        cursor.execute("DROP TABLE IF EXISTS django_app_item")


def create_items():
    for i, (v, sv, f16v, bv) in enumerate(
        zip(VECTORS, SPARSE_VECTORS, FLOAT16_VECTORS, BINARY_VECTORS)
    ):
        Item(
            id=i,
            embedding=v,
            float16_embedding=f16v,
            binary_embedding=bv,
            sparse_embedding=sv,
        ).save()


class VectorForm(ModelForm):
    class Meta:
        model = Item
        fields = ("embedding",)


class Float16VectorForm(ModelForm):
    class Meta:
        model = Item
        fields = ("float16_embedding",)


class BitForm(ModelForm):
    class Meta:
        model = Item
        fields = ("binary_embedding",)


class SparseVectorForm(ModelForm):
    class Meta:
        model = Item
        fields = ("sparse_embedding",)


def test_l2_distance(session: CursorWrapper):
    distance = L2Distance("embedding", L2_DIS_OP)
    items = Item.objects.annotate(distance=distance).order_by(distance)
    for item in items:
        expect = l2_distance(np.array(L2_DIS_OP), item.embedding.to_numpy())
        assert np.allclose(expect, item.distance, atol=1e-10)


def test_max_inner_product(session: CursorWrapper):
    distance = MaxInnerProduct("embedding", MAX_INNER_PROD_OP)
    items = Item.objects.annotate(distance=distance).order_by(distance)
    for item in items:
        expect = max_inner_product(
            np.array(MAX_INNER_PROD_OP), item.embedding.to_numpy()
        )
        assert np.allclose(expect, item.distance, atol=1e-10)


def test_cosine_distance(session: CursorWrapper):
    distance = CosineDistance("embedding", COSINE_DIS_OP)
    items = Item.objects.annotate(distance=distance).order_by(distance)
    for item in items:
        expect = cosine_distance(np.array(COSINE_DIS_OP), item.embedding.to_numpy())
        assert np.allclose(expect, item.distance, atol=1e-10)


def test_binary_jaccard_distance(session: CursorWrapper):
    distance = JaccardDistance("binary_embedding", BinaryVector(JACCARD_DIS_OP))
    items = Item.objects.annotate(distance=distance).order_by(distance)
    for item in items:
        expect = jaccard_distance(JACCARD_DIS_OP, item.binary_embedding.to_numpy())
        assert np.allclose(expect, item.distance, atol=1e-10)


def test_missing(session: CursorWrapper):
    session.execute("TRUNCATE django_app_item")
    Item().save()
    assert Item.objects.first().embedding is None
    assert Item.objects.first().float16_embedding is None
    assert Item.objects.first().binary_embedding is None
    assert Item.objects.first().sparse_embedding is None


def test_float16_vector(session: CursorWrapper):
    Item(id=1, float16_embedding=FLOAT16_VECTORS[0]).save()
    item = Item.objects.get(pk=1)
    assert item.float16_embedding.to_list() == FLOAT16_VECTORS[0].to_list()


def test_sparse_vector(session: CursorWrapper):
    Item(id=1, sparse_embedding=SPARSE_VECTORS[0]).save()
    item = Item.objects.get(pk=1)
    assert np.allclose(
        item.sparse_embedding.to_numpy(), SPARSE_VECTORS[0].to_numpy(), atol=1e-10
    )


def test_filter(session: CursorWrapper):
    distance = L2Distance("embedding", [1, 1, 1])
    items = Item.objects.alias(distance=distance).filter(distance__lt=1)
    assert [v.id for v in items] == [3]


def test_clean(session: CursorWrapper):
    session.execute("TRUNCATE django_app_item")
    item = Item(
        id=1,
        embedding=VECTORS[0],
        float16_embedding=FLOAT16_VECTORS[0],
        binary_embedding=BINARY_VECTORS[0],
        sparse_embedding=SPARSE_VECTORS[0],
    )
    item.full_clean()


def test_get_or_create(session: CursorWrapper):
    Item.objects.get_or_create(embedding=[1, 2, 3])


def test_vector_form(session: CursorWrapper):
    form = VectorForm(data={"embedding": "[1, 2, 3]"})
    assert form.is_valid()
    assert 'value="[1, 2, 3]"' in form.as_div()


def test_vector_form_instance(session):
    Item(id=1, embedding=[1, 2, 3]).save()
    item = Item.objects.get(pk=1)
    form = VectorForm(instance=item)
    assert ' value="Vector([1.0, 2.0, 3.0])"' in form.as_div()


def test_vector_form_save(session):
    Item(id=1, embedding=[1, 2, 3]).save()
    item = Item.objects.get(pk=1)
    form = VectorForm(instance=item, data={"embedding": "[4, 5, 6]"})
    assert form.has_changed()
    assert form.is_valid()
    assert form.save()
    assert [4, 5, 6] == Item.objects.get(pk=1).embedding.to_numpy().tolist()


def test_vector_form_save_missing(session):
    Item(id=1).save()
    item = Item.objects.get(pk=1)
    form = VectorForm(instance=item, data={"embedding": ""})
    assert form.is_valid()
    assert form.save()
    assert Item.objects.get(pk=1).embedding is None
