import numpy as np
from django import forms
from django.db.models import Field

from pgvecto_rs.errors import VectorDimensionError
from pgvecto_rs.types import BinaryVector


class BinaryVectorField(Field):
    description = "BinaryVector"
    empty_strings_allowed = False

    def __init__(self, *args, dim=None, **kwargs):
        if dim is not None and (dim < 0 or dim > 65535):  # noqa: PLR2004
            raise VectorDimensionError(dim)
        self.dim = dim
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.dim is not None:
            kwargs["dim"] = self.dim
        return name, path, args, kwargs

    def db_type(self, connection):
        if self.dim is None:
            return "bvector"
        return "bvector(%d)" % self.dim

    def from_db_value(self, value, expression, connection):
        return BinaryVector._from_db(value)

    def to_python(self, value):
        if value is None or isinstance(value, BinaryVector):
            return value
        elif isinstance(value, str):
            return BinaryVector._from_db(value)
        else:
            return BinaryVector(value)

    def get_prep_value(self, value):
        return BinaryVector._to_db(value)

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))

    def formfield(self, **kwargs):
        return super().formfield(form_class=BinaryVectorFormField, **kwargs)


class BinaryVectorWidget(forms.TextInput):
    def format_value(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return super().format_value(value)


class BinaryVectorFormField(forms.CharField):
    widget = BinaryVectorWidget

    def to_python(self, value):
        if isinstance(value, str) and value == "":
            return None
        return super().to_python(value)
