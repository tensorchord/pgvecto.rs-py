import numpy as np
from django import forms
from django.db.models import Field

from pgvecto_rs.errors import SparseDimensionError
from pgvecto_rs.types import SparseVector


class SparseVectorField(Field):
    description = "SparseVector"
    empty_strings_allowed = False

    def __init__(self, *args, dim=None, **kwargs):
        if dim is not None and (dim < 0 or dim > 1048575):  # noqa: PLR2004
            raise SparseDimensionError(dim)
        self.dim = dim
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.dim is not None:
            kwargs["dim"] = self.dim
        return name, path, args, kwargs

    def db_type(self, connection):
        if self.dim is None:
            return "svector"
        return "svector(%d)" % self.dim

    def from_db_value(self, value, expression, connection):
        return SparseVector._from_db(value)

    def to_python(self, value):
        if value is None or isinstance(value, SparseVector):
            return value
        elif isinstance(value, str):
            return SparseVector._from_db(value)
        else:
            return SparseVector(value)

    def get_prep_value(self, value):
        return SparseVector._to_db(value)

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))

    def formfield(self, **kwargs):
        return super().formfield(form_class=SparseVectorFormField, **kwargs)


class SparseVectorWidget(forms.TextInput):
    def format_value(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return super().format_value(value)


class SparseVectorFormField(forms.CharField):
    widget = SparseVectorWidget

    def to_python(self, value):
        if isinstance(value, str) and value == "":
            return None
        return super().to_python(value)
