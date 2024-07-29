import numpy as np
from django import forms
from django.db.models import Field

from pgvecto_rs.errors import VectorDimensionError
from pgvecto_rs.types import Vector


class VectorField(Field):
    description = "Vector"
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
            return "vector"
        return "vector(%d)" % self.dim

    def from_db_value(self, value, expression, connection):
        return Vector._from_db(value)

    def to_python(self, value):
        if isinstance(value, list):
            return np.array(value, dtype=np.float32)
        return Vector._from_db(value)

    def get_prep_value(self, value):
        return Vector._to_db(value)

    def value_to_string(self, obj):
        return self.get_prep_value(self.value_from_object(obj))

    def validate(self, value, model_instance):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        super().validate(value, model_instance)

    def run_validators(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        super().run_validators(value)

    def formfield(self, **kwargs):
        return super().formfield(form_class=VectorFormField, **kwargs)


class VectorWidget(forms.TextInput):
    def format_value(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return super().format_value(value)


class VectorFormField(forms.CharField):
    widget = VectorWidget

    def has_changed(self, initial, data):
        if isinstance(initial, np.ndarray):
            initial = initial.tolist()
        return super().has_changed(initial, data)

    def to_python(self, value):
        if isinstance(value, str) and value == "":
            return None
        return super().to_python(value)
