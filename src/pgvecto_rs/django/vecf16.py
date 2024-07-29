import numpy as np
from django import forms
from django.db.models import Field

from pgvecto_rs.errors import VectorDimensionError
from pgvecto_rs.types import Float16Vector


class Float16VectorField(Field):
    description = "Float16Vector"
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
            return "vecf16"
        return "vecf16(%d)" % self.dim

    def from_db_value(self, value, expression, connection):
        return Float16Vector._from_db(value)

    def to_python(self, value):
        if value is None or isinstance(value, Float16Vector):
            return value
        elif isinstance(value, str):
            return Float16Vector._from_db(value)
        else:
            return Float16Vector(value)

    def get_prep_value(self, value):
        return Float16Vector._to_db(value)

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
        return super().formfield(form_class=Float16VectorFormField, **kwargs)


class Float16VectorWidget(forms.TextInput):
    def format_value(self, value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return super().format_value(value)


class Float16VectorFormField(forms.CharField):
    widget = Float16VectorWidget

    def has_changed(self, initial, data):
        if isinstance(initial, np.ndarray):
            initial = initial.tolist()
        return super().has_changed(initial, data)

    def to_python(self, value):
        if isinstance(value, str) and value == "":
            return None
        return super().to_python(value)
