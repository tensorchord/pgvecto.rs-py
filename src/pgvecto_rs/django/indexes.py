from django.contrib.postgres.indexes import PostgresIndex

from pgvecto_rs.types import IndexOption


class Index(PostgresIndex):
    suffix = "vectors"

    def __init__(
        self,
        *expressions,
        option: IndexOption,
        **kwargs,
    ):
        self.option = option
        super().__init__(*expressions, **kwargs)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs["option"] = self.option
        return path, args, kwargs

    def get_with_params(self):
        with_params = [f"options = $${self.option.dumps()}$$"]
        return with_params
