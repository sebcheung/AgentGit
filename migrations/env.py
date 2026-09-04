"""Alembic's entry point -- wired to memgit.pg's own engine and metadata, not alembic.ini's URL."""

from logging.config import fileConfig

from alembic import context

from memgit.pg.engine import engine_from_env
from memgit.pg.schema import metadata as target_metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """Emit migration SQL without a live connection -- ``alembic upgrade --sql``."""
    context.configure(
        url=engine_from_env().url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against ``$DATABASE_URL`` -- the same source every other pg-needing command uses."""
    connectable = engine_from_env()

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
