"""Alembic environment for the TORX contextual sovereignty agent.

Two things differ from the stock template, both on purpose:

* **The URL comes from the environment, never from a committed file.**
  ``alembic.ini`` ships with an empty ``sqlalchemy.url`` so a connection string
  — which contains a password — cannot be committed by accident.
  ``TORX_DATABASE_URL`` is the production knob and ``TORX_TEST_DATABASE_URL``
  the test one; the ini value is the last resort, and running with none of them
  fails loudly rather than silently targeting localhost.

* **Offline mode is a first-class path.** ``alembic upgrade head --sql`` renders
  the whole schema, including the row-level-security policies and append-only
  triggers, without a server. That is how this migration is reviewed and how it
  is tested on machines that have no PostgreSQL.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# The repo root, so ``import src.persistence.models`` resolves the same way it
# does for the application and the tests.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.persistence import models  # noqa: E402

config = context.config

if config.config_file_name is not None and config.get_section("loggers"):
    fileConfig(config.config_file_name)

# Importing models already asserted that this metadata matches the model card,
# so autogenerate diffs against a schema the card agrees with.
target_metadata = models.metadata


def _database_url() -> str:
    url = (
        os.environ.get("TORX_DATABASE_URL")
        or os.environ.get("TORX_TEST_DATABASE_URL")
        or config.get_main_option("sqlalchemy.url", "")
    )
    if not url:
        raise RuntimeError(
            "no database URL: set TORX_DATABASE_URL (or TORX_TEST_DATABASE_URL), "
            "or pass -x url=... ; alembic.ini deliberately ships without one so "
            "a password cannot be committed"
        )
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
