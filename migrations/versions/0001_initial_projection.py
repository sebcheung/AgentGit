"""initial projection.

Revision ID: 0001
Revises:
Create Date: 2026-09-04 14:43:32.473150
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create every table in ``memgit.pg.schema``."""
    op.create_table(
        "commits",
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.Column("tree", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("author", sa.String(length=255), nullable=False),
        sa.Column("committed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("is_root", sa.Boolean(), nullable=False),
        sa.Column("is_merge", sa.Boolean(), nullable=False),
        sa.Column(
            "metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("hash"),
    )
    op.create_index("ix_commits_committed_at", "commits", ["committed_at"], unique=False)
    op.create_table(
        "facts",
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("predicate", sa.Text(), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("asserted_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("source_text", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("hash"),
    )
    op.create_index("ix_facts_subject_predicate", "facts", ["subject", "predicate"], unique=False)
    op.create_table(
        "projection",
        sa.Column("id", sa.SmallInteger(), nullable=False),
        sa.Column("format_version", sa.Integer(), nullable=False),
        sa.Column(
            "projected_refs",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("projected_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_projection_singleton"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "refs",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("target_hash", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )
    op.create_table(
        "commit_parents",
        sa.Column("child_hash", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.SmallInteger(), nullable=False),
        sa.Column("parent_hash", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["child_hash"], ["commits.hash"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("child_hash", "ordinal"),
    )
    op.create_index("ix_commit_parents_parent_hash", "commit_parents", ["parent_hash"], unique=False)
    op.create_table(
        "key_deltas",
        sa.Column("commit_hash", sa.String(length=64), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("predicate", sa.Text(), nullable=False),
        sa.Column("fact_hash", sa.String(length=64), nullable=False),
        sa.Column("op", sa.String(length=1), nullable=False),
        sa.CheckConstraint("op IN ('+', '-')", name="ck_key_deltas_op"),
        sa.ForeignKeyConstraint(["commit_hash"], ["commits.hash"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("commit_hash", "subject", "predicate", "fact_hash", "op"),
    )
    op.create_index(
        "ix_key_deltas_subject_predicate_commit",
        "key_deltas",
        ["subject", "predicate", "commit_hash"],
        unique=False,
    )
    op.create_table(
        "tree_entries",
        sa.Column("tree_hash", sa.String(length=64), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("predicate", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.SmallInteger(), nullable=False),
        sa.Column("fact_hash", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["fact_hash"], ["facts.hash"]),
        sa.PrimaryKeyConstraint("tree_hash", "subject", "predicate", "ordinal"),
    )


def downgrade() -> None:
    """Drop every table this revision created, in dependency order."""
    op.drop_table("tree_entries")
    op.drop_index("ix_key_deltas_subject_predicate_commit", table_name="key_deltas")
    op.drop_table("key_deltas")
    op.drop_index("ix_commit_parents_parent_hash", table_name="commit_parents")
    op.drop_table("commit_parents")
    op.drop_table("refs")
    op.drop_table("projection")
    op.drop_index("ix_facts_subject_predicate", table_name="facts")
    op.drop_table("facts")
    op.drop_index("ix_commits_committed_at", table_name="commits")
    op.drop_table("commits")
