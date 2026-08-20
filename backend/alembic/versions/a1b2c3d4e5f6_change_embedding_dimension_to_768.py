"""change embedding dimension to 768

Revision ID: a1b2c3d4e5f6
Revises: d2e4b7a9c1f3
Create Date: 2026-08-19 00:00:00.000000

"""

from typing import Sequence, Union

from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "d2e4b7a9c1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIMENSIONS = 768


def upgrade() -> None:
    # Drop existing embedding data and index before altering column type
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding DROP NOT NULL")
    op.execute("DELETE FROM document_chunks WHERE embedding IS NOT NULL")
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=Vector(EMBEDDING_DIMENSIONS),
        nullable=True,
    )
    # Recreate HNSW index for new dimension
    op.execute(
        """
        CREATE INDEX ix_document_chunks_embedding_hnsw
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=Vector(1536),
        nullable=True,
    )
    op.execute(
        """
        CREATE INDEX ix_document_chunks_embedding_hnsw
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )
