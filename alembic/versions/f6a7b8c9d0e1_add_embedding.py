"""Add embedding column to extracted_history

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-06-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import Float

revision = 'f6a7b8c9d0e1'
down_revision = 'e5f6a7b8c9d0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('extracted_history',
        sa.Column('embedding', ARRAY(Float), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('extracted_history', 'embedding')