"""Drop keywords column from extracted_history

Revision ID: g7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-06-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

revision = 'g7b8c9d0e1f2'
down_revision = 'f6a7b8c9d0e1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column('extracted_history', 'keywords')


def downgrade() -> None:
    op.add_column('extracted_history',
        sa.Column('keywords', ARRAY(sa.Text()), nullable=True)
    )