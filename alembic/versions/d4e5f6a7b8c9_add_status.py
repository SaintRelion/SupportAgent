"""Add status to extracted_history

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-06-19
"""
from alembic import op
import sqlalchemy as sa

revision = 'd4e5f6a7b8c9'
down_revision = 'c3d4e5f6a7b8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('extracted_history',
        sa.Column(
            'status',
            sa.String(),
            nullable=False,
            server_default='pending',
        )
    )


def downgrade() -> None:
    op.drop_column('extracted_history', 'status')