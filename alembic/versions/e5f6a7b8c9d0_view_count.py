"""Replace status with view_count on extracted_history

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-06-19
"""
from alembic import op
import sqlalchemy as sa

revision = 'e5f6a7b8c9d0'
down_revision = 'd4e5f6a7b8c9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column('extracted_history', 'status')
    op.add_column('extracted_history',
        sa.Column(
            'view_count',
            sa.Integer(),
            nullable=False,
            server_default='0',
        )
    )


def downgrade() -> None:
    op.drop_column('extracted_history', 'view_count')
    op.add_column('extracted_history',
        sa.Column(
            'status',
            sa.String(),
            nullable=False,
            server_default='pending',
        )
    )