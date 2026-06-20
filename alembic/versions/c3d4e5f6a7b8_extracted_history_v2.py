"""Refactor extracted_history: drop question/answer, add summary and keywords array

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-06-19
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'c3d4e5f6a7b8'
down_revision = 'b2c3d4e5f6a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column('extracted_history', 'question')
    op.drop_column('extracted_history', 'answer')
    op.add_column('extracted_history',
        sa.Column('summary', sa.Text(), nullable=True)
    )
    op.add_column('extracted_history',
        sa.Column('keywords', postgresql.ARRAY(sa.Text()), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('extracted_history', 'summary')
    op.drop_column('extracted_history', 'keywords')
    op.add_column('extracted_history',
        sa.Column('question', sa.Text(), nullable=True)
    )
    op.add_column('extracted_history',
        sa.Column('answer', sa.Text(), nullable=True)
    )