"""Replace escalations/extracted_rules with extracted_history

Revision ID: b2c3d4e5f6a7
Revises: 97003ee78da1
Create Date: 2026-06-17
"""
from alembic import op
import sqlalchemy as sa

revision = 'b2c3d4e5f6a7'
down_revision = '97003ee78da1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('extracted_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('source_date', sa.Date(), nullable=False),
        sa.Column('topic', sa.String(), nullable=True),
        sa.Column('conversation', sa.Text(), nullable=False),
        sa.Column('question', sa.Text(), nullable=True),
        sa.Column('answer', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.drop_table('extracted_rules')
    op.drop_index('ix_escalations_thread_id', table_name='escalations')
    op.drop_table('escalations')


def downgrade() -> None:
    op.drop_table('extracted_history')