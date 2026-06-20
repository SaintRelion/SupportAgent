"""add escalations, extracted_rules, call_feedback
Revision ID: 97003ee78da1
Revises: 
Create Date: 2026-06-15 16:56:32.756844
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '97003ee78da1'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('call_feedback',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('discord_channel_id', sa.BigInteger(), nullable=False),
    sa.Column('requested_by', sa.BigInteger(), nullable=False),
    sa.Column('recording_url', sa.Text(), nullable=False),
    sa.Column('filename', sa.String(), nullable=True),
    sa.Column('transcript', sa.Text(), nullable=True),
    sa.Column('feedback', sa.Text(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('escalations',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('thread_id', sa.String(), nullable=False),
    sa.Column('discord_channel_id', sa.BigInteger(), nullable=False),
    sa.Column('discord_user_id', sa.BigInteger(), nullable=False),
    sa.Column('user_message', sa.Text(), nullable=False),
    sa.Column('escalation_msg_id', sa.BigInteger(), nullable=True),
    sa.Column('resolved', sa.Boolean(), nullable=False),
    sa.Column('admin_user_id', sa.BigInteger(), nullable=True),
    sa.Column('admin_response', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('resolved_at', sa.DateTime(), nullable=True),
    # Verification columns — null for normal escalations
    sa.Column('is_verification', sa.Boolean(), nullable=False, server_default='false'),
    sa.Column('verification_msg_id', sa.BigInteger(), nullable=True),
    sa.Column('verified', sa.Boolean(), nullable=True),
    sa.Column('bot_answer', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_escalations_thread_id'), 'escalations', ['thread_id'], unique=False)
    op.create_table('extracted_rules',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('escalation_id', sa.Integer(), nullable=True),
    sa.Column('rule_text', sa.Text(), nullable=False),
    sa.Column('approved', sa.Boolean(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('approved_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['escalation_id'], ['escalations.id'], ),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('extracted_rules')
    op.drop_index(op.f('ix_escalations_thread_id'), table_name='escalations')
    op.drop_table('escalations')
    op.drop_table('call_feedback')