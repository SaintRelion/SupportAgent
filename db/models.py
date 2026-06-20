from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text,
    DateTime, Boolean, Date, Float
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ExtractedHistory(Base):
    """
    Stores knowledge extracted from channel history.
    Each row is one resolved exchange — situation, context, decision — as a narrative summary.

    view_count: how many times an admin has opened this entry in /review_knowledge.
                -1 = soft deleted (excluded from agent queries and review list).
    """
    __tablename__ = "extracted_history"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    source_date  = Column(Date, nullable=False)
    topic        = Column(String, nullable=True)
    conversation = Column(Text, nullable=False)
    summary      = Column(Text, nullable=True)
    keywords     = Column(ARRAY(Text), nullable=True)
    view_count   = Column(Integer, nullable=False, default=0)
    embedding    = Column(ARRAY(Float), nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)


class CallFeedback(Base):
    __tablename__ = "call_feedback"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    discord_channel_id = Column(BigInteger, nullable=False)
    requested_by       = Column(BigInteger, nullable=False)
    recording_url      = Column(Text, nullable=False)
    filename           = Column(String, nullable=True)
    transcript         = Column(Text, nullable=True)
    feedback           = Column(Text, nullable=True)
    error              = Column(Text, nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow, nullable=False)