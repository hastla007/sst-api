"""
Database models for STT API
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()


class SubscriptionTier(str, enum.Enum):
    """Subscription tier types"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class TranscriptionStatus(str, enum.Enum):
    """Transcription status types"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Organization(Base):
    """Organization/Multi-Tenant model"""
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    subscription_tier = Column(Enum(SubscriptionTier), default=SubscriptionTier.FREE)

    # Quota settings
    monthly_quota = Column(Integer, default=1000)  # minutes
    used_quota = Column(Float, default=0.0)  # minutes
    quota_reset_date = Column(DateTime, nullable=False)

    # Billing
    billing_email = Column(String(255))
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    projects = relationship("Project", back_populates="organization", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="organization", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Organization(id={self.id}, name={self.name})>"


class Project(Base):
    """Project model for organizing transcriptions"""
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True)
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    organization = relationship("Organization", back_populates="projects")
    transcriptions = relationship("Transcription", back_populates="project", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_org_project', 'organization_id', 'name'),
    )

    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name})>"


class APIKey(Base):
    """API Key model with organization association"""
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255))  # Friendly name for the key

    # Permissions and limits
    is_active = Column(Boolean, default=True)
    rate_limit = Column(Integer, default=10)  # requests per minute

    # Usage tracking
    total_requests = Column(Integer, default=0)
    last_used_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)  # Optional expiration

    # Relationships
    organization = relationship("Organization", back_populates="api_keys")

    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.name})>"


class Transcription(Base):
    """Transcription record model"""
    __tablename__ = "transcriptions"

    id = Column(String(36), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"))
    correlation_id = Column(String(255), unique=True, index=True)

    # File information
    filename = Column(String(255))
    file_size = Column(Integer)  # bytes
    file_hash = Column(String(64), index=True)  # SHA256
    audio_duration = Column(Float)  # seconds

    # Processing information
    status = Column(Enum(TranscriptionStatus), default=TranscriptionStatus.PENDING, index=True)
    language = Column(String(10))  # ISO 639-1 code
    detected_language = Column(String(10))  # Auto-detected language
    model_used = Column(String(50))  # whisper model size

    # Results
    text = Column(Text)
    segments = Column(JSON)  # Full segments with timestamps
    confidence_scores = Column(JSON)  # Confidence scores per segment

    # Advanced features
    speakers = Column(JSON)  # Speaker diarization results
    translation = Column(JSON)  # Translation results
    sentiment = Column(JSON)  # Sentiment analysis results

    # Processing metadata
    processing_time = Column(Float)  # seconds
    webhook_url = Column(String(500))
    webhook_sent = Column(Boolean, default=False)

    # Custom vocabulary/prompts
    initial_prompt = Column(Text)

    # Error information
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Relationships
    project = relationship("Project", back_populates="transcriptions")

    __table_args__ = (
        Index('idx_transcription_search', 'project_id', 'status', 'created_at'),
        Index('idx_transcription_text', 'text', postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'}),
    )

    def __repr__(self):
        return f"<Transcription(id={self.id}, filename={self.filename}, status={self.status})>"


class UsageLog(Base):
    """Usage logging for billing and analytics"""
    __tablename__ = "usage_logs"

    id = Column(String(36), primary_key=True)
    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    transcription_id = Column(String(36), ForeignKey("transcriptions.id"))

    # Usage details
    audio_duration = Column(Float)  # minutes
    processing_time = Column(Float)  # seconds
    model_used = Column(String(50))

    # Features used
    used_diarization = Column(Boolean, default=False)
    used_translation = Column(Boolean, default=False)
    used_sentiment = Column(Boolean, default=False)

    # Cost calculation
    cost = Column(Float, default=0.0)  # in cents

    # Timestamp
    created_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_usage_org_date', 'organization_id', 'created_at'),
    )

    def __repr__(self):
        return f"<UsageLog(id={self.id}, organization_id={self.organization_id})>"


class WebhookLog(Base):
    """Webhook delivery tracking"""
    __tablename__ = "webhook_logs"

    id = Column(String(36), primary_key=True)
    transcription_id = Column(String(36), ForeignKey("transcriptions.id"), nullable=False)
    webhook_url = Column(String(500), nullable=False)

    # Attempt tracking
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)

    # Status
    success = Column(Boolean, default=False)
    status_code = Column(Integer)
    response_body = Column(Text)
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    last_attempt_at = Column(DateTime)
    next_retry_at = Column(DateTime)
    completed_at = Column(DateTime)

    def __repr__(self):
        return f"<WebhookLog(id={self.id}, success={self.success}, attempts={self.attempts})>"
