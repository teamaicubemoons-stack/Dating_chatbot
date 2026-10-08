"""
api/dependencies.py
--------------------
FastAPI dependency injection utilities.

Centralizes shared dependencies (DB session, settings, auth if added later)
so they can be injected into any route without importing app internals directly.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config.settings import settings
from app.models.db_models import Base

# Create engine once at module import time — thread-safe singleton
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # required for SQLite
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they don't exist. Called once at app startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy session per request.
    The session is automatically closed after the request completes.
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
