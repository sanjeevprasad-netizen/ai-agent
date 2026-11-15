"""Async PostgreSQL / SQLite database session manager.

Supports PostgreSQL + pgvector for production, and SQLite for zero-docker local dev.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.sql import text

from app.config import settings
from app.db.models import Base, VectorType

logger = logging.getLogger(__name__)

engine_kwargs = {
    "echo": False,
    "future": True,
}

if settings.is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

engine = create_async_engine(settings.DATABASE_URL, **engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize database tables, extensions, and vector indexes."""
    logger.info("Initializing database schema...")

    if settings.is_postgres:
        async with engine.connect() as conn:
            try:
                # Check if pgvector extension is available in PostgreSQL
                res = await conn.execute(text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"))
                if res.scalar():
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    await conn.commit()
                    VectorType.HAS_PGVECTOR = True
                    logger.info("pgvector extension enabled.")
                else:
                    VectorType.HAS_PGVECTOR = False
                    logger.info("pgvector extension not installed in PostgreSQL. Using JSON vector storage.")
            except Exception as e:
                VectorType.HAS_PGVECTOR = False
                logger.warning("Could not enable pgvector extension: %s", e)

    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

        # If pgvector extension is present, create the HNSW index
        if settings.is_postgres and VectorType.HAS_PGVECTOR:
            try:
                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_policies_embedding_hnsw "
                        "ON policies USING hnsw (embedding vector_cosine_ops);"
                    )
                )
                logger.info("HNSW vector index created on policies table.")
            except Exception as e:
                logger.warning("Could not create HNSW vector index: %s", e)

    logger.info("Database initialized successfully.")


async def reset_db() -> None:
    """Drop and re-create all database tables."""
    logger.info("Dropping all existing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("Re-creating all database tables...")
    await init_db()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
