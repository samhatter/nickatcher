import logging
import os
from sqlalchemy.ext.asyncio import (
    AsyncAttrs, create_async_engine, async_sessionmaker, AsyncSession
)

logger = logging.getLogger('nickatcher')

pg_user = os.getenv("POSTGRES_USER", "")
if pg_user == "":
    logger.warning("POSTGRES_USER not found in environment")    
pg_pass = os.getenv("POSTGRES_PASSWORD", "")
if pg_pass == "":
    logger.warning("POSTGRES_PASSWORD not found in environment")    
pg_db = os.getenv("POSTGRES_DB", "")
if pg_db == "":
    logger.warning("POSTGRES_DB not found in environment")
pg_port = os.getenv("POSTGRES_PORT", "")
if pg_db == "":
    logger.warning("POSTGRES_PORT not found in environment")    

database_url = f"postgresql+asyncpg://{pg_user}:{pg_pass}@nickatcher-db:{pg_port}/{pg_db}"

engine = create_async_engine(database_url, echo=False, pool_pre_ping=True)

SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)
