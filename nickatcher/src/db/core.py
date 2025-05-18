import logging
import os
from sqlalchemy.ext.asyncio import (
    AsyncAttrs, create_async_engine, async_sessionmaker, AsyncSession
)

logger = logging.getLogger('nickatcher')

pg_user = os.getenv("POSTGRES_USER", "")
if pg_user == "":
    logger.warning("Could not find POSGRES_USER in env")    
pg_pass = os.getenv("POSTGRES_PASSWORD", "")
if pg_pass == "":
    logger.warning("Could not find POSGRES_PASSWORD in env")    
pg_db = os.getenv("POSTGRES_DB", "")
if pg_db == "":
    logger.warning("Could not find POSGRES_DB in env")
pg_port = os.getenv("POSTGRES_PORT", "")
if pg_db == "":
    logger.warning("Could not find POSGRES_PORT in env")    

database_url = f"postgresql+asyncpg://{pg_user}:{pg_pass}@nickatcher-db:{pg_port}/{pg_db}"

engine = create_async_engine(database_url, echo=False, pool_pre_ping=True)

SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)
