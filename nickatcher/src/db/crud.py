import datetime as dt
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Message

async def add_message(
    session: AsyncSession,
    *,
    user: str,
    timestamp: dt.datetime,
    room_name: str,
    content: str,
) -> Message:
    msg = Message(user=user, content=content, room=room_name, timestamp=timestamp)
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg

async def get_last_message(
    session: AsyncSession,
    user: str
):
    stmt = select(Message).where(Message.user == user).order_by(Message.id.desc()).limit(1)
    row = await session.execute(stmt)
    return row.scalar_one_or_none()


async def get_latest_timestamp(session: AsyncSession) -> dt.datetime | None:
    stmt = select(Message.timestamp).order_by(Message.timestamp.desc()).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

async def list_messages(
    session: AsyncSession,
    user: str | None = None,
    limit: int = 50,
) -> Sequence[Message]:
    stmt = select(Message).order_by(Message.id).limit(limit)
    if user:
        stmt = stmt.where(Message.user == user)
    rows = await session.execute(stmt)
    return rows.scalars().all()

async def count_messages(
    session: AsyncSession,
    user: str | None = None,
) -> int:
    stmt = select(func.count()).select_from(Message)
    if user:
        stmt = stmt.where(Message.user == user)

    return (await session.scalar(stmt)) or 0


async def count_unique_users(session: AsyncSession) -> int:
    stmt = select(func.count(func.distinct(Message.user)))
    return (await session.scalar(stmt)) or 0