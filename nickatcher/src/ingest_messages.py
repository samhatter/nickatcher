import asyncio
import datetime as dt
import logging
import shlex

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from client import SLSKDClient
from db.core import SessionLocal
from db.crud import add_message, get_latest_timestamp
from get_scores import get_scores


logger = logging.getLogger('nickatcher')

async def ingest_messages(slskd_client: SLSKDClient, lda: LDA, dist: np.ndarray, room_name: str, min_chunks: int):
    processing_tasks: set[asyncio.Task] = set()
    last_timestamp: dt.datetime | None = None
    async with SessionLocal() as session:
        last_timestamp = _ensure_utc(await get_latest_timestamp(session))
    while True:
        messages = []
        try:
            messages = await slskd_client.get_messages(room_name)
        except Exception as e:
            logger.error(f"Could not connect to slskd client: {e}")

        if messages:
            # keep track of active tasks to avoid unbounded growth
            processing_tasks = {task for task in processing_tasks if not task.done()}
            parsed_messages: list[tuple[dict, dt.datetime]] = []
            for message in messages:
                timestamp = _parse_timestamp(message.get('timestamp'))
                if timestamp is None:
                    logger.error(f"Invalid timestamp in message {message}")
                    continue
                parsed_messages.append((message, timestamp))

            for message, timestamp in sorted(parsed_messages, key=lambda item: item[1]):
                if last_timestamp is not None and timestamp <= last_timestamp:
                    continue
                last_timestamp = timestamp

                task = asyncio.create_task(
                    process_message(
                        slskd_client=slskd_client,
                        lda=lda,
                        dist=dist,
                        room_name=room_name,
                        min_chunks=min_chunks,
                        message=message,
                        timestamp=timestamp,
                    )
                )
                task.add_done_callback(_log_task_result)
                processing_tasks.add(task)

        await asyncio.sleep(1)


async def process_message(
    slskd_client: SLSKDClient,
    lda: LDA,
    dist: np.ndarray,
    room_name: str,
    min_chunks: int,
    message: dict,
    timestamp: dt.datetime,
):
    logger.debug(f"New message {message}")
    async with SessionLocal() as session:
        await add_message(
            session=session,
            user=message['username'],
            timestamp=timestamp,
            room_name=message['roomName'],
            content=message['message'],
        )

        await handle_commands(
            slskd_client=slskd_client,
            lda=lda,
            dist=dist,
            min_chunks=min_chunks,
            room_name=room_name,
            user=message['username'],
            text=message['message'],
        )


def _log_task_result(task: asyncio.Task):
    try:
        task.result()
    except Exception as exc:
        logger.error(f"Error processing message: {exc}")


def _parse_timestamp(raw_timestamp: str | None) -> dt.datetime | None:
    if not isinstance(raw_timestamp, str):
        return None

    normalized = raw_timestamp.replace('Z', '+00:00')
    try:
        timestamp = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None

    return _ensure_utc(timestamp)


def _ensure_utc(timestamp: dt.datetime | None) -> dt.datetime | None:
    if timestamp is None:
        return None

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)

    return timestamp.astimezone(dt.timezone.utc)

async def handle_commands(slskd_client: SLSKDClient, lda: LDA, dist: np.ndarray, min_chunks: int, room_name: str, user: str, text: str):
    try:
        parts = shlex.split(text, posix=True)
    except:
        return None
    if len(parts) == 3 and parts[0] == "nickatcher":
        user_1, user_2 = parts[1], parts[2]
        logger.info(f"User {user} called nickatcher on {user_1} and {user_2}")
        await get_scores(slskd_client=slskd_client, lda=lda, dist=dist, room_name=room_name, min_chunks=min_chunks, user_1=user_1, user_2=user_2)
    if len(parts) == 1 and parts[0] == 'nickatcher':
        logger.info(f"User {user} called nickatcher info")
        await slskd_client.send_message(room_name=room_name, message=f"""nickatcher (nickname-catcher) is a bot that calculates the similarity between the style embeddings of different chatters. To invoke say "nickatcher user_1 user_2" or "nickatcher 'user 1' 'user 2'" if the users have spaces in them. References: https://arxiv.org/html/2410.12757v1""")
    

