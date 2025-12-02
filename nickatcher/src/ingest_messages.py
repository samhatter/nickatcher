import asyncio
import datetime as dt
import logging
import shlex

from client import SLSKDClient
from db.core import SessionLocal
from db.crud import add_message, get_latest_timestamp
from get_scores import get_scores, get_similar_users
from model_manager import ModelManager


logger = logging.getLogger('nickatcher')

async def ingest_messages(
    slskd_client: SLSKDClient,
    model_manager: ModelManager,
    room_name: str,
    min_chunks: int,
):
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
                        model_manager=model_manager,
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
    model_manager: ModelManager,
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
            model_manager=model_manager,
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

async def handle_commands(
    slskd_client: SLSKDClient,
    model_manager: ModelManager,
    min_chunks: int,
    room_name: str,
    user: str,
    text: str,
):
    try:
        parts = shlex.split(text, posix=True)
    except:
        return None
    if len(parts) == 1 and parts[0] == 'nickatcher':
        logger.info(f"User {user} called nickatcher info")
        await slskd_client.send_message(room_name=room_name, message=f"""nickatcher (nickname-catcher) is a bot that calculates the similarity between the style embeddings of different chatters. To invoke say "nickatcher user_1 user_2" or "nickatcher user_1 num_responses=10". Wrap names in quotes if they include spaces. References: https://arxiv.org/html/2410.12757v1""")

    if len(parts) >= 2 and parts[0] == "nickatcher":
        num_responses = None
        if len(parts) >= 3 and parts[2].startswith('num_responses='):
            try:
                num_responses = int(parts[2].split('=', 1)[1])
            except ValueError:
                await slskd_client.send_message(
                    room_name=room_name,
                    message="num_responses must be an integer",
                )
                return

        if num_responses is not None or len(parts) == 2:
            user_1 = parts[1]
            logger.info(
                "User %s requested similar users for %s (num_responses=%s)",
                user,
                user_1,
                num_responses,
            )
            await get_similar_users(
                slskd_client=slskd_client,
                model_manager=model_manager,
                room_name=room_name,
                target_user=user_1,
                num_responses=num_responses,
            )
            return

    if len(parts) == 3 and parts[0] == "nickatcher":
        user_1, user_2 = parts[1], parts[2]
        logger.info(f"User {user} called nickatcher on {user_1} and {user_2}")
        await get_scores(
            slskd_client=slskd_client,
            model_manager=model_manager,
            room_name=room_name,
            min_chunks=min_chunks,
            user_1=user_1,
            user_2=user_2,
        )
    

