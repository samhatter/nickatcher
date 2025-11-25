import asyncio
from client import SLSKDClient
import datetime as dt
from db.core import SessionLocal
from db.crud import add_message, get_last_message

from get_scores import get_scores
from lda_state import LDAState
import logging
import numpy as np
import shlex


logger = logging.getLogger('nickatcher')


async def ingest_messages(slskd_client: SLSKDClient, lda_state: LDAState, room_name: str, min_chunks: int):
    processing_tasks: set[asyncio.Task] = set()
    while True:
        messages = []
        try:
            messages = await slskd_client.get_messages(room_name)
        except Exception as e:
            logger.error(f"Could not connect to slskd client: {e}")

        if messages:
            # keep track of active tasks to avoid unbounded growth
            processing_tasks = {task for task in processing_tasks if not task.done()}
            for message in sorted(messages, key=lambda message: message['timestamp']):
                task = asyncio.create_task(
                    process_message(
                        slskd_client=slskd_client,
                        lda_state=lda_state,
                        room_name=room_name,
                        min_chunks=min_chunks,
                        message=message,
                    )
                )
                task.add_done_callback(_log_task_result)
                processing_tasks.add(task)

        await asyncio.sleep(10)


async def process_message(slskd_client: SLSKDClient, lda_state: LDAState, room_name: str, min_chunks: int, message: dict):
    async with SessionLocal() as session:
        last_user_message = await get_last_message(session=session, user=message['username'])
    if not last_user_message:
        logger.debug(f"New user {message['username']}!")

    timestamp = dt.datetime.fromisoformat(message['timestamp'])
    if not last_user_message or timestamp > last_user_message.timestamp:
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
            lda_state=lda_state,
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


async def handle_commands(slskd_client: SLSKDClient, lda_state: LDAState, min_chunks: int, room_name: str, user: str, text: str):
    try:
        parts = shlex.split(text, posix=True)
    except:
        return None
    if not parts or parts[0] != "nickatcher":
        return None

    if len(parts) == 1:
        logger.info(f"User {user} called nickatcher info")
        await slskd_client.send_message(room_name=room_name, message=f"""nickatcher (nickname-catcher) is a bot that calculates the similarity between the style embeddings of different chatters. To invoke say "nickatcher user_1 user_2" or "nickatcher 'user 1' 'user 2'" if the users have spaces in them. You can also ask for nearest neighbors with "nickatcher user_1" (defaults to 5) or "nickatcher user_1 num_responses=10". References: https://arxiv.org/html/2410.12757v1""")
        return None

    target_user = parts[1]
    extra_parts = parts[2:]
    num_results = 5
    comparison_user = None

    for token in extra_parts:
        if token.startswith("num_responses="):
            try:
                num_results = int(token.split("=", 1)[1])
            except ValueError:
                await slskd_client.send_message(room_name=room_name, message="num_responses must be an integer.")
                return None
        else:
            comparison_user = token

    if comparison_user:
        user_1, user_2 = target_user, comparison_user
        logger.info(f"User {user} called nickatcher on {user_1} and {user_2}")
        lda, dist, _, _ = await lda_state.snapshot()
        if lda is None or dist is None:
            await slskd_client.send_message(room_name=room_name, message="Similarity model is still initializing, try again soon!")
            return None
        async with SessionLocal() as session:
            await get_scores(slskd_client=slskd_client, lda=lda, dist=dist, session=session, room_name=room_name, min_chunks=min_chunks, user_1=user_1, user_2=user_2)
    else:
        await send_nearest_neighbors(slskd_client=slskd_client, lda_state=lda_state, room_name=room_name, target_user=target_user, num_results=num_results)


async def send_nearest_neighbors(slskd_client: SLSKDClient, lda_state: LDAState, room_name: str, target_user: str, num_results: int):
    _, _, sim_matrix, users = await lda_state.snapshot()
    if sim_matrix is None or not users:
        await slskd_client.send_message(room_name=room_name, message="Similarity matrix unavailable. Please try again after the next refresh.")
        return

    if target_user not in users:
        await slskd_client.send_message(room_name=room_name, message=f"Not enough data collected for {target_user} yet.")
        return

    user_index = users.index(target_user)
    sims = sim_matrix[user_index]
    neighbor_indices = np.argsort(sims)[::-1]

    closest = []
    for idx in neighbor_indices:
        if idx == user_index:
            continue
        closest.append((users[idx], sims[idx]))
        if len(closest) >= num_results:
            break

    formatted = ", ".join([f"{u} ({score:.3f})" for u, score in closest]) if closest else "No neighbors found"
    await slskd_client.send_message(room_name=room_name, message=f"Nearest neighbors for {target_user}: {formatted}")
