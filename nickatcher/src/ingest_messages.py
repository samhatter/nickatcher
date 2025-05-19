import asyncio
from client import SLSKDClient
import datetime as dt
from db.core import SessionLocal
from db.crud import add_message, count_messages, count_unique_users, get_last_message
from get_embeddings import get_embeddings
from get_scores import get_scores
import logging
import shlex
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger('nickatcher')

async def ingest_messages(slskd_client: SLSKDClient, room_name: str, max_tokens: int, min_chunks: int):
    async with SessionLocal() as session:
        while True:
            messages = []
            try:
                messages = await slskd_client.get_messages(room_name)
            except Exception as e:
                logger.error(f"Could not connect to slskd client: {e}")
            
            if messages:
                for message in sorted(messages, key=lambda message: message['timestamp']):
                    last_user_message = await get_last_message(session=session, user=message['username'])
                    if not last_user_message:
                        logger.debug(f"New user {message['username']}!")
                    timestamp = dt.datetime.fromisoformat(message['timestamp'])
                    if not last_user_message or timestamp > last_user_message.timestamp:
                        logger.debug(f"New message {message}")
                        await add_message(session=session, user=message['username'], timestamp=timestamp, room_name=message['roomName'], content=message['message'])
                        parsed = await parse_commands(slskd_client=slskd_client, room_name=room_name, user=message['username'], text=message['message'])
                        if parsed:
                            user_1, user_2 = parsed
                            logger.debug(f"User {message['username']} called nickatcher on {user_1} and {user_2}")
                            await get_scores(slskd_client=slskd_client, session=session, room_name=room_name, max_tokens=max_tokens, min_chunks=min_chunks, user_1=user_1, user_2=user_2)
            num_messages = await count_messages(session=session)
            num_users = await count_unique_users(session=session)
            logger.info(f"Message history has {num_messages} messages on {num_users} users")
            await asyncio.sleep(10)

async def parse_commands(slskd_client: SLSKDClient, room_name: str, user: str, text: str):
    try:
        parts = shlex.split(text, posix=True)
    except:
        return None
    if len(parts) == 3 and parts[0] == "nickatcher":
        return parts[1], parts[2]
    if len(parts) == 1 and parts[0] == 'nickatcher':
        logger.info(f"User {user} called nickatcher info")
        await slskd_client.send_message(room_name=room_name, message=f"""nickatcher (nickname-catcher) is a bot that calculates the similarity between the style embeddings of different chatters. To invoke say "nickatcher user_1 user_2" or "nickatcher 'user 1' 'user 2'" if the users have spaces in them. References: https://arxiv.org/html/2410.12757v1""")
    