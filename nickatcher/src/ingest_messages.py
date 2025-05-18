import asyncio
from client import SLSKDClient
from get_embeddings import get_embeddings
from get_scores import get_scores
import logging
import shlex

logger = logging.getLogger('nickatcher')

async def ingest_messages(slskd_client: SLSKDClient, room_name: str, message_history: dict, history_window: int):
    while True:
        messages = []
        try:
            messages = await slskd_client.get_messages(room_name)
        except Exception as e:
            logger.error(f"Could not connect to slskd client: {e}")
        
        if messages:
            for message in sorted(messages, key=lambda message: message['timestamp']):
                if message['username'] not in message_history or message['timestamp'] > message_history[message['username']][-1]['timestamp']:
                    logger.debug(f"New message {message}")
                    if message['username'] not in message_history:
                        message_history[message['username']] = [message]
                    else:
                        message_history[message['username']].append(message)
                    if len(message_history[message['username']]) > history_window:
                        message_history[message['username']] = message_history[1:]
                    parsed = parse_commands(message['message'])
                    if parsed:
                        user_1, user_2 = parsed
                        logger.debug(f"User {message['username']} called nickatcher on {user_1} and {user_2}")
                        user_messages_1 = message_history.get(user_1, None)
                        user_messages_2 = message_history.get(user_2, None)
                        if user_messages_1 is None and user_messages_2 is None:
                            await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_1} or {user_2}, check your spelling!")
                        elif user_messages_1 is None:
                            await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_1}, check your spelling!")
                        elif user_messages_2 is None:
                            await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_2}, check your spelling!")
                        else:
                            user_embeddings_1 = get_embeddings(user_messages_1)
                            user_embeddings_2 = get_embeddings(user_messages_2)
                            score = get_scores(user_embeddings_1, user_embeddings_2)  
                            logger.debug(f"Embedding distance for score for {user_1}, {user_2}: {score}")
                            await slskd_client.send_message(room_name=room_name, message=f"Embedding distance for {user_1}, {user_2}: {score}")

        logger.debug(message_history)
        num_messages = 0
        for messages in message_history.values():
            num_messages += len(messages)
        logger.info(f"Message history has {num_messages} messages on {len(message_history.keys())} users")
        await asyncio.sleep(30)

def parse_commands(text: str):
    try:
        parts = shlex.split(text, posix=True)
    except:
        return None
    if len(parts) != 3 or parts[0] != "nickatcher":
        return None
    return parts[1], parts[2]