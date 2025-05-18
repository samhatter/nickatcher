import asyncio
import logging

logger = logging.getLogger('nickatcher')

async def clean_message_history(message_history: dict, history_weeks):
    pass
    await asyncio.sleep(604800*history_weeks)