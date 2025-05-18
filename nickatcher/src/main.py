import asyncio
from client import SLSKDClient
import datetime as dt
from db.core import SessionLocal
from db.init_db import init_db
from ingest_messages import ingest_messages
import logging 
import os
import sys


async def main(url: str, room_name: str, min_tokens: int):
    slskd_client = SLSKDClient(url=url)
    await init_db()
    logger.info('Initialized Database')
    await ingest_messages(
            slskd_client=slskd_client,
            room_name=room_name,
            min_tokens=min_tokens,
    )
        

if __name__ == '__main__':
    slskd_http_port = os.getenv('SLSKD_HTTP_PORT', '8000')
    url = f'http://nickatcher-gluetun:{slskd_http_port}'
    room_name = os.getenv('SLSKD_ROOMS', '')
    logging_level = os.getenv('LOG_LEVEL', 'INFO')
    min_tokens = int(os.getenv('MIN_TOKENS', '12'))

    logger = logging.getLogger('nickatcher')
    logger.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging_level)
    logger.addHandler(sh)
    
    log_filename = f'/logs/nickatcher-{dt.datetime.now().strftime("%y%m%d%H%M")}.log'
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(formatter)
    fh.setLevel(logging_level)
    logger.addHandler(fh)

    asyncio.run(
        main(
            url=url,
            room_name=room_name,
            min_tokens=min_tokens,
        )
    )

