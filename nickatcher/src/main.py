import asyncio
import datetime as dt
import logging
import os
import sys

import numpy as np

from client import SLSKDClient
from db.init_db import init_db
from ingest_messages import ingest_messages
from model_manager import ModelManager


async def main(url: str, room_name: str, min_chunks: int):
    slskd_client = SLSKDClient(url=url)
    await init_db()
    logger.info("Initialized database")
    model_manager = ModelManager(min_chunks=min_chunks)
    await model_manager.initialize()
    await ingest_messages(
            slskd_client=slskd_client,
            model_manager=model_manager,
            room_name=room_name,
            min_chunks=min_chunks,
    )
        

if __name__ == '__main__':
    slskd_http_port = os.getenv('SLSKD_HTTP_PORT', '8000')
    url = f'http://nickatcher-gluetun:{slskd_http_port}'
    room_name = os.getenv('SLSKD_ROOMS', '')
    logging_level = os.getenv('LOG_LEVEL', 'INFO')
    min_chunks = int(os.getenv('MIN_CHUNKS', '10'))

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
            min_chunks=min_chunks,
        )
    )

