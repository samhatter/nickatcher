import asyncio
from client import SLSKDClient
import datetime as dt
from db.init_db import init_db
from get_lda import get_lda
from ingest_messages import ingest_messages
import logging 
import os
import sys
import numpy as np


async def main(url: str, room_name: str, min_chunks: int):
    slskd_client = SLSKDClient(url=url)
    await init_db()
    logger.info('Initialized Database')
    lda, dist = await get_lda(min_chunks=min_chunks)
    logger.info(f'Initialized LDA. Median Similarity: {np.median(dist):.2f}')
    await ingest_messages(
            slskd_client=slskd_client,
            lda=lda,
            dist=dist,
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

