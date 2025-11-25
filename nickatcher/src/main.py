import asyncio
from client import SLSKDClient
import datetime as dt
from db.init_db import init_db
from get_lda import get_lda, load_persisted_lda
from ingest_messages import ingest_messages
from lda_state import LDAState
import logging
import os
import sys
import numpy as np


async def refresh_lda_periodically(lda_state: LDAState, min_chunks: int, interval_hours: int):
    while True:
        lda, dist, sim_matrix, users = await get_lda(min_chunks=min_chunks)
        await lda_state.update(lda=lda, dist=dist, sim_matrix=sim_matrix, users=users)
        logger.info(
            "Refreshed LDA and similarity matrix. Median Similarity: %.2f",
            np.median(dist),
        )
        await asyncio.sleep(interval_hours * 3600)


async def main(url: str, room_name: str, min_chunks: int):
    slskd_client = SLSKDClient(url=url)
    await init_db()
    logger.info('Initialized Database')
    lda_state = LDAState()
    recompute_on_start = os.getenv("RECOMPUTE_LDA_ON_START", "false").lower() == "true"

    if not recompute_on_start:
        persisted = load_persisted_lda()
        if persisted:
            lda, dist, sim_matrix, users = persisted
            await lda_state.update(lda=lda, dist=dist, sim_matrix=sim_matrix, users=users)
            logger.info(
                "Loaded persisted LDA state. Median Similarity: %.2f",
                np.median(dist),
            )
        else:
            logger.info("No valid persisted LDA state found. Recomputing on startup.")
            lda, dist, sim_matrix, users = await get_lda(min_chunks=min_chunks)
            logger.info(f'Initialized LDA. Median Similarity: {np.median(dist):.2f}')
            await lda_state.update(lda=lda, dist=dist, sim_matrix=sim_matrix, users=users)
    else:
        logger.info("RECOMPUTE_LDA_ON_START enabled. Recomputing LDA on startup.")
        lda, dist, sim_matrix, users = await get_lda(min_chunks=min_chunks)
        logger.info(f'Initialized LDA. Median Similarity: {np.median(dist):.2f}')
        await lda_state.update(lda=lda, dist=dist, sim_matrix=sim_matrix, users=users)

    lda_refresh_hours = int(os.getenv('LDA_REFRESH_HOURS', '24'))

    await asyncio.gather(
        refresh_lda_periodically(
            lda_state=lda_state,
            min_chunks=min_chunks,
            interval_hours=lda_refresh_hours,
        ),
        ingest_messages(
            slskd_client=slskd_client,
            lda_state=lda_state,
            room_name=room_name,
            min_chunks=min_chunks,
        ),
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

