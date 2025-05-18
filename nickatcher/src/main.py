import asyncio
from datetime import datetime
import logging
from clean_message_history import clean_message_history
from client import SLSKDClient
from ingest_messages import ingest_messages
import os
import sys

def main(url: str, room_name: str, history_window: int, history_weeks:int):
    slskd_client = SLSKDClient(url=url)
    message_history = {}
    loop = asyncio.new_event_loop()
    loop.create_task(
        ingest_messages(
            slskd_client=slskd_client,
            room_name=room_name,
            message_history=message_history,
            history_window=history_window
        )
    )
    loop.create_task(
        clean_message_history(
            message_history=message_history,
            history_weeks=history_weeks,
        )
    )
    loop.run_forever()
        

if __name__ == '__main__':
    slskd_http_port = os.getenv('SLSKD_HTTP_PORT', '8000')
    url = f'http://nickatcher-gluetun:{slskd_http_port}'
    room_name = os.getenv('SLSKD_ROOMS', '')
    logging_level = os.getenv('LOG_LEVEL', 'INFO')
    history_window = int(os.getenv('HISTORY_WINDOW', '1000'))
    history_weeks = int(os.getenv('HISTORY_WEEKS', '10'))


    logger = logging.getLogger('nickatcher')
    logger.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging_level)
    logger.addHandler(sh)
    
    log_filename = f'/logs/nickatcher-{datetime.now().strftime("%d-%m-%y-%H-%M")}.log'
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(formatter)
    fh.setLevel(logging_level)
    logger.addHandler(fh)

    logger.info('Started')
    main(
        url=url,
        room_name=room_name,
        history_window=history_window,
        history_weeks=history_weeks
    )
