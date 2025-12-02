import logging

import aiohttp

logger = logging.getLogger('nickatcher')

class SLSKDClient:
    _url: str

    def __init__(self, url: str):
        self._url = url

    async def _get(self, endpoint):
        url = self._url + endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                return data
            
    async def _post(self, endpoint, data):
        url = self._url + endpoint
        async with aiohttp.ClientSession() as session:
            headers = {
                'Content-Type': 'application/json'
            }
            async with session.post(url, json=data, headers=headers) as response:
                try:
                    response.raise_for_status()
                    data = await response.text()
                    return data
                except Exception as e:
                    logger.warning("Failed to reach slskd client: %s", e)

    async def get_messages(self, room_name: str):
        return await self._get(endpoint=f"/api/v0/rooms/joined/{room_name}/messages")

    async def send_message(self, room_name: str, message: str):
        return await self._post(endpoint=f"/api/v0/rooms/joined/{room_name}/messages", data=message)