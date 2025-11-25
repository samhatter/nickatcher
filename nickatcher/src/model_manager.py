import asyncio
import logging
import os
from typing import Optional

from get_lda import LDAArtifacts, get_lda

logger = logging.getLogger('nickatcher')


class ModelManager:
    def __init__(self, *, min_chunks: int):
        self._min_chunks = min_chunks
        self._artifacts: Optional[LDAArtifacts] = None
        self._lock = asyncio.Lock()
        self._refresh_hours = int(os.getenv('LDA_REFRESH_HOURS', '24'))
        self._recompute_on_start = (
            os.getenv('RECOMPUTE_LDA_ON_START', 'false').lower() == 'true'
        )
        self._refresh_task: Optional[asyncio.Task] = None

    async def initialize(self) -> LDAArtifacts:
        artifacts = await self.refresh(force=self._recompute_on_start)
        if self._refresh_hours > 0:
            self._refresh_task = asyncio.create_task(self._refresh_loop())
        return artifacts

    async def refresh(self, *, force: bool = False) -> LDAArtifacts:
        async with self._lock:
            self._artifacts = await get_lda(
                min_chunks=self._min_chunks,
                force_recompute=force,
            )
            return self._artifacts

    async def current(self) -> LDAArtifacts:
        async with self._lock:
            if self._artifacts is None:
                self._artifacts = await get_lda(
                    min_chunks=self._min_chunks,
                    force_recompute=self._recompute_on_start,
                )
            return self._artifacts

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_hours * 3600)
            try:
                await self.refresh(force=True)
                logger.info(
                    "Refreshed LDA artifacts after %s hours", self._refresh_hours
                )
            except Exception as exc:
                logger.error("Error refreshing LDA artifacts: %s", exc)
