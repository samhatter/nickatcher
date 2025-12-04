import asyncio
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

from artifacts import Artifacts, get_artifacts

logger = logging.getLogger('nickatcher')


class ModelManager:
    def __init__(self, *, min_chunks: int):
        self._min_chunks = min_chunks
        self._artifacts: Optional[Artifacts] = None
        self._refresh_hours = int(os.getenv('ARTIFACT_REFRESH_HOURS', '24'))
        self._recompute_on_start = (
            os.getenv('RECOMPUTE_ARTIFACTS_ON_START', 'false').lower() == 'true'
        )
        self._refresh_task: Optional[asyncio.Task] = None
        self._artifacts_path = Path(os.getenv('ARTIFACTS_PATH', '/data/artifacts.pkl'))

    def _load_artifacts(self) -> bool:
        """Load cached artifacts from disk into self._artifacts. Returns True if successful."""
        if not self._artifacts_path.exists():
            return False

        try:
            with open(self._artifacts_path, 'rb') as f:
                self._artifacts = pickle.load(f)
            if self._artifacts is None:
                raise ValueError("Loaded artifacts is None")
            self._artifacts.last_updated = self._artifacts_path.stat().st_mtime
            return True
        except Exception as exc:
            logger.error("Failed to load cached artifacts: %s", exc)
            return False

    def _persist_artifacts(self) -> None:
        """Persist artifacts to disk."""
        if self._artifacts is None:
            return
        
        self._artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._artifacts_path, 'wb') as f:
            pickle.dump(self._artifacts, f)
        
        logger.info("Persisted artifacts to disk (%s)", self._artifacts_path)

    async def initialize(self) -> None:
        if self._load_artifacts() and self._artifacts is not None:
            age_hours = (time.time() - self._artifacts.last_updated) / 3600
            logger.info(
                "Loaded cached artifacts (age: %.2f hours)",
                age_hours,
            )

        if self._artifacts is None:
            logger.info("No artifacts loaded from disk, computing new artifacts")
            await self.refresh()
        
        elif self._recompute_on_start:
            logger.info("Recomputing artifacts on start as per configuration")
            _ = asyncio.create_task(self.refresh())
            
        if self._refresh_hours > 0:
            self._refresh_task = asyncio.create_task(self._refresh_loop())
        

    async def refresh(self) -> Artifacts:
        self._artifacts = await get_artifacts(min_chunks=self._min_chunks)
        self._persist_artifacts()
        return self._artifacts

    async def current(self) -> Artifacts:
        if self._artifacts is None:
            raise RuntimeError("Artifacts not initialized. Call initialize() first.")
        
        return self._artifacts

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_hours * 3600)
            try:
                await self.refresh()
                logger.info(
                    "Refreshed artifacts (scheduled refresh after %s hours)", self._refresh_hours
                )
            except Exception as exc:
                logger.error("Failed to refresh artifacts: %s", exc)
