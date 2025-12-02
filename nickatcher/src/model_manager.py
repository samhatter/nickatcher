import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from get_lda import LDAArtifacts, get_lda, _compute_distances

logger = logging.getLogger('nickatcher')


class ModelManager:
    def __init__(self, *, min_chunks: int):
        self._min_chunks = min_chunks
        self._artifacts: Optional[LDAArtifacts] = None
        self._refresh_hours = int(os.getenv('LDA_REFRESH_HOURS', '24'))
        self._recompute_on_start = (
            os.getenv('RECOMPUTE_LDA_ON_START', 'false').lower() == 'true'
        )
        self._refresh_task: Optional[asyncio.Task] = None
        self._sim_matrix_path = Path(os.getenv('SIM_MATRIX_PATH', '/data/similarity_matrix.npz'))
        self._lda_model_path = Path(os.getenv('LDA_MODEL_PATH', '/data/lda_model.joblib'))

    def _load_artifacts(self) -> bool:
        """Load cached artifacts from disk into self._artifacts. Returns True if successful."""
        if not self._sim_matrix_path.exists() or not self._lda_model_path.exists():
            return False

        try:
            lda = joblib.load(self._lda_model_path)
            saved = np.load(self._sim_matrix_path, allow_pickle=True)
            sim_matrix = saved['sim_matrix']
            users = [str(u) for u in saved['users'].tolist()]
            dist = _compute_distances(sim_matrix)
            last_updated = min(
                self._sim_matrix_path.stat().st_mtime,
                self._lda_model_path.stat().st_mtime,
            )
            self._artifacts = LDAArtifacts(
                lda=lda,
                dist=dist,
                sim_matrix=sim_matrix,
                users=users,
                last_updated=last_updated,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load cached LDA artifacts: %s", exc)
            return False

    async def initialize(self) -> LDAArtifacts:
        if not self._recompute_on_start:
            if self._load_artifacts() and self._artifacts is not None:
                age_hours = (time.time() - self._artifacts.last_updated) / 3600
                logger.info(
                    "Loaded cached LDA artifacts (age: %.2f hours)",
                    age_hours,
                )
        
        if self._artifacts is None:
            logger.info("Computing LDA artifacts from scratch")
            self._artifacts = await get_lda(min_chunks=self._min_chunks)
        
        if self._refresh_hours > 0:
            self._refresh_task = asyncio.create_task(self._refresh_loop())
        
        return self._artifacts

    async def refresh(self) -> LDAArtifacts:
        self._artifacts = await get_lda(min_chunks=self._min_chunks)
        return self._artifacts

    async def current(self) -> LDAArtifacts:
        if self._artifacts is None:
            raise RuntimeError("Artifacts not initialized. Call initialize() first.")
        
        return self._artifacts

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_hours * 3600)
            try:
                await self.refresh()
                logger.info(
                    "Refreshed LDA artifacts (scheduled refresh after %s hours)", self._refresh_hours
                )
            except Exception as exc:
                logger.error("Failed to refresh LDA artifacts: %s", exc)
