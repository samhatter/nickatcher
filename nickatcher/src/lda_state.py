import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


@dataclass
class LDAState:
    lda: Optional[LDA] = None
    dist: Optional[np.ndarray] = None
    sim_matrix: Optional[np.ndarray] = None
    users: List[str] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def update(
        self,
        *,
        lda: LDA,
        dist: np.ndarray,
        sim_matrix: np.ndarray,
        users: list[str],
    ) -> None:
        async with self._lock:
            self.lda = lda
            self.dist = dist
            self.sim_matrix = sim_matrix
            self.users = users

    async def snapshot(self) -> Tuple[Optional[LDA], Optional[np.ndarray], Optional[np.ndarray], list[str]]:
        async with self._lock:
            return self.lda, self.dist, self.sim_matrix, list(self.users)
