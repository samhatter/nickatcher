import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity

from db.core import SessionLocal
from db.crud import list_messages
from get_embeddings import EMBEDDING_MAX_TOKENS, get_embeddings, group_messages

logger = logging.getLogger('nickatcher')

SIM_MATRIX_PATH = Path(os.getenv('SIM_MATRIX_PATH', '/data/similarity_matrix.npz'))
LDA_MODEL_PATH = Path(os.getenv('LDA_MODEL_PATH', '/data/lda_model.joblib'))
LDA_REFRESH_HOURS = int(os.getenv('LDA_REFRESH_HOURS', '24'))
RECOMPUTE_LDA_ON_START = os.getenv('RECOMPUTE_LDA_ON_START', 'false').lower() == 'true'


@dataclass
class LDAArtifacts:
    lda: LDA
    dist: np.ndarray
    sim_matrix: np.ndarray
    users: list[str]
    last_updated: float


async def get_lda(min_chunks: int) -> LDAArtifacts:
    """Compute LDA artifacts from scratch."""
    logger.info("Computing LDA artifacts from scratch")

    async with SessionLocal() as session:
        messages = list(await list_messages(session=session, limit=1000000))

    logger.info("Retrieved %d messages from database", len(messages))
    unique_users = list(set([message.user for message in messages]))
    num_users = len(unique_users)
    logger.info("Identified %d unique users", num_users)

    token_threshold = min_chunks * EMBEDDING_MAX_TOKENS

    logger.info("Filtering users by token threshold (%d tokens)", token_threshold)

    messages_by_user = {}
    for message in messages:
        if message.user not in messages_by_user:
            messages_by_user[message.user] = []
        messages_by_user[message.user].append(message)
    
    user_messages_map = {}
    filtered_users = []
    total_filtered_messages = 0
    
    for user in unique_users:
        user_messages = messages_by_user.get(user, [])
        grouped_messages, token_count = group_messages(user_messages)
        
        if token_count >= token_threshold:
            user_messages_map[user] = (user_messages, grouped_messages)
        else:
            filtered_users.append(user)
            total_filtered_messages += len(user_messages)
    
    eligible_user_count = len(user_messages_map)
    filtered_count = len(filtered_users)
    logger.info(
        "Filtered %d/%d users (%d messages) below token threshold, processing %d eligible users",
        filtered_count,
        num_users,
        total_filtered_messages,
        eligible_user_count,
    )

    total_messages = sum(len(um[0]) for um in user_messages_map.values())
    processed_messages = 0
    completed_lock = asyncio.Lock()
    stop_logging = False

    async def periodic_logger():
        """Log progress every 5 minutes regardless of task completion."""
        while not stop_logging:
            await asyncio.sleep(300)  # 5 minutes
            if stop_logging:
                break
            async with completed_lock:
                percent_done = (processed_messages / total_messages * 100) if total_messages else 0
                logger.info(
                    "Encoded %.1f%% of messages (%d/%d)",
                    percent_done,
                    processed_messages,
                    total_messages,
                )

    async def compute_user_embeddings(index: int, user: str):
        nonlocal processed_messages
        user_messages, grouped_messages = user_messages_map[user]
        num_user_messages = len(user_messages)
        user_embeddings = await get_embeddings(grouped_messages)
        list_embeddings = list(user_embeddings.detach().cpu().numpy())

        async with completed_lock:
            processed_messages += num_user_messages

        return user, list_embeddings

    logger_task = asyncio.create_task(periodic_logger())

    embeddings = await asyncio.gather(
        *[
            asyncio.create_task(compute_user_embeddings(i, user))
            for i, user in enumerate(user_messages_map.keys())
        ]
    )
    
    stop_logging = True
    await logger_task
    
    embeddings = [e for e in embeddings if e is not None]
    eligible_users = [e[0] for e in embeddings]
    logger.info(
        "Computed embeddings for %d/%d users",
        len(eligible_users),
        num_users,
    )

    if len(eligible_users) < 2:
        raise ValueError(
            "Not enough eligible users to compute LDA. "
            f"Found {len(eligible_users)} users meeting the token threshold of {token_threshold}."
        )

    X = np.vstack([e[1] for e in embeddings])
    y = [e[0] for e in embeddings for _ in e[1]]
    max_components = min(12, len(eligible_users) - 1)
    lda = LDA(n_components=max_components)
    X_transformed = lda.fit_transform(X, y)
    logger.info("Computed LDA transformation (%d components)", max_components)

    centroids = []
    for user in eligible_users:
        user_indices = [i for i, label in enumerate(y) if label == user]
        user_vecs = X_transformed[user_indices]
        centroids.append(np.mean(user_vecs, axis=0))

    sim_matrix = cosine_similarity(np.vstack(centroids))
    logger.info("Computed similarity matrix")

    dist = _compute_distances(sim_matrix)

    artifacts = LDAArtifacts(
        lda=lda,
        dist=dist,
        sim_matrix=sim_matrix,
        users=eligible_users,
        last_updated=time.time(),
    )
    _persist_artifacts(artifacts)
    return artifacts


def _compute_distances(sim_matrix: np.ndarray) -> np.ndarray:
    dist = []
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[0]):
            dist.append(sim_matrix[i, j])
    return np.array(dist)


def _persist_artifacts(artifacts: LDAArtifacts) -> None:
    SIM_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    LDA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        SIM_MATRIX_PATH,
        sim_matrix=artifacts.sim_matrix,
        users=np.array(artifacts.users, dtype=object),
    )
    joblib.dump(artifacts.lda, LDA_MODEL_PATH)
    logger.info(
        "Persisted LDA artifacts to disk (model: %s, matrix: %s)",
        LDA_MODEL_PATH,
        SIM_MATRIX_PATH,
    )


def _load_artifacts() -> Optional[LDAArtifacts]:
    if not SIM_MATRIX_PATH.exists() or not LDA_MODEL_PATH.exists():
        return None

    try:
        lda = joblib.load(LDA_MODEL_PATH)
        saved = np.load(SIM_MATRIX_PATH, allow_pickle=True)
        sim_matrix = saved['sim_matrix']
        users = [str(u) for u in saved['users'].tolist()]
        dist = _compute_distances(sim_matrix)
        last_updated = min(
            SIM_MATRIX_PATH.stat().st_mtime,
            LDA_MODEL_PATH.stat().st_mtime,
        )
        return LDAArtifacts(
            lda=lda,
            dist=dist,
            sim_matrix=sim_matrix,
            users=users,
            last_updated=last_updated,
        )
    except Exception as exc:
        logger.error("Failed to load cached LDA artifacts: %s", exc)
        return None
