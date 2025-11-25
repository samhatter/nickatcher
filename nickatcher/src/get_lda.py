import asyncio
import logging
import os
from pathlib import Path
import joblib

from db.core import SessionLocal
from db.crud import list_messages
from get_embeddings import EMBEDDING_MAX_TOKENS, get_embeddings, group_messages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger('nickatcher')


SIM_MATRIX_PATH = Path(os.getenv("SIM_MATRIX_PATH", "data/similarity_matrix.npz"))
LDA_MODEL_PATH = Path(os.getenv("LDA_MODEL_PATH", "data/lda_model.joblib"))


def _persist_similarity_data(sim_matrix: np.ndarray, users: list[str], dist: np.ndarray):
    SIM_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        SIM_MATRIX_PATH,
        sim_matrix=sim_matrix,
        users=np.array(users),
        dist=dist,
    )
    logger.info("Stored similarity matrix to %s", SIM_MATRIX_PATH)


def _persist_lda_model(lda: LDA):
    LDA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lda, LDA_MODEL_PATH)
    logger.info("Stored LDA model to %s", LDA_MODEL_PATH)


def load_persisted_lda():
    if not LDA_MODEL_PATH.exists() or not SIM_MATRIX_PATH.exists():
        return None

    try:
        lda = joblib.load(LDA_MODEL_PATH)
        sim_data = np.load(SIM_MATRIX_PATH, allow_pickle=True)
        sim_matrix = sim_data["sim_matrix"]
        users = list(sim_data["users"])
        dist = sim_data["dist"]
    except Exception:
        logger.exception("Failed to load persisted LDA state")
        return None

    logger.info("Loaded persisted LDA state from disk")
    return lda, dist, sim_matrix, users


async def get_lda(min_chunks: int):
    async with SessionLocal() as session:
        messages = list(await list_messages(session=session, limit=1000000))
        logger.info(f"Retrieved {len(messages)} user messages from database")
        unique_users = list(set([message.user for message in messages]))
        num_users = len(unique_users)
        logger.info(f"Identified {num_users} unique users")

        token_threshold = min_chunks * EMBEDDING_MAX_TOKENS

        log_interval = max(1, num_users // 20)
        completed_users = 0
        completed_lock = asyncio.Lock()

        async def compute_user_embeddings(index: int, user: str):
            nonlocal completed_users
            user_messages = [message for message in messages if message.user == user]
            grouped_messages = group_messages(user_messages)
            user_embeddings, num_tokens = await get_embeddings(grouped_messages)
            list_embeddings = list(user_embeddings.detach().cpu().numpy())

            async with completed_lock:
                completed_users += 1
                should_log = (
                    completed_users == num_users
                    or completed_users % log_interval == 0
                )

                if should_log:
                    percent_done = (completed_users / num_users * 100) if num_users else 100
                    logger.info(
                        f"Computed embeddings for {completed_users}/{num_users} users "
                        f"({percent_done:.2f}% complete)"
                    )
            if num_tokens < token_threshold or not list_embeddings:
                logger.debug(
                    f"Skipping user {user}: only {num_tokens} tokens (< {token_threshold})"
                )
                return None

            return user, list_embeddings

        embeddings = await asyncio.gather(
            *[
                asyncio.create_task(compute_user_embeddings(i, user))
                for i, user in enumerate(unique_users)
            ]
        )
        embeddings = [e for e in embeddings if e is not None]
        eligible_users = [e[0] for e in embeddings]
        logger.info(
            "Finished Computing User Embeddings. "
            f"Eligible users: {len(eligible_users)}/{num_users}"
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
        logger.info("Finished computing LDA fit")

        centroids = []
        for user in eligible_users:
            user_indices = [i for i, label in enumerate(y) if label == user]
            user_vecs = X_transformed[user_indices]
            centroids.append(np.mean(user_vecs, axis=0))

        sim_matrix = cosine_similarity(np.vstack(centroids))
        logger.info("Finished Computing Similarity Matrix")

        dist = []
        for i in range(len(eligible_users)):
            for j in range(i + 1, len(eligible_users)):
                dist.append(sim_matrix[i, j])

        dist_array = np.array(dist)
        _persist_similarity_data(sim_matrix=sim_matrix, users=eligible_users, dist=dist_array)
        _persist_lda_model(lda=lda)

        return lda, dist_array, sim_matrix, eligible_users
