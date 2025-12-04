import asyncio
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from db.core import SessionLocal
from db.crud import list_messages
from get_embeddings import EMBEDDING_MAX_TOKENS, get_embeddings, group_messages

logger = logging.getLogger('nickatcher')


@dataclass
class Artifacts:
    pca: PCA
    lda: LDA
    d_lda: int
    dist: np.ndarray
    sim_matrix: np.ndarray
    users: list[str]
    last_updated: float

async def get_artifacts(min_chunks: int) -> Artifacts:
    """Compute artifacts."""

    async with SessionLocal() as session:
        messages = list(await list_messages(session=session, limit=1000000))

    logger.info("Retrieved %d messages from database", len(messages))
    unique_users = list(set([message.user for message in messages]))
    num_users = len(unique_users)
    logger.info("Identified %d unique users", num_users)

    token_threshold = min_chunks * EMBEDDING_MAX_TOKENS

    logger.info("Filtering users by token threshold (%d tokens)", token_threshold)
    user_messages_map, filtered_count, total_filtered_messages = filter_users_by_tokens(
        messages, unique_users, token_threshold
    )
    
    eligible_user_count = len(user_messages_map)
    logger.info(
        "Filtered %d/%d users (%d messages) below token threshold, processing %d eligible users",
        filtered_count,
        num_users,
        total_filtered_messages,
        eligible_user_count,
    )

    total_messages = sum(len(um[0]) for um in user_messages_map.values())
    completed_lock = asyncio.Lock()
    processed_counter = {'count': 0}
    stop_event = asyncio.Event()

    logger_task = asyncio.create_task(
        periodic_logger(stop_event, completed_lock, processed_counter, total_messages, logger)
    )

    embeddings = await asyncio.gather(
        *[
            asyncio.create_task(
                compute_user_embeddings(user, user_messages_map, completed_lock, processed_counter)
            )
            for user in user_messages_map.keys()
        ]
    )
    
    stop_event.set()
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

    # Apply PCA for dimensionality reduction, whitening, keeping 95% explained covariance
    pca = PCA(n_components=0.95, whiten=True)
    X_pca = pca.fit_transform(X)
    logger.info("Applied PCA for dimensionality reduction, reduced to %d dimensions", X_pca.shape[1])

    #Apply LDA on user labels 
    y = [e[0] for e in embeddings for _ in e[1]]
    lda = LDA(solver="eigen")
    lda.fit(X_pca, y)

    # Keep LDA components that explain 90% variance
    threshold = 0.90
    gamma = lda.explained_variance_ratio_
    cum_gamma = np.cumsum(gamma)
    d_lda = int(np.searchsorted(cum_gamma, threshold) + 1)
    d_lda = max(2, d_lda)                 # at least 2
    d_lda = min(d_lda, len(gamma))        # can't exceed # of LDA directions
    d_lda = min(d_lda, X_pca.shape[1])    # can't exceed PCA dim
    d_lda = min(d_lda, 32)                # sanity cap
    X_lda = lda.transform(X_pca)[:, :d_lda]
    print("Using LDA dimension:", d_lda)

    centroids = []
    for user in eligible_users:
        user_indices = [i for i, label in enumerate(y) if label == user]
        user_vecs = X_lda[user_indices]
        centroids.append(np.mean(user_vecs, axis=0))

    sim_matrix = cosine_similarity(np.vstack(centroids))
    logger.info("Computed similarity matrix")

    dist = []
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[0]):
            dist.append(sim_matrix[i, j])
    dist = np.array(dist)

    artifacts = Artifacts(
        pca=pca,
        lda=lda,
        d_lda=d_lda,
        dist=dist,
        sim_matrix=sim_matrix,
        users=eligible_users,
        last_updated=time.time(),
    )
    return artifacts


async def periodic_logger(
    stop_event: asyncio.Event,
    completed_lock: asyncio.Lock,
    processed_counter: dict,
    total_messages: int,
    logger,
) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(300)  # 5 minutes
        if stop_event.is_set():
            break
        async with completed_lock:
            processed = processed_counter.get('count', 0)
            percent_done = (processed / total_messages * 100) if total_messages else 0
            logger.info(
                "Encoded %.1f%% of messages (%d/%d)",
                percent_done,
                processed,
                total_messages,
            )


async def compute_user_embeddings(
    user: str,
    user_messages_map: dict,
    completed_lock: asyncio.Lock,
    processed_counter: dict,
):
    user_messages, grouped_messages = user_messages_map[user]
    num_user_messages = len(user_messages)
    user_embeddings = await get_embeddings(grouped_messages)
    list_embeddings = list(user_embeddings.detach().cpu().numpy())

    async with completed_lock:
        processed_counter['count'] = processed_counter.get('count', 0) + num_user_messages

    return user, list_embeddings
    


def filter_users_by_tokens(messages, unique_users, token_threshold):
    """Filter users by token threshold and return eligible user messages map.
    
    Returns:
        tuple: (user_messages_map, filtered_count, total_filtered_messages)
            where user_messages_map maps user -> (user_messages, grouped_messages)
    """
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
    
    return user_messages_map, len(filtered_users), total_filtered_messages