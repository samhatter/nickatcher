import asyncio
import logging
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity

from db.core import SessionLocal
from db.crud import list_messages
from get_embeddings import EMBEDDING_MAX_TOKENS, get_embeddings, group_messages

logger = logging.getLogger('nickatcher')


async def get_lda(min_chunks: int):
    async with SessionLocal() as session:
        messages = list(await list_messages(session=session, limit=1000000))
    
    logger.info(f"Retrieved {len(messages)} user messages from database")
    unique_users = list(set([message.user for message in messages]))
    num_users = len(unique_users)
    logger.info(f"Identified {num_users} unique users")

    token_threshold = min_chunks * EMBEDDING_MAX_TOKENS

    logger.info("Pre-filtering users by token count...")
    user_messages_map = {}
    filtered_users = []
    total_filtered_messages = 0
    
    for user in unique_users:
        user_messages = [message for message in messages if message.user == user]
        grouped_messages, token_count = group_messages(user_messages)
        
        if token_count >= token_threshold:
            user_messages_map[user] = (user_messages, grouped_messages)
        else:
            filtered_users.append(user)
            total_filtered_messages += len(user_messages)
    
    eligible_user_count = len(user_messages_map)
    filtered_count = len(filtered_users)
    logger.info(
        f"Pre-filtered {filtered_count}/{num_users} users "
        f"({total_filtered_messages} messages) below token threshold. "
        f"Processing {eligible_user_count} eligible users."
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
                    f"Encoded {percent_done:.2f}% of messages "
                    f"({processed_messages}/{total_messages})"
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

    return lda, np.array(dist)
