import asyncio
from db.core import SessionLocal
from db.crud import list_messages
from get_embeddings import get_embeddings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np

logger = logging.getLogger('nickatcher');

async def get_lda():
    async with SessionLocal() as session:
        messages = list(await list_messages(session=session, limit=1000000))
        logger.info(f"Retrieved {len(messages)} user messages from database")
        embeddings = []
        unique_users = list(set([message.user for message in messages]))
        num_users = len(unique_users)
        logger.info(f"Identified {num_users} unique users")
        for i, user in enumerate(unique_users):
            user_messages = [message.content for message in messages if message.user == user]
            user_embeddings, _ = get_embeddings(user_messages)
            list_embeddings = list(user_embeddings.detach().cpu().numpy())
            embeddings.append((user, list_embeddings))
            if i % (num_users // 100) == 0:
                percent_done = i / num_users * 100
                logger.info(f"Computing User Embeddings: {percent_done}%")
        logger.info("Finished Computing User Embeddings")

        X = np.vstack([e[1] for e in embeddings])
        y = [e[0] for e in embeddings for _ in e[1]]
        lda = LDA(n_components=12)
        X_transformed = lda.fit_transform(X, y)
        logger.info("Finished computing LDA fit")

        centroids = []
        for user in unique_users:
            user_indices = [i for i, label in enumerate(y) if label == user]
            user_vecs = X_transformed[user_indices]
            centroids.append(np.mean(user_vecs, axis=0))
        
        sim_matrix = cosine_similarity(np.vstack(centroids))
        logger.info("Finished Computing Similarity Matrix")
        
        dist = []
        for i in range(len(unique_users)):
            for j in range(i + 1, len(unique_users)):
                dist.append(sim_matrix[i, j])

        return lda, np.array(dist)
