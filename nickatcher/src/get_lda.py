import asyncio
from db.core import SessionLocal
from db.crud import list_messages
from get_embeddings import get_embeddings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

async def get_lda(min_chunks: int, max_tokens: int):
    async with SessionLocal() as session:
        messages = list(await list_messages(session=session, limit=100000))
        embeddings = []
        users = []
        for user in set([message.user for message in messages]):
            user_messages = [message.content for message in messages if message.user == user]
            user_embeddings, _ = await asyncio.to_thread(get_embeddings, user_messages, max_tokens=max_tokens)
            list_embeddings = list(user_embeddings.detach().cpu().numpy())
            embeddings += list_embeddings
            users += [user for i in range(len(list_embeddings))]
        
        X = np.vstack(embeddings)
        y = users
        lda = LDA(n_components=12)
        X_transformed = lda.fit_transform(X, y)
        
        zipped = [(users[i], X_transformed[i]) for i in range(len(users))]
        centroids = []
        unique_users = list(set(users))
        for user in unique_users:
            centroids.append(np.mean(np.vstack([zip[1] for zip in zipped if zip[0] == user])))
        
        sim_matrix = cosine_similarity(np.vstack(centroids))
        
        dist = []
        for i in range(len(unique_users)):
            for j in range(i + 1, len(unique_users)):
                dist.append(sim_matrix[i, j])
                
        return lda, np.array(dist)