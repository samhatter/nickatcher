from client import SLSKDClient
from db.crud import list_messages
from get_embeddings import get_embeddings
import logging
import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import normalize
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger('nickatcher')

async def get_scores(slskd_client: SLSKDClient, session: AsyncSession, room_name: str, max_tokens: int, min_chunks:int, user_1: str, user_2: str):
  user_messages_1 = list(await list_messages(session, user=user_1, limit=10000))
  user_messages_2 = list(await list_messages(session, user=user_2, limit=10000))
  if user_messages_1 == [] and user_messages_2 == []:
    await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_1} or {user_2}, check your spelling!")
  elif user_messages_1 == []:
    await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_1}, check your spelling!")
  elif user_messages_2 == []:
    await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_2}, check your spelling!")
  else:
      user_embeddings_1, num_tokens_1 = get_embeddings(messages=user_messages_1, max_tokens=max_tokens)
      user_embeddings_2, num_tokens_2 = get_embeddings(messages=user_messages_2, max_tokens=max_tokens)
      token_threshold = min_chunks*max_tokens
      if num_tokens_1 < token_threshold and num_tokens_2 < token_threshold:
        await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_1} or {user_2}.")
      elif num_tokens_1 < token_threshold:
        await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_1}.")
      elif num_tokens_2 < token_threshold:
        await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_2}.")
      else:
        X = user_embeddings_1.detach().cpu().numpy()
        Y = user_embeddings_2.detach().cpu().numpy()
        
        X_transformed, Y_transformed = await pca(X, Y, max_tokens, session=session)
        cX, cY = center(X_transformed,Y_transformed)
        nX, nY = norm(cX, cY)
        score = euclidean_distance(nX, nY)
        output_msg = f"Similarity for {user_1}, {user_2}: {str(score)[:5]}. Computed from {num_tokens_1} and {num_tokens_2} tokens respectively. Ranges from (0 uncorrelated to 1 correlated)."
        logger.debug(output_msg)
        await slskd_client.send_message(room_name=room_name, message=output_msg)

async def pca(X, Y, max_tokens, session, variance_threshold=0.95):
  all_messages = await list_messages(session=session, limit=10000)
  all_embeddings, _ = get_embeddings(list(all_messages), max_tokens=max_tokens)
  samples = all_embeddings.detach().cpu().numpy()
  pca = PCA(svd_solver='full').fit(samples)
  
  cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
  optimal_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
  logger.debug("Optimal PCA inner dim chosen: %i", optimal_components)

  X_transformed = pca.transform(X)[:,:optimal_components]
  Y_transformed = pca.transform(Y)[:,:optimal_components]
  logger.debug("After PCA:")
  vector_stats(X_transformed, Y_transformed)

  return X_transformed, Y_transformed

def vector_stats(X, Y):
  dot = np.dot(X, Y.T)[0, 0]
  norm_x = np.linalg.norm(X)
  norm_y = np.linalg.norm(Y)
  similarity = dot / (norm_x * norm_y)
  logger.debug("Dot: %.5f, NormX: %.5f, NormY: %.5f, CosSim: %.5f", dot, norm_x, norm_y, similarity)

def center(X, Y):
  cX = X.mean(axis=0, keepdims=True)
  cY = Y.mean(axis=0, keepdims=True)
  logger.debug("After centering:")
  vector_stats(cX, cY)
  return cX, cY

def norm(X, Y):
  nX = normalize(X, 'l2')
  nY = normalize(Y, 'l2')
  logger.debug("After normalizing:")
  vector_stats(nX, nY)
  return nX, nY

def euclidean_distance(X, Y):
  dist = np.linalg.norm(X - Y)
  return float(1/(1+dist))
