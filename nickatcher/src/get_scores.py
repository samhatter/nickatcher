from client import SLSKDClient
from db.crud import list_messages
from get_embeddings import get_embeddings
import logging
import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import StandardScaler
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
        
        X_transformed, Y_transformed = pca(X, Y)
        cX, cY = center(X_transformed,Y_transformed)
        score = euclidean_distance(cX, cY)
        output_msg = f"Similarity for {user_1}, {user_2}: {str(score)[:5]}. Computed from {num_tokens_1} and {num_tokens_2} tokens respectively. Ranges from (0 uncorrelated to 1 correlated)."
        logger.debug(output_msg)
        await slskd_client.send_message(room_name=room_name, message=output_msg)

def pca(X, Y, variance_threshold=0.95):
  samples = np.concatenate((X, Y), axis=0)
  scaler = StandardScaler()
  scalar_samples = scaler.fit_transform(samples)
  pca = PCA(svd_solver='full').fit(scalar_samples)
  
  cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
  optimal_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
  logger.debug("Optimal PCA inner dim chosen: %i", optimal_components)

  transformed_samples = PCA(n_components=optimal_components, svd_solver='full').fit_transform(scalar_samples)

  X_transformed = transformed_samples[:X.shape[0]]
  Y_transformed = transformed_samples[X.shape[0]:]

  return X_transformed, Y_transformed

def center(X, Y):
  cX = X.mean(axis=0, keepdims=True)
  cY = Y.mean(axis=0, keepdims=True)
  dot = np.dot(cX, cY.T)[0, 0]
  norm_x = np.linalg.norm(cX)
  norm_y = np.linalg.norm(cY)
  similarity = dot / (norm_x * norm_y)
  logger.debug("Dot: %.5f, NormX: %.5f, NormY: %.5f, CosSim: %.5f", dot, norm_x, norm_y, similarity)
  return cX, cY

def euclidean_distance(cX, cY):
  dist = np.linalg.norm(cX - cY)
  return float(1/(1+dist))

def cosine(cX, cY):
  similarity = cosine_similarity(cX, cY)[0, 0]
  return float(similarity)

def mmd_rbf(user_embeddings_1, user_embeddings_2, gamma=1.0):
    """
    Calculates the squared MMD using an RBF (Gaussian) kernel.
    
    Args:
      X: A numpy array representing the first sample (n_samples1, n_features).
      Y: A numpy array representing the second sample (n_samples2, n_features).
      gamma: The kernel coefficient for RBF kernel.
    
    Returns:
      The squared MMD value.
    """

    X = user_embeddings_1.detach().cpu().numpy()
    Y = user_embeddings_2.detach().cpu().numpy()

    k_xx = pairwise_kernels(X, X, metric='rbf', gamma=gamma)
    k_yy = pairwise_kernels(Y, Y, metric='rbf', gamma=gamma)
    k_xy = pairwise_kernels(X, Y, metric='rbf', gamma=gamma)
    
    return np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)