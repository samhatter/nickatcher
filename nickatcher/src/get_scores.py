import asyncio
from client import SLSKDClient
from db.crud import list_messages
from get_embeddings import get_embeddings, group_messages
import logging
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger('nickatcher')

async def get_scores(slskd_client: SLSKDClient, session: AsyncSession, lda: LDA, dist: np.ndarray, room_name: str, max_tokens: int, min_chunks:int, user_1: str, user_2: str):
  user_messages_1 = list(await list_messages(session, user=user_1, limit=10000))
  user_messages_2 = list(await list_messages(session, user=user_2, limit=10000))
  if not await filter_user_messages(slskd_client=slskd_client, room_name=room_name, user_1=user_1, user_2=user_2, user_messages_1=user_messages_1, user_messages_2=user_messages_2):
    return
  
  group_messages_1 = group_messages(user_messages_1, max_tokens=max_tokens)
  group_messages_2 = group_messages(user_messages_2, max_tokens=max_tokens)
  user_embeddings_1, num_tokens_1 = get_embeddings(group_messages_1, max_tokens=max_tokens)
  user_embeddings_2, num_tokens_2 = get_embeddings(group_messages_2, max_tokens=max_tokens)
  if not await filter_user_tokens(slskd_client=slskd_client, room_name=room_name, user_1=user_1, user_2=user_2, num_tokens_1=num_tokens_1, num_tokens_2=num_tokens_2, min_chunks=min_chunks, max_tokens=max_tokens):
    return
  
  X = user_embeddings_1.detach().cpu().numpy()
  Y = user_embeddings_2.detach().cpu().numpy()
  
  X_transformed = lda.transform(X)
  Y_transformed = lda.transform(Y)

  X_mean = np.mean(X_transformed, axis=0)
  Y_mean = np.mean(Y_transformed, axis=0)
  
  score = cosine_similarity(X_mean.reshape(1, -1), Y_mean.reshape(1, -1))[0,0]
  percentile = (np.sum(dist < score) / len(dist)) * 100

  output_msg = f"Similarity for {user_1}, {user_2}: {score:.3f} ({percentile:.5} percentile). Computed from {num_tokens_1} and {num_tokens_2} tokens respectively. Ranges from (-1 dissimilar to 1 similar)."
  logger.debug(output_msg)
  await slskd_client.send_message(room_name=room_name, message=output_msg)


async def filter_user_messages(slskd_client: SLSKDClient, room_name: str, user_1: str, user_2: str, user_messages_1: list, user_messages_2: list):
  if user_messages_1 == [] and user_messages_2 == []:
    await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_1} or {user_2}, check your spelling!")
    return False
  elif user_messages_1 == []:
    await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_1}, check your spelling!")
    return False
  elif user_messages_2 == []:
    await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_2}, check your spelling!")
    return False
  return True

async def filter_user_tokens(slskd_client: SLSKDClient, room_name: str, user_1: str, user_2: str, num_tokens_1: int, num_tokens_2: int, min_chunks: int, max_tokens: int):
  token_threshold = min_chunks*max_tokens
  if num_tokens_1 < token_threshold and num_tokens_2 < token_threshold:
    await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_1} or {user_2}.")
  elif num_tokens_1 < token_threshold:
    await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_1}.")
  elif num_tokens_2 < token_threshold:
    await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_2}.")