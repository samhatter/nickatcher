import logging
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from client import SLSKDClient
from db.core import SessionLocal
from db.crud import list_messages
from embeddings import EMBEDDING_MAX_TOKENS, get_embeddings, group_messages
from model_manager import ModelManager

logger = logging.getLogger('nickatcher')

DEFAULT_NUM_RESPONSES = int(os.getenv('DEFAULT_NUM_RESPONSES', '5'))


def _compute_percentile(score: float, dist: np.ndarray) -> float:
  return (np.sum(dist < score) / len(dist)) * 100


async def get_scores(
    slskd_client: SLSKDClient,
    model_manager: ModelManager,
    room_name: str,
    min_chunks: int,
    user_1: str,
    user_2: str,
):
  artifacts = await model_manager.current()
  
  async with SessionLocal() as session:
    user_messages_1 = list(await list_messages(session, user=user_1, limit=10000))
    user_messages_2 = list(await list_messages(session, user=user_2, limit=10000))
  if not await filter_user_messages(slskd_client=slskd_client, room_name=room_name, user_1=user_1, user_2=user_2, user_messages_1=user_messages_1, user_messages_2=user_messages_2):
    return

  grouped_messages_1, num_tokens_1 = group_messages(user_messages_1)
  grouped_messages_2, num_tokens_2 = group_messages(user_messages_2)

  if not await filter_user_tokens(slskd_client=slskd_client, room_name=room_name, user_1=user_1, user_2=user_2, num_tokens_1=num_tokens_1, num_tokens_2=num_tokens_2, min_chunks=min_chunks):
    return
  
  user_embeddings_1 = await get_embeddings(grouped_messages_1)
  user_embeddings_2 = await get_embeddings(grouped_messages_2)

  X = user_embeddings_1.detach().cpu().numpy()
  Y = user_embeddings_2.detach().cpu().numpy()

  X_pca = artifacts.pca.transform(X)
  Y_pca = artifacts.pca.transform(Y)

  X_lda = artifacts.lda.transform(X_pca)[:, :artifacts.d_lda]
  Y_lda = artifacts.lda.transform(Y_pca)[:, :artifacts.d_lda]

  X_mean = np.mean(X_lda, axis=0)
  Y_mean = np.mean(Y_lda, axis=0)

  score = cosine_similarity(X_mean.reshape(1, -1), Y_mean.reshape(1, -1))[0,0]
  percentile = _compute_percentile(score, artifacts.dist)

  output_msg = f"Similarity for {user_1}, {user_2}: {score:.3f} ({percentile:.5} percentile). Computed from {num_tokens_1} and {num_tokens_2} tokens respectively. Ranges from (-1 dissimilar to 1 similar)."
  logger.info(output_msg)
  await slskd_client.send_message(room_name=room_name, message=output_msg)


async def get_similar_users(
    slskd_client: SLSKDClient,
    model_manager: ModelManager,
    room_name: str,
    target_user: str,
    num_responses: int | None = None,
):
  artifacts = await model_manager.current()

  if target_user not in artifacts.users:
    await slskd_client.send_message(
        room_name=room_name,
        message=(
            f"No embeddings found for {target_user}. "
            "Make sure they have enough messages to be processed."
        ),
    )
    return

  desired = num_responses or DEFAULT_NUM_RESPONSES
  desired = max(1, min(desired, len(artifacts.users) - 1))

  target_idx = artifacts.users.index(target_user)
  similarities = artifacts.sim_matrix[target_idx]
  neighbors = [
      (artifacts.users[i], similarities[i])
      for i in np.argsort(similarities)[::-1]
      if i != target_idx
  ][:desired]

  formatted = ", ".join(
      [
          (
              f"{i+1}. {name} "
              f"({score:.3f}, {_compute_percentile(score, artifacts.dist):.2f} percentile)"
          )
          for i, (name, score) in enumerate(neighbors)
      ]
  )
  await slskd_client.send_message(
      room_name=room_name,
      message=(
          f"Closest users to {target_user}: {formatted}" if formatted else
          f"No neighbors available for {target_user}."
      ),
  )

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

async def filter_user_tokens(slskd_client: SLSKDClient, room_name: str, user_1: str, user_2: str, num_tokens_1: int, num_tokens_2: int, min_chunks: int):
  token_threshold = min_chunks*EMBEDDING_MAX_TOKENS
  if num_tokens_1 < token_threshold and num_tokens_2 < token_threshold:
    await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_1} or {user_2}.")
    return False
  elif num_tokens_1 < token_threshold:
    await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_1}.")
    return False
  elif num_tokens_2 < token_threshold:
    await slskd_client.send_message(room_name=room_name, message=f"Not enough tokens found for {user_2}.")
    return False
  return True
