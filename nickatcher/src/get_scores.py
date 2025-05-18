from client import SLSKDClient
from db.crud import list_messages
from get_embeddings import get_embeddings
import logging
import numpy as np
from sklearn.metrics import pairwise_kernels
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger('nickatcher')

async def get_scores(slskd_client: SLSKDClient, session: AsyncSession, room_name: str, user_1: str, user_2: str):
    user_messages_1 = list(await list_messages(session, user=user_1, limit=10000))
    user_messages_2 = list(await list_messages(session, user=user_2, limit=10000))
    if user_messages_1 == [] and user_messages_2 == []:
        await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_1} or {user_2}, check your spelling!")
    elif user_messages_1 == []:
        await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_1}, check your spelling!")
    elif user_messages_2 == []:
        await slskd_client.send_message(room_name=room_name, message=f"No messages found for {user_2}, check your spelling!")
    else:
        user_embeddings_1, num_tokens_1 = get_embeddings(user_messages_1)
        user_embeddings_2, num_tokens_2 = get_embeddings(user_messages_2)
        X = user_embeddings_1.detach().cpu().numpy()
        Y = user_embeddings_2.detach().cpu().numpy()
        score = mmd_rbf(X, Y)
        output_msg = f"Embedding distance for score for {user_1}, {user_2}: {score}. Computed from {num_tokens_1} and {num_tokens_2} tokens respectively."
        logger.debug(output_msg)
        await slskd_client.send_message(room_name=room_name, message=output_msg)

def mmd_rbf(X, Y, gamma=1.0):
    """
    Calculates the squared MMD using an RBF (Gaussian) kernel.
    
    Args:
      X: A numpy array representing the first sample (n_samples1, n_features).
      Y: A numpy array representing the second sample (n_samples2, n_features).
      gamma: The kernel coefficient for RBF kernel.
    
    Returns:
      The squared MMD value.
    """

    k_xx = pairwise_kernels(X, X, metric='rbf', gamma=gamma)
    k_yy = pairwise_kernels(Y, Y, metric='rbf', gamma=gamma)
    k_xy = pairwise_kernels(X, Y, metric='rbf', gamma=gamma)
    
    return np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)