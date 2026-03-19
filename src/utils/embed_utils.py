"""Shared embedding utilities for memory systems."""

from typing import Iterable, List, Union

import numpy as np


def embed_texts(
    embed_client,
    texts: Union[List[str], Iterable[str]],
    model_name: str,
) -> np.ndarray:
    """
    Embed a batch of texts using the given OpenAI-compatible embed client.

    :param embed_client: OpenAI-compatible client with embeddings.create()
    :param texts: Texts to embed (list or iterable)
    :param model_name: Embedding model name
    :return: np.ndarray of shape (n, dim) with embeddings in order
    """
    if embed_client is None:
        raise ValueError("embed_client is required for embedding.")
    batch_inputs = list(texts)
    if not batch_inputs:
        return np.empty((0, 0), dtype=np.float32)

    response = embed_client.embeddings.create(
        input=batch_inputs,
        model=model_name,
    )
    indexed_embeddings = {
        item.index: np.asarray(item.embedding, dtype=np.float32)
        for item in response.data
    }
    ordered_embeddings = [
        indexed_embeddings[i] for i in range(len(batch_inputs))
    ]
    return np.vstack(ordered_embeddings).astype(np.float32)
