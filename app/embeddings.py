import numpy as np

from sentence_transformers import (
    SentenceTransformer,
)


MODEL_NAME = "all-MiniLM-L6-v2"


_model = None


def get_embedding_model():

    global _model

    if _model is None:

        _model = SentenceTransformer(
            MODEL_NAME
        )

    return _model


def embed_text(
    text: str,
) -> np.ndarray:

    model = get_embedding_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
    )