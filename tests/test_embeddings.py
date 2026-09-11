import numpy as np

from app.embeddings import (
    cosine_similarity,
    embed_text,
)


def test_embedding_is_numpy_array():

    embedding = embed_text(
        "abnormal cutter vibration"
    )

    assert isinstance(
        embedding,
        np.ndarray,
    )

    assert (
        embedding.ndim
        == 1
    )

    assert (
        len(embedding)
        > 0
    )


def test_similar_sentences_have_higher_similarity():

    query = embed_text(
        "cutter is shaking badly"
    )

    similar = embed_text(
        "abnormal vibration in cutter"
    )

    unrelated = embed_text(
        "monthly financial report"
    )

    similar_score = cosine_similarity(
        query,
        similar,
    )

    unrelated_score = cosine_similarity(
        query,
        unrelated,
    )

    assert (
        similar_score
        > unrelated_score
    )
    