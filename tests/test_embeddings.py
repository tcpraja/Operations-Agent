import hashlib
import math
import re


EMBEDDING_DIMENSION = 512


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _hash_token(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % EMBEDDING_DIMENSION


def embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION

    tokens = _tokenize(text)

    for token in tokens:
        index = _hash_token(token)
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))

    if norm > 0:
        vector = [value / norm for value in vector]

    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_text(text) for text in texts]


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    return sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )