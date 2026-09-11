import hashlib
import pickle
from pathlib import Path

from app.embeddings import (
    cosine_similarity,
    embed_text,
)


KNOWLEDGE_DIR = Path("knowledge")
INDEX_PATH = Path("data") / "vector_index.pkl"


# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_knowledge_documents() -> list[dict]:

    documents = []

    if not KNOWLEDGE_DIR.exists():
        return documents

    for file_path in sorted(
        KNOWLEDGE_DIR.glob("*.txt")
    ):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "source": file_path.name,
                "content": text,
            }
        )

    return documents


# =========================================================
# CHUNK TEXT
# =========================================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    chunks = []

    start = 0

    while start < len(cleaned_text):

        end = start + chunk_size

        chunk = cleaned_text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned_text):
            break

        start = end - overlap

    return chunks


# =========================================================
# CREATE KNOWLEDGE CHUNKS
# =========================================================

def create_knowledge_chunks(
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:

    documents = (
        load_knowledge_documents()
    )

    chunks = []

    for document in documents:

        document_chunks = (
            chunk_text(
                text=document["content"],
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

        for chunk_index, chunk in enumerate(
            document_chunks
        ):

            chunks.append(
                {
                    "source": (
                        document["source"]
                    ),
                    "chunk_id": (
                        f"{document['source']}"
                        f"::chunk-{chunk_index}"
                    ),
                    "chunk_index": (
                        chunk_index
                    ),
                    "content": (
                        chunk
                    ),
                }
            )

    return chunks


# =========================================================
# KNOWLEDGE FINGERPRINT
# =========================================================

def calculate_knowledge_fingerprint() -> str:

    hasher = hashlib.sha256()

    documents = (
        load_knowledge_documents()
    )

    for document in sorted(
        documents,
        key=lambda item: item["source"],
    ):

        hasher.update(
            document["source"].encode(
                "utf-8"
            )
        )

        hasher.update(
            document["content"].encode(
                "utf-8"
            )
        )

    return hasher.hexdigest()


# =========================================================
# IN-MEMORY CACHE
# =========================================================

_embedding_cache = None
_embedding_fingerprint = None


def clear_embedding_cache():

    global _embedding_cache
    global _embedding_fingerprint

    _embedding_cache = None
    _embedding_fingerprint = None


# =========================================================
# KNOWLEDGE CHANGE DETECTION
# =========================================================

def knowledge_has_changed() -> bool:

    current_fingerprint = (
        calculate_knowledge_fingerprint()
    )

    if _embedding_fingerprint is None:
        return True

    return (
        current_fingerprint
        != _embedding_fingerprint
    )


# =========================================================
# BUILD EMBEDDING CACHE
# =========================================================

def build_embedding_cache() -> list[dict]:

    global _embedding_cache
    global _embedding_fingerprint

    chunks = (
        create_knowledge_chunks()
    )

    cached_chunks = []

    for chunk in chunks:

        embedding = embed_text(
            chunk["content"]
        )

        cached_chunks.append(
            {
                **chunk,
                "embedding": (
                    embedding
                ),
            }
        )

    _embedding_cache = (
        cached_chunks
    )

    _embedding_fingerprint = (
        calculate_knowledge_fingerprint()
    )

    return _embedding_cache


# =========================================================
# SAVE EMBEDDING INDEX
# =========================================================

def save_embedding_index() -> None:

    global _embedding_cache
    global _embedding_fingerprint

    if _embedding_cache is None:
        build_embedding_cache()

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "fingerprint": (
            _embedding_fingerprint
        ),
        "chunks": (
            _embedding_cache
        ),
    }

    with INDEX_PATH.open(
        "wb"
    ) as file:

        pickle.dump(
            payload,
            file,
        )


# =========================================================
# LOAD EMBEDDING INDEX
# =========================================================

def load_embedding_index() -> bool:

    global _embedding_cache
    global _embedding_fingerprint

    if not INDEX_PATH.exists():
        return False

    try:

        with INDEX_PATH.open(
            "rb"
        ) as file:

            payload = pickle.load(
                file
            )

        fingerprint = payload.get(
            "fingerprint"
        )

        chunks = payload.get(
            "chunks"
        )

        if fingerprint is None:
            return False

        if chunks is None:
            return False

        if not isinstance(
            chunks,
            list,
        ):
            return False

        _embedding_cache = (
            chunks
        )

        _embedding_fingerprint = (
            fingerprint
        )

        return True

    except Exception:

        clear_embedding_cache()

        return False


# =========================================================
# BUILD OR LOAD INDEX
# =========================================================

def build_or_load_index() -> list[dict]:

    global _embedding_cache
    global _embedding_fingerprint

    current_fingerprint = (
        calculate_knowledge_fingerprint()
    )

    # -----------------------------------------------------
    # 1. Reuse valid in-memory cache
    # -----------------------------------------------------

    if (
        _embedding_cache is not None
        and
        _embedding_fingerprint
        == current_fingerprint
    ):
        return _embedding_cache

    # -----------------------------------------------------
    # 2. Try persistent disk index
    # -----------------------------------------------------

    loaded = (
        load_embedding_index()
    )

    if (
        loaded
        and
        _embedding_fingerprint
        == current_fingerprint
    ):
        return _embedding_cache

    # -----------------------------------------------------
    # 3. Knowledge changed or index is missing/stale
    # -----------------------------------------------------

    build_embedding_cache()

    # -----------------------------------------------------
    # 4. Save refreshed index
    # -----------------------------------------------------

    save_embedding_index()

    return _embedding_cache


# =========================================================
# CACHE ACCESS
# =========================================================

def get_embedding_cache() -> list[dict]:

    return build_or_load_index()


# =========================================================
# SEMANTIC RETRIEVAL
# =========================================================

def retrieve_knowledge(
    query: str,
    max_results: int = 3,
    minimum_score: float = 0.20,
) -> list[dict]:

    if max_results <= 0:
        return []

    cleaned_query = query.strip()

    if not cleaned_query:
        return []

    cached_chunks = (
        build_or_load_index()
    )

    if not cached_chunks:
        return []

    query_embedding = embed_text(
        cleaned_query
    )

    scored_chunks = []

    for chunk in cached_chunks:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"],
        )

        if score >= minimum_score:

            scored_chunks.append(
                (
                    score,
                    chunk,
                )
            )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        {
            "source": (
                chunk["source"]
            ),
            "chunk_id": (
                chunk["chunk_id"]
            ),
            "chunk_index": (
                chunk["chunk_index"]
            ),
            "content": (
                chunk["content"]
            ),
            "score": round(
                score,
                4,
            ),
        }
        for score, chunk
        in scored_chunks[:max_results]
    ]