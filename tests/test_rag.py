from app.rag import (
    build_embedding_cache,
    build_or_load_index,
    calculate_knowledge_fingerprint,
    chunk_text,
    clear_embedding_cache,
    create_knowledge_chunks,
    get_embedding_cache,
    knowledge_has_changed,
    load_embedding_index,
    load_knowledge_documents,
    retrieve_knowledge,
    save_embedding_index,
)


# =========================================================
# LOAD KNOWLEDGE
# =========================================================

def test_load_knowledge():

    documents = (
        load_knowledge_documents()
    )

    assert len(documents) >= 1

    sources = [
        document["source"]
        for document in documents
    ]

    assert (
        "cutter_troubleshooting.txt"
        in sources
    )


# =========================================================
# CHUNK TEXT
# =========================================================

def test_chunk_text():

    text = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    chunks = chunk_text(
        text=text,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) >= 3

    assert (
        chunks[0]
        == "ABCDEFGHIJ"
    )

    assert (
        chunks[1].startswith(
            "IJ"
        )
    )


# =========================================================
# INVALID CHUNK SIZE
# =========================================================

def test_chunk_text_invalid_chunk_size():

    try:

        chunk_text(
            text="sample text",
            chunk_size=0,
            overlap=0,
        )

        assert False

    except ValueError as error:

        assert (
            "chunk_size"
            in str(error)
        )


# =========================================================
# INVALID OVERLAP
# =========================================================

def test_chunk_text_invalid_overlap():

    try:

        chunk_text(
            text="sample text",
            chunk_size=100,
            overlap=100,
        )

        assert False

    except ValueError as error:

        assert (
            "overlap"
            in str(error)
        )


# =========================================================
# CREATE KNOWLEDGE CHUNKS
# =========================================================

def test_create_knowledge_chunks():

    chunks = (
        create_knowledge_chunks(
            chunk_size=300,
            overlap=50,
        )
    )

    assert len(chunks) >= 1

    first_chunk = (
        chunks[0]
    )

    assert (
        "source"
        in first_chunk
    )

    assert (
        "chunk_id"
        in first_chunk
    )

    assert (
        "chunk_index"
        in first_chunk
    )

    assert (
        "content"
        in first_chunk
    )


# =========================================================
# CHUNK METADATA
# =========================================================

def test_chunk_metadata():

    chunks = (
        create_knowledge_chunks(
            chunk_size=300,
            overlap=50,
        )
    )

    assert len(chunks) >= 1

    first_chunk = (
        chunks[0]
    )

    assert (
        first_chunk["source"]
        == "cutter_troubleshooting.txt"
    )

    assert (
        first_chunk["chunk_index"]
        >= 0
    )

    assert (
        "cutter_troubleshooting.txt"
        in first_chunk["chunk_id"]
    )


# =========================================================
# FINGERPRINT
# =========================================================

def test_knowledge_fingerprint_exists():

    fingerprint = (
        calculate_knowledge_fingerprint()
    )

    assert isinstance(
        fingerprint,
        str,
    )

    assert (
        len(fingerprint)
        == 64
    )


# =========================================================
# EMBEDDING CACHE BUILD
# =========================================================

def test_embedding_cache_builds():

    clear_embedding_cache()

    cache = (
        build_embedding_cache()
    )

    assert len(cache) >= 1

    assert (
        "embedding"
        in cache[0]
    )

    assert (
        "content"
        in cache[0]
    )

    assert (
        "source"
        in cache[0]
    )


# =========================================================
# CACHE REUSE
# =========================================================

def test_embedding_cache_is_reused():

    clear_embedding_cache()

    first_cache = (
        get_embedding_cache()
    )

    second_cache = (
        get_embedding_cache()
    )

    assert (
        first_cache
        is second_cache
    )


# =========================================================
# CLEAR CACHE
# =========================================================

def test_clear_embedding_cache():

    clear_embedding_cache()

    first_cache = (
        get_embedding_cache()
    )

    assert (
        len(first_cache)
        >= 1
    )

    clear_embedding_cache()

    second_cache = (
        get_embedding_cache()
    )

    assert (
        len(second_cache)
        >= 1
    )

    assert (
        first_cache
        is not second_cache
    )


# =========================================================
# KNOWLEDGE HAS NOT CHANGED AFTER BUILD
# =========================================================

def test_knowledge_has_not_changed_after_build():

    clear_embedding_cache()

    build_embedding_cache()

    assert (
        knowledge_has_changed()
        is False
    )


# =========================================================
# KNOWLEDGE CHANGE DETECTION
# =========================================================

def test_knowledge_change_detection(
    tmp_path,
    monkeypatch,
):

    from app import rag

    test_knowledge_dir = (
        tmp_path
        / "knowledge"
    )

    test_knowledge_dir.mkdir()

    test_file = (
        test_knowledge_dir
        / "test.txt"
    )

    test_file.write_text(
        "Initial cutter knowledge.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rag,
        "KNOWLEDGE_DIR",
        test_knowledge_dir,
    )

    rag.clear_embedding_cache()

    rag.build_embedding_cache()

    assert (
        rag.knowledge_has_changed()
        is False
    )

    test_file.write_text(
        (
            "Initial cutter knowledge. "
            "Added abnormal vibration guidance."
        ),
        encoding="utf-8",
    )

    assert (
        rag.knowledge_has_changed()
        is True
    )


# =========================================================
# SAVE INDEX
# =========================================================

def test_embedding_index_can_be_saved():

    clear_embedding_cache()

    build_embedding_cache()

    save_embedding_index()

    from app import rag

    assert (
        rag.INDEX_PATH.exists()
    )


# =========================================================
# LOAD INDEX
# =========================================================

def test_embedding_index_can_be_loaded():

    clear_embedding_cache()

    build_embedding_cache()

    save_embedding_index()

    clear_embedding_cache()

    loaded = (
        load_embedding_index()
    )

    assert (
        loaded is True
    )

    cache = (
        get_embedding_cache()
    )

    assert (
        len(cache)
        >= 1
    )

    assert (
        "embedding"
        in cache[0]
    )


# =========================================================
# BUILD OR LOAD INDEX
# =========================================================

def test_build_or_load_index():

    clear_embedding_cache()

    cache = (
        build_or_load_index()
    )

    assert (
        len(cache)
        >= 1
    )

    assert (
        "embedding"
        in cache[0]
    )


# =========================================================
# INDEX SURVIVES MEMORY CLEAR
# =========================================================

def test_saved_index_survives_memory_cache_clear():

    clear_embedding_cache()

    build_embedding_cache()

    save_embedding_index()

    from app import rag

    assert (
        rag.INDEX_PATH.exists()
    )

    clear_embedding_cache()

    assert (
        rag._embedding_cache
        is None
    )

    loaded = (
        load_embedding_index()
    )

    assert (
        loaded is True
    )

    assert (
        rag._embedding_cache
        is not None
    )


# =========================================================
# BUILD OR LOAD RETURNS VALID CHUNKS
# =========================================================

def test_build_or_load_returns_valid_chunks():

    clear_embedding_cache()

    cache = (
        build_or_load_index()
    )

    assert (
        len(cache)
        >= 1
    )

    first_chunk = (
        cache[0]
    )

    assert (
        first_chunk["source"]
        == "cutter_troubleshooting.txt"
    )

    assert (
        "embedding"
        in first_chunk
    )

    assert (
        "content"
        in first_chunk
    )


# =========================================================
# INDEX REBUILDS AFTER KNOWLEDGE CHANGE
# =========================================================

def test_index_rebuilds_after_knowledge_change(
    tmp_path,
    monkeypatch,
):

    from app import rag

    test_knowledge_dir = (
        tmp_path
        / "knowledge"
    )

    test_data_dir = (
        tmp_path
        / "data"
    )

    test_knowledge_dir.mkdir()
    test_data_dir.mkdir()

    test_file = (
        test_knowledge_dir
        / "test.txt"
    )

    test_index_path = (
        test_data_dir
        / "vector_index.pkl"
    )

    monkeypatch.setattr(
        rag,
        "KNOWLEDGE_DIR",
        test_knowledge_dir,
    )

    monkeypatch.setattr(
        rag,
        "INDEX_PATH",
        test_index_path,
    )

    test_file.write_text(
        (
            "Cutter vibration "
            "inspection guidance."
        ),
        encoding="utf-8",
    )

    rag.clear_embedding_cache()

    first_cache = (
        rag.build_or_load_index()
    )

    first_fingerprint = (
        rag.calculate_knowledge_fingerprint()
    )

    assert (
        len(first_cache)
        >= 1
    )

    assert (
        test_index_path.exists()
    )

    test_file.write_text(
        (
            "Cutter vibration "
            "inspection guidance. "
            "Also inspect bearing "
            "temperature and "
            "cutter alignment."
        ),
        encoding="utf-8",
    )

    second_fingerprint = (
        rag.calculate_knowledge_fingerprint()
    )

    assert (
        first_fingerprint
        != second_fingerprint
    )

    assert (
        rag.knowledge_has_changed()
        is True
    )

    second_cache = (
        rag.build_or_load_index()
    )

    assert (
        len(second_cache)
        >= 1
    )

    assert (
        rag._embedding_fingerprint
        == second_fingerprint
    )

    assert (
        rag.knowledge_has_changed()
        is False
    )


# =========================================================
# SEMANTIC RETRIEVAL
# =========================================================

def test_retrieve_cutter_knowledge():

    clear_embedding_cache()

    results = (
        retrieve_knowledge(
            (
                "PET cutter tripped repeatedly "
                "with abnormal vibration"
            )
        )
    )

    assert (
        len(results)
        >= 1
    )

    assert (
        results[0]["source"]
        == "cutter_troubleshooting.txt"
    )

    assert (
        results[0]["score"]
        > 0
    )

    assert (
        "chunk_id"
        in results[0]
    )

    assert (
        "chunk_index"
        in results[0]
    )

    assert (
        "content"
        in results[0]
    )


# =========================================================
# SEMANTIC RETRIEVAL WITHOUT EXACT WORD MATCH
# =========================================================

def test_semantic_retrieval_without_exact_word_match():

    clear_embedding_cache()

    results = (
        retrieve_knowledge(
            (
                "The cutter is shaking badly "
                "and stopping repeatedly."
            )
        )
    )

    assert (
        len(results)
        >= 1
    )

    assert (
        results[0]["source"]
        == "cutter_troubleshooting.txt"
    )


# =========================================================
# MAX RESULTS
# =========================================================

def test_retrieval_respects_max_results():

    clear_embedding_cache()

    results = (
        retrieve_knowledge(
            (
                "cutter vibration trip "
                "bearing motor blade"
            ),
            max_results=2,
        )
    )

    assert (
        len(results)
        <= 2
    )


# =========================================================
# SCORE SORTING
# =========================================================

def test_retrieval_scores_are_sorted():

    clear_embedding_cache()

    results = (
        retrieve_knowledge(
            (
                "cutter vibration trip "
                "bearing motor blade"
            ),
            max_results=3,
        )
    )

    if len(results) > 1:

        scores = [
            item["score"]
            for item in results
        ]

        assert (
            scores
            == sorted(
                scores,
                reverse=True,
            )
        )


# =========================================================
# EMPTY QUERY
# =========================================================

def test_empty_query_returns_no_results():

    results = (
        retrieve_knowledge(
            "   "
        )
    )

    assert (
        results
        == []
    )


# =========================================================
# ZERO MAX RESULTS
# =========================================================

def test_zero_max_results_returns_no_results():

    results = (
        retrieve_knowledge(
            "cutter vibration",
            max_results=0,
        )
    )

    assert (
        results
        == []
    )