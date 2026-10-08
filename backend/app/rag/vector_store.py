"""
rag/vector_store.py
-------------------
ChromaDB wrapper for vector storage and similarity search.

Architecture:
  - One ChromaDB collection per persona profile (isolated namespacing).
  - Collections store: persona trait/backstory chunks AND long-term memory
    facts extracted from conversations.
  - Collection naming: f"profile_{profile_id}"

Swappability: Replace this file with a Pinecone/Weaviate wrapper and keep
the same public API (add_texts, query, delete_collection) — nothing else changes.
"""

import logging
import uuid
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.settings import settings
from app.rag.embedder import embed_text, embed_batch
from app.utils.error_handler import RAGError

logger = logging.getLogger(__name__)

# Module-level singleton client — one connection for the whole app lifetime
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    """Return (or lazily create) the shared ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        logger.info("Initialising ChromaDB at: %s", settings.chroma_persist_dir)
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def _collection_name(profile_id: str) -> str:
    """Deterministic, safe collection name for a profile."""
    # ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens
    return f"profile-{profile_id.replace('_', '-')}"


def get_or_create_collection(profile_id: str):
    """Return the ChromaDB collection for a profile, creating it if absent."""
    client = _get_client()
    name = _collection_name(profile_id)
    try:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
        )
    except Exception as exc:
        raise RAGError(f"Failed to get/create collection for {profile_id}: {exc}") from exc


def add_texts(
    profile_id: str,
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
) -> None:
    """
    Add a list of text chunks to a profile's ChromaDB collection.

    Args:
        profile_id: The persona's ID (determines which collection to use).
        texts:      List of text strings to embed and store.
        metadatas:  Optional list of metadata dicts (same length as texts).
        ids:        Optional list of unique string IDs. Auto-generated if None.
    """
    if not texts:
        return

    try:
        collection = get_or_create_collection(profile_id)
        embeddings = embed_batch(texts)
        doc_ids = ids or [str(uuid.uuid4()) for _ in texts]
        metas = metadatas or [{} for _ in texts]

        collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )
        logger.debug("Added %d chunks to collection '%s'", len(texts), profile_id)
    except Exception as exc:
        raise RAGError(f"Failed to add texts to vector store: {exc}") from exc


def query(
    profile_id: str,
    query_text: str,
    top_k: int = 4,
    where: Optional[dict] = None,
) -> List[str]:
    """
    Retrieve the top-k most semantically relevant text chunks for a query.

    Args:
        profile_id:  Profile whose collection to search.
        query_text:  The query string (user message or topic).
        top_k:       Number of results to return.
        where:       Optional ChromaDB metadata filter dict.

    Returns:
        List of relevant text strings (documents), sorted by relevance.
    """
    try:
        collection = get_or_create_collection(profile_id)
        count = collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        query_embedding = embed_text(query_text)

        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents"],
        )
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        docs = results.get("documents", [[]])[0]
        return docs
    except Exception as exc:
        logger.warning("Vector store query failed for %s: %s", profile_id, exc)
        return []  # Degrade gracefully — chat continues without RAG context


def delete_collection(profile_id: str) -> None:
    """Delete a profile's entire ChromaDB collection (e.g., for reset)."""
    try:
        client = _get_client()
        client.delete_collection(_collection_name(profile_id))
        logger.info("Deleted collection for profile: %s", profile_id)
    except Exception as exc:
        logger.warning("Could not delete collection for %s: %s", profile_id, exc)


def seed_persona_knowledge(profile_id: str, persona: dict) -> None:
    """
    Index a persona's backstory and traits into its ChromaDB collection at startup.
    Idempotent: checks if already seeded before writing.

    Args:
        profile_id: Persona ID string.
        persona:    Loaded persona config dict from persona_loader.
    """
    try:
        collection = get_or_create_collection(profile_id)
        # Skip re-seeding if collection already has entries
        if collection.count() > 0:
            logger.debug("Collection for %s already seeded — skipping", profile_id)
            return

        chunks = []
        metas = []
        ids = []

        # Index the backstory as a single chunk
        backstory = persona.get("backstory", "")
        if backstory:
            chunks.append(backstory)
            metas.append({"type": "backstory", "profile_id": profile_id})
            ids.append(f"{profile_id}_backstory")

        # Index each personality trait as a separate chunk for fine-grained retrieval
        for i, trait in enumerate(persona.get("personality_traits", [])):
            trait_text = f"{persona['name']} is {trait}."
            chunks.append(trait_text)
            metas.append({"type": "trait", "profile_id": profile_id})
            ids.append(f"{profile_id}_trait_{i}")

        # Index conversation style rules
        for i, rule in enumerate(persona.get("conversation_style_rules", [])):
            chunks.append(rule)
            metas.append({"type": "style_rule", "profile_id": profile_id})
            ids.append(f"{profile_id}_style_{i}")

        if chunks:
            add_texts(profile_id, chunks, metadatas=metas, ids=ids)
            logger.info("Seeded %d knowledge chunks for persona: %s", len(chunks), profile_id)

    except Exception as exc:
        logger.warning("Failed to seed persona knowledge for %s: %s", profile_id, exc)


def delete_persona_collection(profile_id: str) -> None:
    """
    Delete ChromaDB collection for a profile if it exists.
    """
    try:
        client = _get_client()
        name = _collection_name(profile_id)
        client.delete_collection(name=name)
        logger.info("Deleted ChromaDB collection for %s (%s)", profile_id, name)
    except Exception as exc:
        logger.warning("ChromaDB collection for %s could not be deleted (might not exist): %s", profile_id, exc)

