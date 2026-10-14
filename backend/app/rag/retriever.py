"""
rag/retriever.py
----------------
High-level RAG retrieval interface. Combines:
  1. Persona backstory & trait knowledge retrieval from ChromaDB.
  2. Long-term user memories retrieval with multi-signal scoring
     (semantic similarity, keyword overlap, importance, recency, confidence).
  3. Strict isolation:
     - user_id is enforced on EVERY memory check (cross-user leakage impossible).
     - relationship memories are strictly scoped to the current profile_id.

Follows Section 7 of the Master Implementation Plan.
"""

import logging
import math
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from app.rag.vector_store import query, add_texts, delete_documents, delete_documents_by_metadata
from app.rag.embedder import embed_text
from app.memory.memory_store import get_active_user_memories
from app.models.db_models import UserMemory

logger = logging.getLogger(__name__)

# Weights for multi-signal memory ranking
W_SEMANTIC = 0.40
W_KEYWORD = 0.25
W_IMPORTANCE = 0.15
W_RECENCY = 0.10
W_CONFIDENCE = 0.10

_PERSONA_CHUNKS = 2
_MAX_RETRIEVED_MEMORIES = 4


def _compute_keyword_overlap(query_text: str, doc_text: str) -> float:
    """Compute normalized token overlap ratio between query and document."""
    q_words = set(re.findall(r"\w+", query_text.lower()))
    if not q_words:
        return 0.0
    doc_words = set(re.findall(r"\w+", doc_text.lower()))
    if not doc_words:
        return 0.0
    intersection = q_words.intersection(doc_words)
    return len(intersection) / len(q_words)


def _compute_recency_score(created_at: Optional[datetime]) -> float:
    """Exponential decay recency score normalized between 0.0 and 1.0 (half-life: 14 days)."""
    if not created_at:
        return 0.5
    age_days = (datetime.utcnow() - created_at).total_seconds() / 86400.0
    return math.exp(-0.05 * max(0.0, age_days))


def _dot_product(vec_a: List[float], vec_b: List[float]) -> float:
    """Dot product of two unit vectors (cosine similarity)."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def retrieve_context(
    profile_id: str,
    user_message: str,
    user_id: str,
    db: Optional[Session] = None,
) -> str:
    """
    Retrieve and format context for a chat turn.
    Returns clean context string combining:
      - Relevant persona backstory details
      - Scored relevant user memories and relationship facts
    """
    context_sections: List[str] = []

    # --- 1. Persona knowledge retrieval ---
    try:
        persona_chunks = query(
            profile_id=profile_id,
            query_text=user_message,
            top_k=_PERSONA_CHUNKS,
            where={"$or": [{"type": {"$eq": "backstory"}}, {"type": {"$eq": "trait"}}]},
        )
        if persona_chunks:
            # Deduplicate persona chunks
            unique_persona = list(dict.fromkeys(persona_chunks))
            context_sections.append(
                "PERSONA BACKGROUND FACTS:\n" + "\n".join(f"• {c}" for c in unique_persona)
            )
            logger.debug("Retrieved %d persona knowledge chunks", len(unique_persona))
    except Exception as exc:
        logger.warning("Persona RAG retrieval failed: %s", exc)

    # --- 2. Long-term user memories retrieval (Multi-signal Scored Ranking) ---
    if db:
        try:
            active_mems = get_active_user_memories(
                db=db,
                user_id=user_id,
                profile_id=profile_id,
            )

            if active_mems:
                query_vec = embed_text(user_message)
                scored_mems: List[Tuple[float, UserMemory]] = []

                for mem in active_mems:
                    # Enforce strict scope: relationship memories ONLY for this persona
                    if mem.memory_type == "relationship" and mem.profile_id != profile_id:
                        continue

                    # 1. Semantic similarity
                    mem_vec = embed_text(mem.memory_text)
                    sem_sim = max(0.0, min(1.0, _dot_product(query_vec, mem_vec)))

                    # 2. Keyword match
                    kw_score = _compute_keyword_overlap(user_message, mem.memory_text)

                    # 3. Importance & Confidence
                    imp_score = max(0.0, min(1.0, mem.importance or 0.5))
                    conf_score = max(0.0, min(1.0, mem.confidence or 1.0))

                    # 4. Recency
                    rec_score = _compute_recency_score(mem.last_confirmed_at or mem.created_at)

                    # Total composite score
                    total_score = (
                        (W_SEMANTIC * sem_sim)
                        + (W_KEYWORD * kw_score)
                        + (W_IMPORTANCE * imp_score)
                        + (W_RECENCY * rec_score)
                        + (W_CONFIDENCE * conf_score)
                    )

                    scored_mems.append((total_score, mem))

                # Sort descending by composite score
                scored_mems.sort(key=lambda x: x[0], reverse=True)

                # Select top memories
                top_mems = [m for score, m in scored_mems[:_MAX_RETRIEVED_MEMORIES] if score > 0.20]

                if top_mems:
                    mem_lines = []
                    for m in top_mems:
                        label = f"[{m.memory_type.replace('_', ' ').title()}]"
                        mem_lines.append(f"• {label}: {m.memory_text}")

                    context_sections.append(
                        "KNOWN FACTS & MEMORIES ABOUT USER:\n" + "\n".join(mem_lines)
                    )
                    logger.debug("Injected %d scored user memories for user %s", len(top_mems), user_id[:8])

        except Exception as exc:
            logger.warning("User memory scored retrieval failed: %s", exc)

    return "\n\n".join(context_sections) if context_sections else ""


def store_memory_fact(
    profile_id: str,
    user_id: str,
    fact: str,
    fact_id: str,
) -> None:
    """Store distilled user memory in ChromaDB for semantic retrieval."""
    try:
        add_texts(
            profile_id=profile_id,
            texts=[fact],
            metadatas=[{"type": "memory", "user_id": user_id, "memory_id": fact_id}],
            ids=[f"mem_{fact_id}"],
        )
        logger.debug("Stored memory fact in ChromaDB: id=%s", fact_id)
    except Exception as exc:
        logger.warning("Failed to store memory fact in ChromaDB: %s", exc)


def delete_memory_from_vector_store(profile_id: str, memory_id: str) -> None:
    """Remove a specific memory document from ChromaDB."""
    try:
        delete_documents(profile_id, [f"mem_{memory_id}"])
    except Exception as exc:
        logger.warning("Failed to delete memory %s from vector store: %s", memory_id, exc)


def forget_user_vector_memories(user_id: str, profile_id: Optional[str] = None) -> None:
    """Remove user's memory vector records."""
    try:
        if profile_id:
            delete_documents_by_metadata(profile_id, {"$and": [{"type": "memory"}, {"user_id": user_id}]})
        else:
            from app.personas.persona_loader import load_all_personas
            for p in load_all_personas():
                delete_documents_by_metadata(p["id"], {"$and": [{"type": "memory"}, {"user_id": user_id}]})
    except Exception as exc:
        logger.warning("Failed to purge vector memories for user %s: %s", user_id, exc)
