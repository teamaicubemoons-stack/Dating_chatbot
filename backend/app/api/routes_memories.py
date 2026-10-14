"""
api/routes_memories.py
----------------------
FastAPI endpoints for memory management, user correction, and privacy controls.

Follows Section 13 & 15 of the Master Implementation Plan:
  - GET    /memories              : list active memories for user
  - PATCH  /memories/{memory_id}  : correct or update memory content
  - DELETE /memories/{memory_id}  : delete individual memory
  - POST   /memories/forget       : bulk forget by scope ('all', 'relationship', 'profile_facts')
  - GET    /privacy/data-export   : GDPR / privacy user data export
  - DELETE /account/data          : permanent account and data deletion
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.memory.memory_store import (
    get_active_user_memories, update_user_memory, delete_user_memory,
    forget_memories, export_user_data, delete_user_account_data
)
from app.rag.retriever import delete_memory_from_vector_store, forget_user_vector_memories
from app.models.schemas import (
    UserMemoryOut, UserMemoryPatch, MemoryForgetRequest,
    DataExportResponse, AccountDeleteResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memories"])


@router.get("/memories", response_model=List[UserMemoryOut])
def list_memories(
    user_id: str = Query(..., description="User ID for memory isolation"),
    profile_id: Optional[str] = Query(None, description="Optional persona profile ID filter"),
    memory_type: Optional[str] = Query(None, description="Optional memory type filter"),
    db: Session = Depends(get_db),
):
    """
    List all active, non-expired memories belonging strictly to the requested user.
    Enforces user isolation and profile scoping.
    """
    types_list = [memory_type] if memory_type else None
    memories = get_active_user_memories(
        db=db,
        user_id=user_id,
        profile_id=profile_id,
        memory_types=types_list,
    )
    return memories


@router.patch("/memories/{memory_id}", response_model=UserMemoryOut)
def update_memory(
    memory_id: str,
    payload: UserMemoryPatch,
    user_id: str = Query(..., description="User ID required for authorization"),
    db: Session = Depends(get_db),
):
    """
    User-requested memory correction or importance adjustment.
    Ensures memory ownership belongs to user_id.
    """
    updated = update_user_memory(
        db=db,
        memory_id=memory_id,
        user_id=user_id,
        memory_text=payload.memory_text,
        importance=payload.importance,
        status=payload.status,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found or not owned by user",
        )
    return updated


@router.delete("/memories/{memory_id}")
def delete_single_memory(
    memory_id: str,
    user_id: str = Query(..., description="User ID required for authorization"),
    db: Session = Depends(get_db),
):
    """
    Soft-delete an individual memory and invalidate it from the vector index.
    """
    # Fetch memory first to determine profile_id for vector deletion
    from app.models.db_models import UserMemory
    mem = db.query(UserMemory).filter(UserMemory.id == memory_id, UserMemory.user_id == user_id).first()
    if not mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found or not owned by user",
        )

    profile_id = mem.profile_id or "profile_1"
    deleted = delete_user_memory(db=db, memory_id=memory_id, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Delete operation failed")

    # Invalidate in ChromaDB
    delete_memory_from_vector_store(profile_id=profile_id, memory_id=memory_id)

    return {"deleted": True, "id": memory_id}


@router.post("/memories/forget")
def forget_memories_endpoint(
    payload: MemoryForgetRequest,
    db: Session = Depends(get_db),
):
    """
    Bulk delete memories according to requested scope ('all', 'relationship', 'profile_facts').
    """
    count = forget_memories(
        db=db,
        user_id=payload.user_id,
        scope=payload.scope,
        profile_id=payload.profile_id,
    )

    # Invalidate vector store records
    forget_user_vector_memories(user_id=payload.user_id, profile_id=payload.profile_id)

    return {
        "success": True,
        "scope": payload.scope,
        "forgotten_count": count,
        "user_id": payload.user_id,
    }


@router.get("/privacy/data-export", response_model=DataExportResponse)
def export_user_data_endpoint(
    user_id: str = Query(..., description="User ID to export"),
    db: Session = Depends(get_db),
):
    """
    GDPR / Privacy data export: provides all stored messages, memories,
    summaries, and state for the authenticated user.
    """
    data = export_user_data(db=db, user_id=user_id)
    return data


@router.delete("/account/data", response_model=AccountDeleteResponse)
def delete_account_endpoint(
    user_id: str = Query(..., description="User ID to permanently delete"),
    db: Session = Depends(get_db),
):
    """
    Permanently delete all user account data, messages, memories, and state.
    """
    deleted_counts = delete_user_account_data(db=db, user_id=user_id)
    forget_user_vector_memories(user_id=user_id)

    return AccountDeleteResponse(
        success=True,
        user_id=user_id,
        deleted_records=deleted_counts,
        message="All account records and memories have been permanently erased.",
    )
