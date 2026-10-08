"""
api/routes_profiles.py
-----------------------
FastAPI router for profile/persona-related endpoints.

Endpoints:
  GET /profiles          → list all persona summaries (for homepage grid)
  GET /profiles/{id}     → full public persona detail (for profile page)

IMPORTANT: The raw system_prompt_template and internal config fields are
NEVER returned by these endpoints. Only public-safe fields are exposed.
"""

import json
import logging
import re
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.personas.persona_loader import (
    _CONFIGS_DIR,
    load_all_personas,
    load_persona,
    get_public_persona_info,
    delete_persona_config,
)
from app.models.schemas import ProfileSummary, ProfileDetail, ProfileCreate
from app.models.db_models import Profile as DBProfile
from app.rag.vector_store import seed_persona_knowledge, delete_persona_collection
from app.api.dependencies import get_db
from app.utils.error_handler import PersonaNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("", response_model=List[ProfileSummary])
def list_profiles():
    """
    Return a lightweight list of all available persona profiles.
    Used by the frontend HomePage to render the profile card grid.
    """
    personas = load_all_personas()
    summaries = []
    for p in personas:
        summaries.append(ProfileSummary(
            id=p["id"],
            name=p["name"],
            short_bio=p.get("short_bio", ""),
            avatar_url=p.get("avatar_url", ""),
            country=p.get("country", ""),
            city=p.get("city", ""),
            age=p.get("age"),
            gender=p.get("gender", ""),
        ))
    logger.info("Listed %d profiles", len(summaries))
    return summaries


@router.get("/{profile_id}", response_model=ProfileDetail)
def get_profile(profile_id: str):
    """
    Return full public profile info for a specific persona.
    Private fields (system_prompt_template, style rules) are excluded.
    """
    try:
        persona = load_persona(profile_id)
        public = get_public_persona_info(persona)
        return ProfileDetail(
            id=public["id"],
            name=public["name"],
            age=public.get("age"),
            gender=public.get("gender", ""),
            country=public.get("country", ""),
            city=public.get("city", ""),
            short_bio=public.get("short_bio", ""),
            avatar_url=public.get("avatar_url", ""),
            personality_traits=public.get("personality_traits", []),
            tone=public.get("tone", ""),
            relationship_intent=public.get("relationship_intent", ""),
        )
    except PersonaNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")


@router.post("", response_model=ProfileDetail)
def create_profile(body: ProfileCreate, db: Session = Depends(get_db)):
    """
    Create a new AI dating persona.
    Saves config permanently to personas/configs/ as JSON, syncs to SQLite DB,
    and indexes the backstory into ChromaDB vector memory.
    """
    # 1. Generate unique, clean profile ID
    clean_name = re.sub(r"[^a-zA-Z0-9]", "", body.name.lower().strip()) or "persona"
    unique_suffix = uuid.uuid4().hex[:6]
    profile_id = f"profile_{clean_name}_{unique_suffix}"

    # 2. Format location
    location_str = f"{body.city.strip()}, {body.country.strip()}" if body.city and body.city.strip() else body.country.strip()

    # 3. Assemble detailed backstory
    interests_str = ", ".join(body.interests) if body.interests else "music, travel, good food"
    traits_list = body.personality_traits if body.personality_traits else ["warm", "authentic", "fun"]
    
    if body.backstory and len(body.backstory.strip()) > 10:
        full_backstory = f"{body.name} is {body.age} years old, living in {location_str}. {body.backstory.strip()}"
    else:
        full_backstory = (
            f"{body.name} is a {body.age}-year-old living in {location_str}. "
            f"Bio: {body.short_bio}. Passionate about {interests_str}. "
            f"Vibe: {body.tone or 'warm & playful'}. {body.relationship_intent}."
        )

    # 4. Openers
    openers = [
        f"Hey 🙂 How's your day going?",
        f"Hey. Glad you stopped by.",
        f"Hey! What are you up to today?",
    ]
    if body.initial_opener and body.initial_opener.strip():
        openers.insert(0, body.initial_opener.strip())

    # 5. Assemble JSON config
    persona_dict = {
        "id": profile_id,
        "name": body.name.strip(),
        "age": body.age,
        "gender": body.gender,
        "country": body.country.strip(),
        "city": (body.city or "").strip(),
        "short_bio": body.short_bio.strip(),
        "avatar_url": (body.avatar_url or "").strip(),
        "personality_traits": traits_list,
        "tone": body.tone or "warm & playful",
        "relationship_intent": body.relationship_intent or "",
        "backstory": full_backstory,
        "initial_openers": openers,
        "conversation_style_rules": [
            "Texting style: write like a real person typing on a smartphone — mostly 1 to 2 short sentences.",
            "Acknowledge user emotions first before moving forward.",
            "Never invent fake recent travel or personal anecdotes.",
            "Ask at most one question per message, and frequently zero questions — just react like a real person.",
            "Share genuine opinions and banter, never be a submissive yes-man.",
            "Zero politics and zero religion: gracefully and playfully deflect heavy political/religious debates like a real person changing the topic, never break character or say 'As an AI'."
        ],
        "system_prompt_template": (
            f"You are {body.name}, {body.age} years old, living in {location_str}. "
            f"Tone: {body.tone}. Backstory: {{backstory}}. Style rules: {{conversation_style_rules}}."
        ),
    }

    # 6. Save JSON file to configs directory
    file_path = _CONFIGS_DIR / f"{profile_id}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(persona_dict, f, indent=2, ensure_ascii=False)
        logger.info("Saved new persona config to %s", file_path.name)
    except Exception as exc:
        logger.error("Failed to write persona config file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save profile configuration.")

    # 7. Sync to SQLite DB
    try:
        db_profile = DBProfile(
            id=profile_id,
            name=body.name.strip(),
            short_bio=body.short_bio.strip(),
            avatar_url=body.avatar_url or "",
        )
        db.merge(db_profile)
        db.commit()
    except Exception as exc:
        logger.error("Failed to save profile in SQLite DB: %s", exc)
        db.rollback()

    # 8. Seed into ChromaDB vector store
    try:
        seed_persona_knowledge(profile_id, persona_dict)
        logger.info("Seeded persona knowledge into ChromaDB for %s", profile_id)
    except Exception as exc:
        logger.warning("Could not seed ChromaDB knowledge for %s: %s", profile_id, exc)

    return ProfileDetail(
        id=profile_id,
        name=body.name.strip(),
        age=body.age,
        gender=body.gender,
        country=body.country.strip(),
        city=body.city or "",
        short_bio=body.short_bio.strip(),
        avatar_url=body.avatar_url or "",
        personality_traits=traits_list,
        tone=body.tone or "",
        relationship_intent=body.relationship_intent or "",
    )


@router.delete("/{profile_id}")
def delete_profile(profile_id: str, db: Session = Depends(get_db)):
    """
    Permanently delete a companion persona profile.
    1. Removes persona JSON config file from disk.
    2. Deletes DB profile record (cascading all associated messages).
    3. Cleans up ChromaDB vector collection.
    """
    logger.info("Attempting to delete profile: %s", profile_id)

    # 1. Delete JSON config
    file_deleted = delete_persona_config(profile_id)

    # 2. Delete from DB (cascade deletes messages)
    db_profile = db.query(DBProfile).filter(DBProfile.id == profile_id).first()
    db_deleted = False
    if db_profile:
        try:
            db.delete(db_profile)
            db.commit()
            db_deleted = True
            logger.info("Deleted profile %s from SQLite DB", profile_id)
        except Exception as exc:
            db.rollback()
            logger.error("Error deleting profile %s from DB: %s", profile_id, exc)

    # 3. Clean up ChromaDB collection
    try:
        delete_persona_collection(profile_id)
    except Exception as exc:
        logger.warning("Error cleaning ChromaDB collection for %s: %s", profile_id, exc)

    if not file_deleted and not db_deleted:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")

    return {
        "success": True,
        "profile_id": profile_id,
        "message": f"Profile '{profile_id}' was successfully deleted.",
    }

