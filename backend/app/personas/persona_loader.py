"""
personas/persona_loader.py
--------------------------
Plugin-style persona loading system. This module is the ONLY place that
reads persona JSON files from disk. Everything else receives a parsed dict.

SWAPPABILITY GUARANTEE:
  To deploy this backend for a completely different chatbot product, a developer
  only needs to:
    1. Replace/add JSON files in personas/configs/.
    2. Zero changes to llm/, rag/, memory/, api/, or chat_engine.py.

The loader validates structure, fills optional fields with defaults, and
returns a clean dict ready for downstream use.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.utils.error_handler import PersonaNotFoundError

logger = logging.getLogger(__name__)

# Absolute path to the configs directory — works regardless of CWD
_CONFIGS_DIR = Path(__file__).parent / "configs"

# Fields that must be present in every valid persona config
_REQUIRED_FIELDS = {"id", "name", "backstory", "system_prompt_template"}

# Excluded from public API responses (never sent to frontend)
_PRIVATE_FIELDS = {"system_prompt_template", "conversation_style_rules"}


def _load_raw(path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a single JSON file; return None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            logger.warning("Persona %s missing required fields: %s — skipping", path.name, missing)
            return None
        # Skip the template file
        if data.get("id") == "persona_template":
            return None
        return data
    except Exception as exc:
        logger.error("Failed to load persona config %s: %s", path.name, exc)
        return None


def list_persona_ids() -> List[str]:
    """Return list of all valid persona IDs found in configs/."""
    ids = []
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        data = _load_raw(path)
        if data:
            ids.append(data["id"])
    return ids


def load_all_personas() -> List[Dict[str, Any]]:
    """Load and return all valid persona configs as a list of dicts."""
    personas = []
    for path in sorted(_CONFIGS_DIR.glob("*.json")):
        data = _load_raw(path)
        if data:
            personas.append(_apply_defaults(data))
    logger.info("Loaded %d persona configs from disk", len(personas))
    return personas


def load_persona(persona_id: str) -> Dict[str, Any]:
    """
    Load a single persona by ID.

    Raises:
        PersonaNotFoundError: if no config file with that ID exists.
    """
    for path in _CONFIGS_DIR.glob("*.json"):
        data = _load_raw(path)
        if data and data.get("id") == persona_id:
            return _apply_defaults(data)
    raise PersonaNotFoundError(f"Persona '{persona_id}' not found in configs/")


def delete_persona_config(persona_id: str) -> bool:
    """
    Find and delete the persona JSON config file for the given ID.
    Returns True if deleted, False if not found.
    """
    for path in _CONFIGS_DIR.glob("*.json"):
        # Prevent deleting template file
        if path.name == "persona_template.json":
            continue
        data = _load_raw(path)
        if data and data.get("id") == persona_id:
            try:
                path.unlink()
                logger.info("Deleted persona config file: %s", path.name)
                return True
            except Exception as exc:
                logger.error("Failed to delete persona file %s: %s", path.name, exc)
                raise exc
    return False



def build_system_prompt(persona: Dict[str, Any], rag_context: str = "") -> str:
    """
    Render the persona's system_prompt_template by injecting its own fields.
    Also appends any retrieved RAG context as a memory/context block.

    Args:
        persona:     Loaded persona dict from load_persona().
        rag_context: Retrieved relevant context from vector store (may be empty).

    Returns:
        Fully rendered system prompt string, ready for the LLM.
    """
    template: str = persona["system_prompt_template"]

    class SafeDict(dict):
        def __missing__(self, key):
            return ""

    params = {
        "name": persona.get("name", ""),
        "age": persona.get("age", ""),
        "personality_traits": ", ".join(persona.get("personality_traits", [])),
        "tone": persona.get("tone", ""),
        "backstory": persona.get("backstory", ""),
        "conversation_style_rules": "\n- ".join(persona.get("conversation_style_rules", [])),
    }

    try:
        filled = template.format_map(SafeDict(params))
    except Exception as exc:
        logger.warning("Error formatting persona prompt template: %s", exc)
        filled = template

    # Append RAG context block if available — keeps it separate from core persona prompt
    if rag_context.strip():
        filled += (
            "\n\n---\nRelevant memory / context from past conversations:\n"
            + rag_context.strip()
            + "\n---"
        )

    return filled


def get_public_persona_info(persona: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of the persona dict with private/internal fields stripped out.
    Safe to serialize and send to the frontend.
    """
    return {k: v for k, v in persona.items() if k not in _PRIVATE_FIELDS}


def _apply_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill optional persona fields with sensible defaults."""
    data.setdefault("age", None)
    data.setdefault("short_bio", data.get("backstory", "")[:120] + "…")
    data.setdefault("avatar_url", "")
    data.setdefault("personality_traits", [])
    data.setdefault("tone", "friendly")
    data.setdefault("conversation_style_rules", [
        "Texting style: write like a real person typing on a smartphone — mostly 1 to 2 short sentences.",
        "Zero politics or religious debates: casually and playfully deflect heavy topics like a real person changing the subject, never say 'As an AI'.",
    ])
    data.setdefault("initial_openers", [
        "Hey 🙂 How's your day going?",
        "Hey. What are you up to today?",
        "Hey. Glad you're here."
    ])
    return data
