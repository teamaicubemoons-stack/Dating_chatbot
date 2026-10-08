"""
app/main.py
-----------
FastAPI application entry point.

Startup sequence:
  1. Configure logging.
  2. Initialize SQLite DB (create tables).
  3. Sync persona configs → DB Profile rows.
  4. Seed persona knowledge into ChromaDB (idempotent).
  5. Initialize LLMStatus row in DB.
  6. Start APScheduler background job.
  7. Register API routers.

Shutdown sequence:
  1. Stop the APScheduler gracefully.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.logging_config import setup_logging
from app.config.settings import settings
from app.api.dependencies import init_db, SessionLocal
from app.api import routes_chat, routes_profiles, routes_health
from app.models.db_models import LLMStatus, Profile
from app.personas.persona_loader import load_all_personas
from app.rag.vector_store import seed_persona_knowledge
from app.scheduler.groq_health_check import start_scheduler, stop_scheduler

# Configure logging before any other imports that might log
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager — handles startup and shutdown."""
    # ---- STARTUP ----
    logger.info("=" * 60)
    logger.info("AI Companion Platform — Starting up")
    logger.info("=" * 60)

    # 1. Initialize database tables
    logger.info("Initializing database...")
    init_db()

    db = SessionLocal()
    try:
        # 2. Ensure LLMStatus row exists (id=1 is the singleton row)
        if db.get(LLMStatus, 1) is None:
            db.add(LLMStatus(id=1, current_provider="groq", groq_available=True))
            db.commit()
            logger.info("LLMStatus row created — default provider: groq")

        # 3. Sync persona JSON configs → DB Profile rows
        personas = load_all_personas()
        for persona in personas:
            existing = db.get(Profile, persona["id"])
            if existing is None:
                db.add(Profile(
                    id=persona["id"],
                    name=persona["name"],
                    short_bio=persona.get("short_bio", ""),
                    avatar_url=persona.get("avatar_url", ""),
                ))
                logger.info("Registered new profile in DB: %s", persona["name"])
        db.commit()

        # 4. Seed persona knowledge into ChromaDB (idempotent)
        logger.info("Seeding persona knowledge into ChromaDB...")
        for persona in personas:
            seed_persona_knowledge(persona["id"], persona)

    finally:
        db.close()

    # 5. Start background scheduler
    start_scheduler()

    logger.info("Startup complete — %d personas loaded", len(personas))
    logger.info("=" * 60)

    yield  # App is now running and serving requests

    # ---- SHUTDOWN ----
    logger.info("Shutting down AI Companion Platform...")
    stop_scheduler()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Factory function that creates and configures the FastAPI application."""
    app = FastAPI(
        title="AI Companion Platform",
        description=(
            "A modular AI companion chatbot platform with swappable personas, "
            "RAG-enhanced memory, and transparent LLM provider routing."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow frontend dev server, any *.onrender.com subdomain, and custom env origins
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]
    if settings.cors_origins:
        allowed_origins.extend([o.strip() for o in settings.cors_origins.split(",") if o.strip()])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"https://.*\.onrender\.com",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(routes_profiles.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_health.router)

    # Check for built frontend in common locations (Docker / Hugging Face Spaces)
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    base_dir = Path(__file__).resolve().parent.parent
    possible_dist_paths = [
        base_dir.parent / "frontend" / "dist",
        base_dir / "static",
        Path("/app/frontend/dist"),
        Path("/home/user/app/frontend/dist"),
    ]
    frontend_dist = next((p for p in possible_dist_paths if (p / "index.html").exists()), None)

    if frontend_dist:
        logger.info("Mounted frontend static files from: %s", frontend_dist)
        assets_dir = frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        avatars_dir = frontend_dist / "avatars"
        if avatars_dir.exists():
            app.mount("/avatars", StaticFiles(directory=str(avatars_dir)), name="avatars")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            file_path = frontend_dist / full_path
            if full_path and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")
    else:
        @app.get("/", tags=["root"])
        def root():
            return {
                "name": "AI Companion Platform API",
                "version": "1.0.0",
                "docs": "/docs",
            }

    return app


app = create_app()
