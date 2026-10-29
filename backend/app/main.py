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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config.logging_config import setup_logging
from app.config.settings import settings
from app.api.dependencies import init_db, SessionLocal
from app.api import routes_chat, routes_profiles, routes_health, routes_memories
from app.models.db_models import LLMStatus, Profile
from app.personas.persona_loader import load_all_personas
from app.rag.vector_store import seed_persona_knowledge
from app.scheduler.openai_health_check import start_scheduler, stop_scheduler

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
        existing_status = db.get(LLMStatus, 1)
        if existing_status is None:
            db.add(LLMStatus(id=1, current_provider="openai", groq_available=False))
            db.commit()
            logger.info("LLMStatus row created — provider: openai (gpt-5.6-luna)")
        elif existing_status.current_provider != "openai":
            existing_status.current_provider = "openai"
            db.commit()
            logger.info("LLMStatus updated → openai")

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
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.\d+\.\d+\.\d+)(:\d+)?$|https://.*\.onrender\.com|https://.*\.hf\.space|https://huggingface\.co",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(routes_profiles.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_health.router)
    app.include_router(routes_memories.router)

    # Check for built frontend in common locations (Docker / Hugging Face Spaces / Render)
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, HTMLResponse
    from starlette.routing import Mount

    base_dir = Path(__file__).resolve().parent.parent
    possible_dist_paths = [
        base_dir.parent / "frontend" / "dist",
        base_dir / "static",
        Path("/app/frontend/dist"),
        Path("/home/user/app/frontend/dist"),
    ]
    frontend_dist = next((p for p in possible_dist_paths if (p / "index.html").exists()), None)

    possible_avatar_paths = [
        base_dir.parent / "avatars",
        base_dir / "static" / "avatars",
        (frontend_dist / "avatars") if frontend_dist else None,
        base_dir.parent / "frontend" / "public" / "avatars",
        Path("/app/avatars"),
        Path("/home/user/app/avatars"),
    ]
    possible_avatar_paths = [p for p in possible_avatar_paths if p and p.exists()]
    avatars_dir = possible_avatar_paths[0] if possible_avatar_paths else None

    if avatars_dir:
        app.router.routes.insert(0, Mount("/avatars", StaticFiles(directory=str(avatars_dir)), name="avatars_top"))
        app.router.routes.insert(0, Mount("/gradio_api/avatars", StaticFiles(directory=str(avatars_dir)), name="gradio_avatars_top"))

    @app.get("/avatars/{filename:path}", include_in_schema=False)
    @app.get("/gradio_api/avatars/{filename:path}", include_in_schema=False)
    def serve_avatar(filename: str):
        for candidate_dir in possible_avatar_paths:
            f = candidate_dir / filename
            if f.is_file():
                return FileResponse(f)
        return HTMLResponse("Avatar not found", status_code=404)

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    def serve_favicon():
        if frontend_dist:
            fav = frontend_dist / "favicon.svg"
            if fav.exists():
                return FileResponse(fav)
        return HTMLResponse("", status_code=204)

    if frontend_dist:
        logger.info("Mounted frontend static files from: %s", frontend_dist)
        possible_asset_paths = [
            frontend_dist / "assets",
            base_dir / "static" / "assets",
            base_dir.parent / "frontend" / "dist" / "assets",
        ]
        assets_dir = next((p for p in possible_asset_paths if p.exists()), None)
        if assets_dir:
            app.router.routes.insert(0, Mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets_top"))

        @app.get("/chat/{profile_id}", response_class=HTMLResponse, include_in_schema=False)
        @app.get("/chat/{profile_id}/", response_class=HTMLResponse, include_in_schema=False)
        @app.get("/settings", response_class=HTMLResponse, include_in_schema=False)
        @app.get("/settings/", response_class=HTMLResponse, include_in_schema=False)
        @app.get("/about", response_class=HTMLResponse, include_in_schema=False)
        @app.get("/privacy", response_class=HTMLResponse, include_in_schema=False)
        def serve_spa_page(profile_id: str = ""):
            return FileResponse(frontend_dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            file_path = frontend_dist / full_path
            if full_path and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")

        @app.exception_handler(404)
        async def spa_404_handler(request: Request, exc):
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                idx = frontend_dist / "index.html"
                if idx.is_file():
                    return FileResponse(idx)
            return HTMLResponse("Not Found", status_code=404)
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
