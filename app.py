import os
import sys
import logging
from pathlib import Path

# Disable ChromaDB telemetry errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ZeroGPU support for Hugging Face Spaces
try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, *args, **kwargs):
            if fn is not None:
                return fn
            return lambda f: f
    spaces = _MockSpaces()

@spaces.GPU
def run_zerogpu_task(x: str = "") -> str:
    """ZeroGPU detection hook required by Hugging Face ZeroGPU runtime."""
    return x

# Ensure backend modules can be imported without namespace collision
base_dir = Path(__file__).resolve().parent
backend_dir = base_dir / "backend"
if str(backend_dir) in sys.path:
    sys.path.remove(str(backend_dir))
sys.path.insert(0, str(backend_dir))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, HTTPException
from starlette.routing import Mount, Route
import gradio as gr

from app.config.settings import settings
from app.api.dependencies import init_db, SessionLocal
from app.models.db_models import LLMStatus, Profile
from app.personas.persona_loader import load_all_personas
from app.rag.vector_store import seed_persona_knowledge
from app.scheduler.openai_health_check import start_scheduler
from app.api import routes_chat, routes_profiles, routes_health, routes_memories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spark_react_app")

# Initialize database, personas, and ChromaDB vector store
try:
    init_db()
    db = SessionLocal()
    if db.get(LLMStatus, 1) is None:
        db.add(LLMStatus(id=1, current_provider="openai", groq_available=False))
        db.commit()

    personas = load_all_personas()
    for persona in personas:
        if isinstance(persona, dict) and "id" in persona:
            existing = db.get(Profile, persona["id"])
            if existing is None:
                db.add(Profile(
                    id=persona["id"],
                    name=persona.get("name", "Companion"),
                    short_bio=persona.get("short_bio", ""),
                    avatar_url=persona.get("avatar_url", ""),
                ))
            seed_persona_knowledge(persona["id"], persona)
    db.commit()
    db.close()

    start_scheduler()
    logger.info("Spark application initialized successfully with %d personas.", len(personas))
except Exception as e:
    logger.error(f"Initialization warning: {e}", exc_info=True)

# Locate static build directories
possible_dist_paths = [
    base_dir / "frontend" / "dist",
    backend_dir / "static",
    base_dir / "static",
]
frontend_dist = next((p for p in possible_dist_paths if (p / "index.html").exists()), base_dir / "frontend" / "dist")
assets_dir = frontend_dist / "assets"

possible_avatar_paths = [
    base_dir / "avatars",
    backend_dir / "static" / "avatars",
    frontend_dist / "avatars",
    base_dir / "frontend" / "public" / "avatars",
    Path("/app/avatars"),
    Path("/home/user/app/avatars"),
]
avatars_dir = next((p for p in possible_avatar_paths if p.exists()), base_dir / "avatars")

# Build Gradio Demo containing the full-screen React Viewport
with gr.Blocks(title="Spark — AI Companions") as demo:
    gr.HTML(
        """
        <style>
            html, body, .gradio-container, .main, .wrap {
                margin: 0 !important;
                padding: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                max-width: 100vw !important;
                overflow: hidden !important;
                background: #f5f4f8 !important;
            }
            footer { display: none !important; }
            #react-viewport {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                border: none;
                margin: 0;
                padding: 0;
                z-index: 999999;
                background: #f5f4f8;
            }
        </style>
        <iframe id="react-viewport" src="/gradio_api/ui" allow="autoplay; camera; microphone; clipboard-write"></iframe>
        """
    )

from gradio.routes import App

def configure_fastapi_app(app):
    # Enable CORS for all API calls (including Render, local, and custom domains)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_origin_regex=r"^https?://.*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount backend API routers under root, /api, and /gradio_api
    for prefix in ["", "/api", "/gradio_api"]:
        app.include_router(routes_profiles.router, prefix=prefix)
        app.include_router(routes_chat.router, prefix=prefix)
        app.include_router(routes_health.router, prefix=prefix)
        app.include_router(routes_memories.router, prefix=prefix)

    # Mount static assets and avatars at the front of routes so they take precedence
    if assets_dir.exists():
        app.router.routes.insert(0, Mount("/gradio_api/assets", StaticFiles(directory=str(assets_dir)), name="gradio_assets"))
        app.router.routes.insert(0, Mount("/assets", StaticFiles(directory=str(assets_dir)), name="custom_assets"))

    if avatars_dir.exists():
        app.router.routes.insert(0, Mount("/gradio_api/avatars", StaticFiles(directory=str(avatars_dir)), name="gradio_avatars"))
        app.router.routes.insert(0, Mount("/avatars", StaticFiles(directory=str(avatars_dir)), name="custom_avatars"))

    # Dedicated direct endpoints for avatars to guarantee 100% reliability
    @app.get("/gradio_api/avatars/{filename:path}", include_in_schema=False)
    @app.get("/avatars/{filename:path}", include_in_schema=False)
    def serve_avatar_image(filename: str):
        for candidate_dir in possible_avatar_paths:
            f = candidate_dir / filename
            if f.is_file():
                return FileResponse(f)
        raise HTTPException(status_code=404, detail="Avatar image not found")

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    @app.get("/gradio_api/favicon.svg", include_in_schema=False)
    @app.get("/gradio_api/favicon.ico", include_in_schema=False)
    def serve_favicon():
        fav = frontend_dist / "favicon.svg"
        if fav.exists():
            return FileResponse(fav)
        return HTMLResponse("", status_code=204)

    # Direct SPA routes for browser reload / direct links
    @app.get("/chat/{profile_id}", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/chat/{profile_id}/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/settings", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/settings/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/about", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/about/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/privacy", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/privacy/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/gradio_api/ui", response_class=HTMLResponse)
    @app.get("/gradio_api/ui/{full_path:path}", response_class=HTMLResponse)
    def serve_react_ui(full_path: str = "", profile_id: str = ""):
        run_zerogpu_task()
        idx = frontend_dist / "index.html"
        if idx.exists():
            return FileResponse(idx)
        return HTMLResponse("<h1>Spark React App Building... Please refresh</h1>")

    # Catch-all 404 handler for any HTML SPA page reload
    @app.exception_handler(404)
    async def spa_404_handler(request: Request, exc):
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            idx = frontend_dist / "index.html"
            if idx.is_file():
                return FileResponse(idx)
        return HTMLResponse("Not Found", status_code=404)

    return app

# Patch App.create_app so Gradio preserves our custom API routers and CORS on launch
_orig_create_app = App.create_app

@staticmethod
def _custom_create_app(blocks, *args, **kwargs):
    app = _orig_create_app(blocks, *args, **kwargs)
    return configure_fastapi_app(app)

App.create_app = _custom_create_app

# Also configure on demo.app directly for any pre-launch access
configure_fastapi_app(demo.app)

os.environ["ANONYMIZED_TELEMETRY"] = "False"

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(avatars_dir)] if avatars_dir.exists() else []
    )
