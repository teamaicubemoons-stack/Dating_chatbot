import os
import sys
import logging
from pathlib import Path

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
logger = logging.getLogger("spark_app")

# Initialize database, personas, and ChromaDB vector store
try:
    init_db()
    db = SessionLocal()
    # Ensure LLMStatus exists
    if db.get(LLMStatus, 1) is None:
        db.add(LLMStatus(id=1, current_provider="openai", groq_available=False))
        db.commit()

    # Load personas and sync with Profile DB table
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

    # Start background health-check scheduler
    start_scheduler()
    logger.info("Application initialized successfully with %d personas.", len(personas))
except Exception as e:
    logger.error(f"Initialization warning: {e}", exc_info=True)

# Locate static build directories
possible_dist_paths = [
    base_dir / "frontend" / "dist",
    backend_dir / "static",
    base_dir / "static",
]
static_dir = next((p for p in possible_dist_paths if (p / "index.html").exists()), None)
assets_dir = (static_dir / "assets") if static_dir else None

possible_avatar_paths = [
    base_dir / "avatars",
    backend_dir / "static" / "avatars",
    base_dir / "frontend" / "public" / "avatars",
]
avatars_dir = next((p for p in possible_avatar_paths if p.exists()), None)

# Build Gradio demo (provides ZeroGPU runtime compatibility and fallback UI)
with gr.Blocks(title="Spark — AI Companions") as demo:
    gr.HTML(
        '<iframe id="react-wrapper" src="/" style="position:fixed;top:0;left:0;width:100vw;height:100vh;border:none;z-index:99999;" allow="autoplay; camera; microphone"></iframe>'
    )

# Enable CORS for API routes
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Register backend API routers
demo.app.include_router(routes_profiles.router)
demo.app.include_router(routes_chat.router)
demo.app.include_router(routes_health.router)
demo.app.include_router(routes_memories.router)

# 2. Prepend static file mounts & SPA routes
if assets_dir and assets_dir.exists():
    demo.app.router.routes.insert(
        0, Mount("/assets", StaticFiles(directory=str(assets_dir)), name="custom_assets")
    )

if avatars_dir and avatars_dir.exists():
    demo.app.router.routes.insert(
        0, Mount("/avatars", StaticFiles(directory=str(avatars_dir)), name="custom_avatars")
    )

def serve_index(request):
    if static_dir and (static_dir / "index.html").exists():
        return FileResponse(static_dir / "index.html")
    return HTMLResponse("<h1>Spark Backend Running</h1><p>Frontend assets not found.</p>")

def serve_favicon(request):
    if static_dir:
        fav = static_dir / "favicon.svg"
        if fav.exists():
            return FileResponse(fav)
    return HTMLResponse("", status_code=204)

demo.app.router.routes.insert(0, Route("/favicon.svg", serve_favicon, methods=["GET"]))
demo.app.router.routes.insert(0, Route("/chat/{profile_id}", serve_index, methods=["GET"]))
demo.app.router.routes.insert(0, Route("/settings", serve_index, methods=["GET"]))
demo.app.router.routes.insert(0, Route("/", serve_index, methods=["GET"]))

# 3. Client-side SPA fallback for any deep React Router links
@demo.app.exception_handler(404)
async def custom_404_handler(request, exc):
    path = request.url.path
    api_prefixes = ("/gradio_api", "/profiles", "/chat", "/health", "/memories", "/assets", "/avatars")
    if not any(path.startswith(prefix) for prefix in api_prefixes):
        if static_dir and (static_dir / "index.html").exists():
            return FileResponse(static_dir / "index.html")
    return HTMLResponse("Not Found", status_code=404)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[str(avatars_dir)] if avatars_dir else [],
    )
