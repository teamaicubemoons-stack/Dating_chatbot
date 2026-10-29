import os
import sys
import logging
import asyncio
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

# Ensure backend modules can be imported
base_dir = Path(__file__).resolve().parent
backend_dir = base_dir / "backend"
if str(backend_dir) in sys.path:
    sys.path.remove(str(backend_dir))
sys.path.insert(0, str(backend_dir))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import gradio as gr
from app.config.settings import settings
from app.api.dependencies import init_db, SessionLocal
from app.models.db_models import LLMStatus, Profile
from app.personas.persona_loader import load_all_personas
from app.rag.vector_store import seed_persona_knowledge
from app.chat.chat_engine import process_chat_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spark_dating_app")

# Load personas
personas = load_all_personas()
persona_map = {p["id"]: p for p in personas if isinstance(p, dict) and "id" in p}
default_persona_id = personas[0]["id"] if personas and isinstance(personas[0], dict) else "profile_1"

# Initialize SQLite database and seed vector store
try:
    init_db()
    db = SessionLocal()
    if db.get(LLMStatus, 1) is None:
        db.add(LLMStatus(id=1, current_provider="openai", groq_available=False))
    for p in personas:
        if isinstance(p, dict) and "id" in p:
            existing = db.get(Profile, p["id"])
            if existing is None:
                db.add(Profile(
                    id=p["id"],
                    name=p.get("name", "Companion"),
                    short_bio=p.get("short_bio", ""),
                    avatar_url=p.get("avatar_url", ""),
                ))
            seed_persona_knowledge(p["id"], p)
    db.commit()
    db.close()
    logger.info("Spark database and personas initialized successfully.")
except Exception as e:
    logger.error(f"Initialization error: {e}", exc_info=True)

# Helper to find avatar path
def get_avatar_path(avatar_url: str) -> str:
    filename = os.path.basename(avatar_url or "")
    for candidate in [
        os.path.join("avatars", filename),
        os.path.join("backend", "static", "avatars", filename),
        os.path.join("frontend", "public", "avatars", filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    return ""

# Auto-detect Chatbot message format (messages vs tuples) across Gradio versions
def detect_chatbot_format():
    try:
        gr.Chatbot(value=[{"role": "assistant", "content": "test"}])
        return "messages"
    except Exception:
        return "tuples"

CHATBOT_FORMAT = detect_chatbot_format()
logger.info(f"Detected Gradio Chatbot format: {CHATBOT_FORMAT}")

def get_initial_greeting(p_id: str):
    p = persona_map.get(p_id)
    greeting = "Hey! Nice to meet you 🙂"
    if p and p.get("initial_openers"):
        greeting = p["initial_openers"][0]
    
    if CHATBOT_FORMAT == "messages":
        return [{"role": "assistant", "content": greeting}]
    return [[None, greeting]]

def format_profile_header_html(p_id: str) -> str:
    p = persona_map.get(p_id)
    if not p:
        return "<p>Companion not found</p>"
    avatar_src = f"/gradio_api/file={get_avatar_path(p.get('avatar_url', ''))}"
    tags_html = "".join([f'<span class="spark-badge">{t}</span>' for t in p.get("personality_traits", [])])
    return f"""
    <div class="spark-profile-header-card">
        <div class="spark-avatar-container">
            <img src="{avatar_src}" alt="{p.get('name', '')}" class="spark-avatar-img" />
            <span class="spark-status-dot"></span>
        </div>
        <div class="spark-profile-info">
            <div class="spark-profile-topline">
                <h2 class="spark-profile-name">{p.get('name', '')}, {p.get('age', '')}</h2>
                <span class="spark-verified-badge">✓ Verified</span>
                <span class="spark-online-text">🟢 Active now</span>
            </div>
            <p class="spark-bio">{p.get('short_bio', '')}</p>
            <div class="spark-tags-row">
                <span class="spark-location-chip">📍 {p.get('city', '')}, {p.get('country', '')}</span>
                {tags_html}
            </div>
        </div>
    </div>
    """

def format_matches_stories_html() -> str:
    items = []
    for p in personas:
        avatar_src = f"/gradio_api/file={get_avatar_path(p.get('avatar_url', ''))}"
        name = p.get("name", "").split()[0]
        items.append(f"""
        <div class="spark-story-item">
            <div class="spark-story-avatar-wrap">
                <img src="{avatar_src}" alt="{name}" class="spark-story-img" />
                <span class="spark-story-dot"></span>
            </div>
            <span class="spark-story-name">{name}</span>
        </div>
        """)
    return f"""
    <div class="spark-stories-section">
        <div class="spark-stories-label">✨ Matches</div>
        <div class="spark-stories-track">
            {"".join(items)}
        </div>
    </div>
    """

# Custom CSS matching the React Spark Dating UI theme
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

/* Design Tokens */
:root {
    --spark-violet: #9b35d4;
    --spark-pink: #d63f7a;
    --spark-coral: #e8573a;
    --spark-brand-gradient: linear-gradient(135deg, #9b35d4 0%, #d63f7a 55%, #e8573a 100%);
    --spark-bg-page: linear-gradient(160deg, #fdf5ff 0%, #fff0f7 50%, #fff6f0 100%);
    --spark-card: #ffffff;
    --spark-border: #ede9f5;
    --spark-text-primary: #1a1028;
    --spark-text-secondary: #5a4e72;
    --spark-online: #00c98a;
}

body, .gradio-container {
    background: var(--spark-bg-page) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 1260px !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
}

/* Header */
.spark-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid var(--spark-border);
    border-radius: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(155, 53, 212, 0.05);
}

.spark-brand-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: var(--spark-brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.spark-badge-status {
    background: rgba(0, 201, 138, 0.1);
    color: var(--spark-online);
    border: 1px solid rgba(0, 201, 138, 0.3);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Stories / Matches Bar */
.spark-stories-section {
    background: #ffffff;
    border: 1px solid var(--spark-border);
    border-radius: 18px;
    padding: 12px 16px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(155, 53, 212, 0.03);
}

.spark-stories-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--spark-violet);
    margin-bottom: 8px;
}

.spark-stories-track {
    display: flex;
    gap: 14px;
    overflow-x: auto;
    padding-bottom: 4px;
}

.spark-story-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    flex-shrink: 0;
}

.spark-story-avatar-wrap {
    position: relative;
    width: 54px;
    height: 54px;
    padding: 2px;
    border-radius: 50%;
    background: var(--spark-brand-gradient);
}

.spark-story-img {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #ffffff;
}

.spark-story-dot {
    position: absolute;
    bottom: 2px;
    right: 2px;
    width: 12px;
    height: 12px;
    background-color: var(--spark-online);
    border: 2px solid #ffffff;
    border-radius: 50%;
}

.spark-story-name {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--spark-text-primary);
    margin-top: 4px;
}

/* Active Companion Header Card */
.spark-profile-header-card {
    display: flex;
    align-items: center;
    gap: 18px;
    background: #ffffff;
    border: 1px solid var(--spark-border);
    border-radius: 20px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 18px rgba(155, 53, 212, 0.05);
}

.spark-avatar-container {
    position: relative;
    width: 72px;
    height: 72px;
    flex-shrink: 0;
}

.spark-avatar-img {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid var(--spark-pink);
    box-shadow: 0 0 12px rgba(214, 63, 122, 0.25);
}

.spark-status-dot {
    position: absolute;
    bottom: 2px;
    right: 2px;
    width: 14px;
    height: 14px;
    background-color: var(--spark-online);
    border: 2px solid #ffffff;
    border-radius: 50%;
}

.spark-profile-info {
    flex: 1;
}

.spark-profile-topline {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.spark-profile-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--spark-text-primary);
    margin: 0;
}

.spark-verified-badge {
    background: #00a8e8;
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
}

.spark-online-text {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--spark-online);
}

.spark-bio {
    color: var(--spark-text-secondary);
    font-size: 0.9rem;
    margin: 4px 0 8px;
    line-height: 1.35;
}

.spark-tags-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
}

.spark-location-chip {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--spark-violet);
    background: rgba(155, 53, 212, 0.08);
    padding: 3px 9px;
    border-radius: 12px;
}

.spark-badge {
    background: #f4effa;
    color: var(--spark-text-secondary);
    border: 1px solid var(--spark-border);
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* Gradio Chatbot Message Bubbles Styling */
.gradio-container .message-wrap, .gradio-container .bubble-wrap {
    gap: 10px !important;
}

/* User Message */
.gradio-container .user, 
.gradio-container [data-testid="user"], 
.gradio-container .message.user {
    background: var(--spark-brand-gradient) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 14px rgba(214, 63, 122, 0.2) !important;
    font-size: 0.95rem !important;
    border: none !important;
}

.gradio-container .user p, 
.gradio-container [data-testid="user"] p,
.gradio-container .message.user p {
    color: #ffffff !important;
}

/* Bot Message */
.gradio-container .bot, 
.gradio-container [data-testid="bot"], 
.gradio-container .message.bot {
    background: #ffffff !important;
    color: var(--spark-text-primary) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 18px !important;
    box-shadow: 0 4px 16px rgba(155, 53, 212, 0.06) !important;
    border: 1px solid var(--spark-border) !important;
    font-size: 0.95rem !important;
}

.gradio-container .bot p,
.gradio-container [data-testid="bot"] p,
.gradio-container .message.bot p {
    color: var(--spark-text-primary) !important;
}

/* Radio Persona Switcher */
.gradio-container .radio-group {
    background: #ffffff !important;
    border: 1px solid var(--spark-border) !important;
    border-radius: 16px !important;
    padding: 10px !important;
}

/* Buttons */
.spark-send-btn {
    background: var(--spark-brand-gradient) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 999px !important;
    box-shadow: 0 4px 14px rgba(214, 63, 122, 0.3) !important;
}

.spark-starter-chip {
    background: #ffffff !important;
    color: var(--spark-violet) !important;
    border: 1px solid var(--spark-border) !important;
    border-radius: 999px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.2s ease !important;
}
.spark-starter-chip:hover {
    border-color: var(--spark-violet) !important;
    background: #faf7fd !important;
}

.spark-clear-btn {
    background: transparent !important;
    color: var(--spark-text-secondary) !important;
    border: 1px solid var(--spark-border) !important;
    border-radius: 12px !important;
    font-size: 0.8rem !important;
}

/* Audio Call Banner */
.spark-call-box {
    background: rgba(0, 201, 138, 0.08);
    border: 1px solid rgba(0, 201, 138, 0.3);
    border-radius: 16px;
    padding: 12px;
    text-align: center;
    margin-bottom: 12px;
}
"""

with gr.Blocks(title="Spark — AI Dating & Companions") as demo:
    # Inject styling
    gr.HTML(f"<style>{custom_css}</style>")

    # Topbar
    gr.HTML("""
    <div class="spark-topbar">
        <h1 class="spark-brand-title">✦ Spark Dating & Companion</h1>
        <div class="spark-badge-status">🟢 100% Human Immersion • Zero AI Leaks</div>
    </div>
    """)

    active_persona_state = gr.State(value=default_persona_id)
    user_session_state = gr.State(value="hf_user_session")
    call_active_state = gr.State(value=False)

    with gr.Row():
        # LEFT COLUMN: Sidebar & Matches
        with gr.Column(scale=4):
            # Stories / Matches Avatar Carousel
            gr.HTML(value=format_matches_stories_html())

            gr.Markdown("#### 💬 Companions")
            persona_radio = gr.Radio(
                choices=[(f"{p.get('name', '')} ({p.get('city', '')})", p.get('id', '')) for p in personas if isinstance(p, dict)],
                value=default_persona_id,
                label="Choose Conversation",
                interactive=True
            )

            gr.Markdown("---")
            gr.Markdown("#### 💡 Quick Conversation Starters")
            starter1 = gr.Button("Hey 🙂 How's your day going?", elem_classes=["spark-starter-chip"])
            starter2 = gr.Button("What are you passionate about?", elem_classes=["spark-starter-chip"])
            starter3 = gr.Button("What kind of music or vibe do you like?", elem_classes=["spark-starter-chip"])

            gr.Markdown("---")
            with gr.Row():
                call_button = gr.Button("📞 Voice Call", variant="secondary")
                hangup_button = gr.Button("🔴 Hang Up", variant="stop", visible=False)

            call_status_banner = gr.HTML(value="", visible=False)
            clear_button = gr.Button("🧹 Clear Chat", elem_classes=["spark-clear-btn"], size="sm")

        # RIGHT COLUMN: Chat Area
        with gr.Column(scale=8):
            # Active Companion Header Card
            profile_display = gr.HTML(value=format_profile_header_html(default_persona_id))

            # Chatbot
            default_avatar = get_avatar_path(persona_map[default_persona_id].get("avatar_url", "")) if default_persona_id in persona_map else None
            chatbot = gr.Chatbot(
                value=get_initial_greeting(default_persona_id),
                height=560,
                show_label=False,
                avatar_images=(None, default_avatar)
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message in Hinglish or English...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                send_button = gr.Button("Send 💌", scale=2, elem_classes=["spark-send-btn"])

    # 1. Voice Call Handlers
    def start_call(p_id):
        p = persona_map.get(p_id)
        name = p.get("name", "Companion") if p else "Companion"
        call_html = f"""
        <div class="spark-call-box">
            <h4 style="color: #00c98a; margin: 0 0 4px;">📞 Voice Call Connected with {name}</h4>
            <p style="color: #5a4e72; margin: 0; font-size: 0.82rem;">Audio channel live • Natural voice conversational flow</p>
        </div>
        """
        return (
            True,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=call_html, visible=True)
        )

    def stop_call():
        return (
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False)
        )

    call_button.click(
        fn=start_call,
        inputs=[active_persona_state],
        outputs=[call_active_state, call_button, hangup_button, call_status_banner]
    )

    hangup_button.click(
        fn=stop_call,
        inputs=[],
        outputs=[call_active_state, call_button, hangup_button, call_status_banner]
    )

    # 2. Switch Persona
    def on_persona_change(selected_id):
        p = persona_map.get(selected_id)
        avatar = get_avatar_path(p.get("avatar_url", "")) if p else None
        greeting = get_initial_greeting(selected_id)
        card_html = format_profile_header_html(selected_id)
        return (
            selected_id,
            card_html,
            gr.update(value=greeting, avatar_images=(None, avatar)),
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False)
        )

    persona_radio.change(
        fn=on_persona_change,
        inputs=[persona_radio],
        outputs=[active_persona_state, profile_display, chatbot, call_active_state, call_button, hangup_button, call_status_banner]
    )

    # 3. Message interaction
    async def user_send_message(user_msg, history, p_id, user_session):
        if not user_msg or not user_msg.strip():
            return history, ""

        # Trigger ZeroGPU hook for HF Spaces runtime
        run_zerogpu_task(user_msg)

        try:
            db = SessionLocal()
            reply = await process_chat_message(
                db=db,
                profile_id=p_id,
                user_id=user_session,
                user_message=user_msg
            )
            db.close()
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            reply = "I'm having a little trouble connecting right now, let's chat in just a moment! 🙂"

        if CHATBOT_FORMAT == "messages":
            new_history = list(history or []) + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply}
            ]
        else:
            new_history = list(history or []) + [[user_msg, reply]]

        return new_history, ""

    msg_input.submit(
        fn=user_send_message,
        inputs=[msg_input, chatbot, active_persona_state, user_session_state],
        outputs=[chatbot, msg_input]
    )

    send_button.click(
        fn=user_send_message,
        inputs=[msg_input, chatbot, active_persona_state, user_session_state],
        outputs=[chatbot, msg_input]
    )

    # Starters
    starter1.click(
        fn=lambda h, p, u: asyncio.run(user_send_message("Hey 🙂 How's your day going?", h, p, u)),
        inputs=[chatbot, active_persona_state, user_session_state],
        outputs=[chatbot, msg_input]
    )

    starter2.click(
        fn=lambda h, p, u: asyncio.run(user_send_message("What are you passionate about?", h, p, u)),
        inputs=[chatbot, active_persona_state, user_session_state],
        outputs=[chatbot, msg_input]
    )

    starter3.click(
        fn=lambda h, p, u: asyncio.run(user_send_message("What kind of music or vibe do you like?", h, p, u)),
        inputs=[chatbot, active_persona_state, user_session_state],
        outputs=[chatbot, msg_input]
    )

    clear_button.click(
        fn=lambda p_id: get_initial_greeting(p_id),
        inputs=[active_persona_state],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=["avatars", "frontend/public/avatars", "backend/static/avatars"]
    )
