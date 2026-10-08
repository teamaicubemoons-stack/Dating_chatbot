import os
import sys
import asyncio
import logging

# Ensure backend modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import gradio as gr
from app.config.settings import settings
from app.api.dependencies import init_db, SessionLocal
from app.personas.persona_loader import load_all_personas, load_persona
from app.rag.vector_store import seed_persona_knowledge
from app.chat.chat_engine import process_chat_message
from app.models.db_models import LLMStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spark_gradio")

# Initialize SQLite database and seed vector store
try:
    init_db()
    seed_persona_knowledge()
    db = SessionLocal()
    if db.get(LLMStatus, 1) is None:
        db.add(LLMStatus(id=1, current_provider="groq", groq_available=True))
        db.commit()
    db.close()
    logger.info("Spark database and vector store initialized successfully.")
except Exception as e:
    logger.error(f"Initialization error: {e}")

# Load personas
personas = load_all_personas()
persona_map = {p.id: p for p in personas}
default_persona_id = personas[0].id if personas else "profile_1"

def get_avatar_path(avatar_url: str) -> str:
    filename = os.path.basename(avatar_url)
    local_path = os.path.join("avatars", filename)
    if os.path.exists(local_path):
        return local_path
    frontend_path = os.path.join("frontend", "public", "avatars", filename)
    if os.path.exists(frontend_path):
        return frontend_path
    return ""

def format_profile_html(p_id: str) -> str:
    p = persona_map.get(p_id)
    if not p:
        return "<p>Persona not found</p>"
    avatar_src = f"/file/{get_avatar_path(p.avatar_url)}"
    tags_html = "".join([f'<span class="spark-badge">{t}</span>' for t in p.personality_traits])
    return f"""
    <div class="spark-profile-card">
        <div class="spark-avatar-container">
            <img src="{avatar_src}" alt="{p.name}" class="spark-avatar-img" />
            <span class="spark-status-dot"></span>
        </div>
        <div class="spark-profile-info">
            <div class="spark-profile-header">
                <h2 class="spark-profile-name">{p.name}, {p.age}</h2>
                <span class="spark-location">📍 {p.city}</span>
            </div>
            <p class="spark-bio">{p.short_bio}</p>
            <div class="spark-tags">{tags_html}</div>
            <div class="spark-tone"><strong>Vibe:</strong> {p.tone}</div>
        </div>
    </div>
    """

def get_initial_greeting(p_id: str):
    p = persona_map.get(p_id)
    if p and p.initial_openers:
        return [({"role": "assistant", "content": p.initial_openers[0]})]
    return [({"role": "assistant", "content": "Hey! Nice to meet you 🙂"})]

# Custom CSS for dark glassmorphism Spark Dating UI
custom_css = """
/* Spark AI Dating App Theme */
:root {
    --spark-pink: #ec4899;
    --spark-purple: #8b5cf6;
    --spark-bg: #0f172a;
    --spark-card: rgba(30, 41, 59, 0.7);
    --spark-border: rgba(255, 255, 255, 0.1);
}

.gradio-container {
    background: radial-gradient(circle at top right, #1e1136, #0f172a 60%) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.spark-header {
    text-align: center;
    padding: 24px 0 16px;
    border-bottom: 1px solid var(--spark-border);
    margin-bottom: 24px;
}

.spark-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f472b6, #c084fc, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.spark-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 6px;
}

.spark-profile-card {
    display: flex;
    gap: 20px;
    background: var(--spark-card);
    border: 1px solid var(--spark-border);
    border-radius: 18px;
    padding: 20px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
    align-items: center;
}

.spark-avatar-container {
    position: relative;
    width: 90px;
    height: 90px;
    flex-shrink: 0;
}

.spark-avatar-img {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid var(--spark-pink);
    box-shadow: 0 0 16px rgba(236, 72, 153, 0.35);
}

.spark-status-dot {
    position: absolute;
    bottom: 4px;
    right: 4px;
    width: 16px;
    height: 16px;
    background-color: #22c55e;
    border: 2px solid #0f172a;
    border-radius: 50%;
}

.spark-profile-info {
    flex: 1;
}

.spark-profile-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
}

.spark-profile-name {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
}

.spark-location {
    color: #94a3b8;
    font-size: 0.9rem;
}

.spark-bio {
    color: #cbd5e1;
    font-size: 0.95rem;
    margin: 6px 0;
    line-height: 1.4;
}

.spark-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
}

.spark-badge {
    background: rgba(236, 72, 153, 0.15);
    color: #f472b6;
    border: 1px solid rgba(236, 72, 153, 0.3);
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
}

.spark-tone {
    font-size: 0.85rem;
    color: #94a3b8;
}

.spark-call-box {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(236, 72, 153, 0.4);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    margin-bottom: 16px;
    animation: pulseGlow 2s infinite ease-in-out;
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 10px rgba(236, 72, 153, 0.2); }
    50% { box-shadow: 0 0 25px rgba(236, 72, 153, 0.5); }
}

.call-btn {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

.hangup-btn {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}
"""

with gr.Blocks(css=custom_css, title="Spark — AI Dating & Companion", theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple")) as demo:
    gr.HTML("""
    <div class="spark-header">
        <h1 class="spark-title">✦ Spark Dating & Companion</h1>
        <p class="spark-subtitle">Multi-Persona AI Companion powered by Groq LLaMA 3.3 70B & Claude • Hinglish & English Conversationalist</p>
    </div>
    """)

    active_persona_state = gr.State(value=default_persona_id)
    user_session_state = gr.State(value="hf_user_session")
    call_active_state = gr.State(value=False)

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### 🌟 Choose Your Persona")
            persona_dropdown = gr.Radio(
                choices=[(f"{p.name} ({p.city})", p.id) for p in personas],
                value=default_persona_id,
                label="Available Personas",
                interactive=True
            )
            
            profile_display = gr.HTML(value=format_profile_html(default_persona_id))

            with gr.Row():
                call_button = gr.Button("📞 Call Persona", elem_classes=["call-btn"])
                hangup_button = gr.Button("🔴 End Call", elem_classes=["hangup-btn"], visible=False)
            
            call_status_banner = gr.HTML(value="", visible=False)

            gr.Markdown("---")
            gr.Markdown("#### 💡 Quick Starters")
            starter1 = gr.Button("Hey 🙂 How's your day going?")
            starter2 = gr.Button("What are you passionate about?")
            starter3 = gr.Button("What kind of music or vibe do you like?")

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                value=get_initial_greeting(default_persona_id),
                height=560,
                show_label=False,
                type="messages",
                bubble_full_width=False,
                avatar_images=(
                    None,
                    get_avatar_path(persona_map[default_persona_id].avatar_url) if default_persona_id in persona_map else None
                )
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message in Hinglish or English...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                send_button = gr.Button("Send 💌", scale=2, variant="primary")

            clear_button = gr.Button("🧹 Clear Conversation", size="sm")

    # Call simulation handlers
    def start_call(p_id):
        p = persona_map.get(p_id)
        name = p.name if p else "Persona"
        call_html = f"""
        <div class="spark-call-box">
            <h3 style="color: #22c55e; margin: 0 0 6px;">📞 Call Connected with {name}</h3>
            <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">Audio link active • Micro-synthesizer audio channel open</p>
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

    # Change persona handler
    def on_persona_change(selected_id):
        p = persona_map.get(selected_id)
        avatar = get_avatar_path(p.avatar_url) if p else None
        greeting = get_initial_greeting(selected_id)
        card_html = format_profile_html(selected_id)
        return (
            selected_id,
            card_html,
            gr.update(value=greeting, avatar_images=(None, avatar)),
            False,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value="", visible=False)
        )

    persona_dropdown.change(
        fn=on_persona_change,
        inputs=[persona_dropdown],
        outputs=[active_persona_state, profile_display, chatbot, call_active_state, call_button, hangup_button, call_status_banner]
    )

    # Chat interaction
    async def user_send_message(user_msg, history, p_id, user_session):
        if not user_msg or not user_msg.strip():
            return history, ""

        # Append user message
        new_history = list(history) + [{"role": "user", "content": user_msg}]

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

        new_history.append({"role": "assistant", "content": reply})
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

    def on_starter_click(starter_text, history, p_id, user_session):
        return asyncio.run(user_send_message(starter_text, history, p_id, user_session))

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
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["avatars", "frontend/public/avatars"])
