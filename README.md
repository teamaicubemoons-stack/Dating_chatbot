---
title: Spark Dating AI Chatbot
emoji: ✦
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# ✦ Spark — AI Companion Chatbot Platform

A production-grade, fully modular AI companion/chat platform. Browse distinct AI personas, each with a unique personality, backstory, and conversational style powered by Groq (primary) and Anthropic Claude (fallback).

---

## Architecture Overview

```
project-root/
├── backend/          # FastAPI + Python
│   └── app/
│       ├── config/       # Settings, logging
│       ├── llm/          # Groq + Claude clients, router
│       ├── personas/     # Plugin-style JSON persona configs
│       ├── rag/          # ChromaDB embeddings & retrieval
│       ├── memory/       # Conversation persistence
│       ├── chat/         # Orchestration engine + prompt builder
│       ├── api/          # FastAPI routes
│       ├── models/       # SQLAlchemy + Pydantic schemas
│       ├── scheduler/    # APScheduler health checks
│       └── utils/        # Token counting, error types
└── frontend/         # React + Vite
    └── src/
        ├── components/   # ProfileCard, ChatWindow, etc.
        ├── pages/        # HomePage, ChatPage, SettingsPage
        ├── context/      # ChatContext (global state)
        └── services/     # Axios API client
```

---

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm**
- A **Groq API key** (free tier works) — https://console.groq.com
- An **Anthropic API key** — https://console.anthropic.com

---

## Backend Setup

### 1. Navigate to backend folder
```bash
cd backend
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the embedding model (~80MB) on first run. This is a one-time operation.

### 4. Configure environment variables
```bash
copy .env.example .env   # Windows
# or
cp .env.example .env     # macOS/Linux
```

Edit `.env` and fill in your keys:
```env
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
CLAUDE_MODEL=claude-3-5-haiku-20241022
DATABASE_URL=sqlite:///./app.db
CHROMA_PERSIST_DIR=./chroma_data
```

### 5. Run the backend
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will:
- Auto-create the SQLite database and all tables
- Load and sync all persona configs from `app/personas/configs/`
- Seed persona knowledge into ChromaDB (first run downloads the embedding model)
- Start the APScheduler background job for Groq health checks
- Serve the API at `http://localhost:8000`

**API Docs:** http://localhost:8000/docs

---

## Frontend Setup

### 1. Navigate to frontend folder
```bash
cd frontend
```

### 2. Install dependencies
```bash
npm install
```

### 3. Configure environment
```bash
copy .env.example .env   # Windows
# or
cp .env.example .env     # macOS/Linux
```

The default `.env` is fine for local development:
```env
VITE_API_BASE_URL=http://localhost:8000
```

### 4. Run the frontend dev server
```bash
npm run dev
```

The app will open at `http://localhost:5173`

---

## Adding a New Persona (Plugin System)

To add a new AI companion, **only** create a new JSON file in `backend/app/personas/configs/`:

```json
{
  "id": "profile_4",
  "name": "Your Character Name",
  "age": 27,
  "short_bio": "One-liner shown on the profile card.",
  "avatar_url": "",
  "personality_traits": ["trait1", "trait2", "trait3"],
  "tone": "describe tone here",
  "backstory": "Detailed backstory...",
  "conversation_style_rules": [
    "Style rule 1",
    "Style rule 2"
  ],
  "system_prompt_template": "You are {name}, {age} years old. Personality: {personality_traits}. Tone: {tone}. Backstory: {backstory}. Style: {conversation_style_rules}. Never break character."
}
```

Restart the backend — the new persona is automatically loaded, synced to DB, and knowledge-seeded into ChromaDB. **Zero changes to any other file.**

---

## LLM Provider Routing

| State | Behaviour |
|---|---|
| Startup | Groq is the default provider |
| Groq quota exceeded | Instantly falls back to Claude (transparent to user) |
| While on Claude | APScheduler checks Groq every 12 hours |
| Groq quota reset | Automatically switches back to Groq |
| Both fail | HTTP 503 returned to frontend |

Check current provider status: `GET http://localhost:8000/health`

---

## Available Personas (Demo-Ready)

| ID | Name | Personality |
|---|---|---|
| `profile_1` | **Luna** | Adventurous travel photographer, Portland-raised, dry wit |
| `profile_2` | **Ethan** | Neuroscience PhD, nerdy-charming, self-deprecating |
| `profile_3` | **Zara** | Startup founder & underground DJ, bold and unfiltered |

---

## AI Companion Disclosure

As per industry standard practice, a clear AI companion disclosure is accessible via the **About** link in the top navigation or the Settings page. The chat interface itself maintains full immersion — the disclosure is never displayed within the chat UI.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/profiles` | List all personas (for homepage grid) |
| `GET` | `/profiles/{id}` | Full public persona detail |
| `POST` | `/chat/{profile_id}` | Send message → receive AI reply |
| `GET` | `/chat/{profile_id}/history` | Conversation history for a user |
| `GET` | `/health` | Internal LLM provider status |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| LLM Primary | Groq (LLaMA 3.3 70B) |
| LLM Fallback | Anthropic Claude (Haiku) |
| Vector DB | ChromaDB (local) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Database | SQLite via SQLAlchemy |
| Scheduling | APScheduler |
| Frontend | React 18 + Vite |
| HTTP Client | Axios |
| Routing | React Router v6 |
| Fonts | Inter + Outfit (Google Fonts) |
