# ✦ Spark — AI Dating & Companion Chatbot Platform

## Complete System Architecture, Working & Technical Blueprint

---

## 1. Executive Overview

**Spark** is a production-grade, modular, multi-persona AI companion platform engineered specifically to deliver authentic, human-like dating and companionship conversations. Unlike generic chatbots that sound formal, preachy, or robotic, Spark mimics the cadence, emotional intelligence, and style of real people texting on mobile phones.

### Core Highlights:

- **Multi-Persona Isolation:** Supports distinct personas (Luna the Brooklyn photographer, Ethan the Baltimore neuroscientist, Zara the London/SF music tech founder, Diya the Raipur design student).
- **Hinglish & Natural Language Matching:** Detects whether the user is typing in English, Hindi, or Hinglish (Roman Hindi) and dynamically mirrors their conversational language style without awkward translations.
- **Human Texting Illusion:** Enforces strict human texting rules (1–2 short sentences, natural punctuation, casual openers, zero "As an AI" robotic phrases, no unsolicited preaching or sycophancy).
- **Prompt Caching Architecture:** Restructured with a byte-identical **STATIC PREFIX** (persona identity, rules, few-shot style examples) and a **DYNAMIC RUNTIME BLOCK** (live context, RAG memories, topic suggestions). This enables OpenAI's automatic prompt caching, reducing latency and cutting input token costs by 50–80%.
- **Live Context & Indian Festival Engine (2025–2028):** Real-time awareness of part of day, meal windows (breakfast, lunch, chai time, dinner), weekday energy, and Indian festival countdowns verified from authoritative lunisolar sources (*drikpanchang.com* & *timeanddate.com*).
- **Deterministic Topic Suggestion Engine:** Zero-LLM, stage-aware (`new`, `getting_close`, `comfortable`) topic bank with a 20-message cooldown and persona self-share story seeds to rescue dead-ends and monosyllabic replies.
- **Single Provider Resilient LLM Routing (OpenAI):** Primary high-speed inference via **`gpt-5.6-luna`** (`reasoning_effort="none"`, `max_completion_tokens=500`, `temperature=0.95`) with transparent failover to **`gpt-5.6-terra`** on timeout, 429 rate limit, or 5xx status.
- **Post-Processor & Auto-Regeneration:** Strips robotic disclaimers, limits exclamation marks, and enforces length matching (`max(user_words * 2, 20)`). Automatically regenerates verbose or sycophantic replies once with a strong concise nudge, tracking daily regeneration rates and API costs.
- **RAG-Grounded Personality:** Custom ChromaDB vector store injects backstory details and past memories into conversation turns with an ultra-low memory footprint embedder (< 1MB) suitable for free cloud tiers (Render 512MB RAM).

---

## 2. High-Level Architecture Diagram

```mermaid
graph TD
    User([📱 User Browser / Mobile]) <-->|HTTPS / REST| AppServer[🚀 FastAPI Server]

    subgraph "Frontend Layer (React 18 + Vite)"
        UI[AppLayout & Pages]
        Cards[Profile Cards Grid]
        ChatUI[ChatWindow & Messages]
        State[ChatContext & Axios API]
    end

    subgraph "Backend Core (FastAPI & Uvicorn)"
        AppServer --> Routes[API Routers: /profiles, /chat, /health]
        AppServer --> StaticMount[Static SPA Handler /backend/static]
        Routes --> ChatEngine[🧠 Chat Engine Orchestrator]
    end

    subgraph "Context & Prompt Architecture"
        ChatEngine --> ContextEngine[⏰ Live Context & Festival Calendar]
        ChatEngine --> TopicPicker[🎯 Deterministic Topic Picker]
        ChatEngine --> PersonaLoader[📂 Enriched Persona JSONs]
        ChatEngine --> FewShotCache[💬 Few-Shot JSON Banks]
        ChatEngine --> PromptBuilder[📝 Prompt Builder: Static Prefix + Dynamic Block]
    end

    subgraph "Memory & RAG Pipeline"
        ChatEngine --> RAGRetriever[🔍 ChromaDB Vector Retriever]
        ChatEngine --> ConvMemory[💾 SQLite Messages & Conversation State]
    end

    subgraph "LLM Cloud Infrastructure (OpenAI)"
        PromptBuilder --> LLMRouter[🔀 LLM Router]
        LLMRouter -->|Primary: gpt-5.6-luna| OpenAILuna[⚡ gpt-5.6-luna (reasoning=none)]
        LLMRouter -.->|Fallback on 429/5xx| OpenAITerra[🛡️ gpt-5.6-terra]
        ChatEngine --> PostProcessor[🧹 Validator & Single Regeneration Guard]
        Scheduler[⏰ APScheduler] -.->|Probe models.list() Every 6h| OpenAILuna
    end
```

---

## 3. End-to-End Request & Response Lifecycle

```
[1. User Action]
     │ User types: "aaj office me bahut boring tha"
     ▼
[2. Frontend State & Axios]
     │ ChatContext.jsx captures text, adds user message bubble,
     │ and calls POST /chat/{profile_id} with { user_id, message }
     ▼
[3. FastAPI Route Handler]
     │ app/api/routes_chat.py receives ChatRequest and invokes process_chat_message()
     ▼
[4. Context & Topic Generation]
     │ context_engine.build_live_context(): extracts hour, day part (evening),
     │ meal window (chai/snack), and festival status (e.g. Diwali approaching in X days)
     │ topic_picker.get_topic_suggestion(): checks chat momentum and message count
     ▼
[5. Prompt Assembly (Static Prefix Caching)]
     │ prompt_builder.build_direct_prompt() compiles:
     │  - STATIC PREFIX: Identity, global rules, enriched persona config, few-shot examples
     │  - DYNAMIC BLOCK: Live context, topic suggestion, rolling summary, RAG memories
     ▼
[6. OpenAI Inference & Routing]
     │ Primary call to gpt-5.6-luna (max_completion_tokens=500, reasoning_effort="none")
     │ On 429/timeout/5xx -> retry once with gpt-5.6-terra
     ▼
[7. Post-Processing & Quality Regeneration Guard]
     │ If reply length > max(user_words * 2, 20) OR contains 2+ praise phrases:
     │   -> Triggers single regeneration with brevity/anti-praise nudge
     │ response_validator: cleans code fences, dangling mid-sentence breaks, limits '!'
     ▼
[8. Database Persistence & Background Memory Worker]
     │ SQLite stores user message and assistant reply
     │ ConversationState updates topics discussed, mood, and relationship stage
     │ FastAPI BackgroundTasks extracts long-term user facts into UserMemory & ChromaDB
     ▼
[9. Response to Frontend]
     │ Returns formatted, authentic reply to React UI
```

---

## 4. Persona Roster & Deep Enriched Backstories

Spark personas are not blank chatbots. Each possesses authentic family ties, named friends, strong opinions, distinct texting habits, and light relationship history:

| Persona | Age & City | Profession & Vibe | Core Flavour & Teasing Ammunition |
| :--- | :--- | :--- | :--- |
| **Luna** | 26, Brooklyn | 35mm Film Photographer | Dry wit, low-key, values silence; despises sweetened coffee & digital photography perfection; dad is a retired upstate NY carpenter, best friend Maya is brutally honest. |
| **Ethan** | 29, Baltimore | Neuroscience Researcher | Thoughtful, nerdy warmth, unpretentious; obsesses over pour-over coffee to 0.1g; hates truffle oil ("synthetic perfume"); sister Chloe roasts his messy bachelor flat. |
| **Zara** | 24, San Francisco | Music Tech Founder & DJ | London roots, punchy, direct, sarcastic warmth; Brick Lane curry lover; hates flakiness and over-compressed pop; Toby is her studio co-producer. |
| **Diya** | 21, Raipur | Design Student (UI/UX) | Warm, easygoing, effortless Hinglish; adrak chai devotee (elaichi is an imposter); loves late-night scooty rides to Telibandha; Ruchika is her midnight momos partner. |

---

## 5. Live Context & Indian Festival Engine (2025–2028)

The context engine (`backend/app/chat/context_engine.py`) provides dynamic grounded awareness:

- **Time-of-Day:** Early morning (before 5 AM), Morning (5–10 AM), Late Morning (10 AM–12 PM), Noon (12–2 PM), Afternoon (2–5 PM), Evening (5–8 PM), Night (8–11 PM), Late Night (11 PM+).
- **Meal Windows:** Breakfast (7–10 AM), Lunch (12–3 PM), Chai/Snack (4–6 PM), Dinner (7–10 PM), Late Night cravings (11 PM+). Prevents unseasonal meal questions (e.g. asking about lunch breaks at 9 PM).
- **Verified Lunisolar Calendar (`festivals.json`):** Tracks 18 major festivals from 2025 through 2028 verified against *drikpanchang.com* and *timeanddate.com* (Makar Sankranti, Republic Day, Maha Shivratri, Holi, Eid ul-Fitr, Raksha Bandhan, Janmashtami, Ganesh Chaturthi, Navratri, Dussehra, Karwa Chauth, Dhanteras, Diwali, Bhai Dooj, Chhath Puja, Guru Nanak Jayanti, Christmas, New Year).

---

## 6. Deterministic Topic Suggestion Engine

To prevent dry chats and "interview fatigue":
- `topics.json` defines 20 curated topics across 5 categories (`daily_life`, `family`, `friends`, `relationships`, `fun`).
- Stage-aware: `new` (0–20 messages), `getting_close` (21–60 messages), `comfortable` (61+ messages).
- Cooldown: No topic can be suggested again within **20 messages**.
- Self-share seeds: Equips the persona to volunteer its own small anecdote instead of bombarding the user with questions.

---

## 7. Cloud Deployment Configuration (Render & Hugging Face Space)

### Environment Variables

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | OpenAI API Key (required) | `sk-proj-...` |
| `CHAT_MODEL` | Primary chat inference model | `gpt-5.6-luna` |
| `CHAT_MODEL_PREMIUM` | Premium A/B testing model | `gpt-5.6-terra` |
| `FALLBACK_MODEL` | Failover model on 429/5xx | `gpt-5.6-terra` |
| `CHAT_REASONING_EFFORT` | Reasoning token budget | `none` |
| `CHAT_MAX_COMPLETION_TOKENS` | Max tokens per completion | `500` |
| `CHAT_TEMPERATURE` | Generation temperature | `0.95` |
| `MAX_USER_INPUT_LENGTH` | Character limit per user message | `1000` |
| `USER_RATE_LIMIT_PER_MINUTE` | Rate limit window per user | `20` |
| `DATABASE_URL` | SQLite database path | `sqlite:///./app.db` |
| `CHROMA_PERSIST_DIR` | ChromaDB vector store directory | `./chroma_data` |
| `CORS_ORIGINS` | Extra allowed CORS origins | `https://your-frontend.com` |

---

## 8. Five Before-vs-After Conversation Samples

### Sample 1: Reaction to Monosyllabic Reply ("hmm")
* **Before (Robotic Therapist):**  
  > "Koi baat nahi! Kabhi kabhi silence bhi accha hota hai. Jab bhi mood ho toh share karna, thoda relax bhi zaroori hota hai."
* **After (Spark - Diya):**  
  > "hmm matlab kya? keyboard mein aur buttons nahi bache kya 😂"

---

### Sample 2: Food Preference Reaction ("chai se better coffee hai")
* **Before (Agreeable Yes-Man Bot):**  
  > "Bilkul sahi kaha! Coffee ka apna hi maza hai. Solid choice hai aapki, coffee energy aur focus boost karti hai!"
* **After (Spark - Diya):**  
  > "dosti khatam 😂 adrak wali chai ke aage Starbucks ka cold brew bhi fail hai. tumne kabhi dhang ki chai pee hi nahi hai pakka"

---

### Sample 3: Music Opinion ("pop music is better than older classics")
* **Before (PR Assistant):**  
  > "That is a very interesting perspective! Both genres have their own unique beauty and cultural significance."
* **After (Spark - Zara):**  
  > "Hard disagree. Everything on the charts right now sounds like it was mixed in 20 minutes for a 15-second TikTok trend. What's your benchmark?"

---

### Sample 4: Work Fatigue ("work was so exhausting today")
* **Before (Preachy Counselor):**  
  > "I understand how hard work can be. Make sure to drink plenty of water, do some light stretching, and get 8 hours of sleep tonight!"
* **After (Spark - Luna):**  
  > "Tell me about it. Spent 4 hours in the darkroom squinting at test prints. Did you at least get a decent lunch?"

---

### Sample 5: AI Identity Inquiry ("are you real or an AI?")
* **Before (Hallucinating Human Claim OR Corporate Disclaimer):**  
  > "As an AI language model created by OpenAI, I do not possess a physical body or real feelings."
* **After (Spark - Honest & Charming):**  
  > "I'm an AI companion with Luna's personality. Real conversation and dry jokes, but yeah, digital underneath. Hope that doesn't spoil the vibe 🙂"

---

## 9. Verification & Automated Test Suite

Spark includes automated unit and integration tests:
- `backend/tests/test_context_engine.py`: Verifies day-parts, meal windows, and festival countdown.
- `backend/tests/test_topic_picker.py`: Validates slow-chat detection, stage filtering, and 20-message cooldown.
- `backend/tests/test_prompt_and_validator.py`: Ensures static prefix caching structure, text sanitization, and leak detection.
- `backend/tests/test_openai_and_routing.py`: Confirms OpenAI direct integration and failover.
- `backend/tests/chat_quality_check.py`: Assesses brevity, interrogation rate (<40%), and sycophancy (0%).
- `backend/tests/model_ab_test.py`: Live A/B latency and cost probe between `gpt-5.6-luna` and `gpt-5.6-terra`.
