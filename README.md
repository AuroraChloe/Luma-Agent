<div align="center">

**English** | [简体中文](README_CN.md)

# ✦ Luma

### A self-hosted AI automation service for conversations, tools, media, and production workflows.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#-quick-start)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](#-architecture)
[![LangChain](https://img.shields.io/badge/LangChain-Agent%20Runtime-1C3C3C)](#-agent-runtime)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)](#-architecture)
[![Redis](https://img.shields.io/badge/Redis-Jobs%20%26%20SSE-DC382D?logo=redis&logoColor=white)](#-architecture)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](#-quick-start)

</div>

> **Luma** turns a natural-language request into an observable, tool-driven workflow. It supports ordinary chat, structured tool calling, vision-aware conversations, image creation, RAG, codebase analysis, MCP services, speech, and storyboard-guided short-video production.

## ✨ What Luma Can Do

| | Capability | What it handles |
| :--: | --- | --- |
| 🧠 | **Agent runtime** | Dynamic tool selection, multi-step execution, context-aware follow-ups, and safe final responses |
| 🔎 | **Research & MCP** | Web search, weather, remote MCP services, and business workflows exposed as tools |
| 👁️ | **Vision & images** | Image understanding, asset selection, image generation, and image editing |
| 🎬 | **Video workflow** | Concept planning, subject and scene references, multi-panel storyboards, visual review, and storyboard-guided video generation |
| 📚 | **RAG** | Document ingestion, semantic chunking, vector retrieval, focused knowledge-base answers |
| 💻 | **Coding analysis** | Upload a codebase for read-only structure, flow, risk, and implementation analysis |
| 🎙️ | **Speech** | Real-time ASR over WebSocket and TTS for final text responses |

## 🗺️ Architecture

```mermaid
flowchart LR
    Client[Client / App] -->|Bearer key| API[FastAPI service]
    API --> Runtime[Luma Agent Runtime]
    API --> Queue[Redis jobs + SSE]
    API --> DB[(PostgreSQL + pgvector)]
    API --> Media[Local media storage]

    Runtime --> LLM[OpenAI-compatible LLM]
    Runtime --> Tools[Local tools + MCP]
    Runtime --> Vision[Vision provider]
    Runtime --> Image[Image provider]
    Runtime --> Video[Video gateway]
    Runtime --> Speech[ASR / TTS provider]
```

**Durable state** lives in PostgreSQL. **Live execution**, cancellation, and server-sent events live in Redis. Media stays local under `./data/media`, so the project can run without S3 or a CDN.

## 🚀 Quick Start

### 1. Prepare configuration

```bash
git clone https://github.com/<your-account>/Luma-Agent.git
cd Luma-Agent

cp .env.example .env
cp backend/.env.example backend/.env
```

Set matching database values in both files, then add at least these settings to `backend/.env`:

```dotenv
LUMA_API_KEY=replace-with-a-long-random-service-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=replace-with-your-provider-key
AGENT_MODEL=your-agent-model
```

### 2. Start the stack

```bash
docker compose up --build -d
curl http://localhost:8001/healthz
```

Expected response:

```json
{"status":"ok"}
```

### 3. Create a conversation

```bash
curl -X POST http://localhost:8001/v1/sessions \
  -H "Authorization: Bearer $LUMA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"demo","title":"My first Luma session"}'
```

> 💡 Port `8001` already occupied? Set `LUMA_PORT` in the root `.env`. When using a reverse proxy, update `PUBLIC_API_BASE`, `PUBLIC_IMAGE_BASE_URL`, and `PUBLIC_MEDIA_BASE_URL` in `backend/.env` too.

## 🧠 Agent Runtime

Luma does not hard-code a fixed tool sequence. For every request it:

1. **Builds context** from the active conversation, relevant visual assets, project state, and optional Skill guidance.
2. **Selects candidates** with a lightweight tool selector that reads recent history, not only the last user message.
3. **Runs the agent loop** with the selected local tools and/or an MCP service domain.
4. **Feeds tool observations back** to the model until it has enough information to answer or complete the workflow.
5. **Persists outcomes** to PostgreSQL and streams live job state over Redis-backed SSE.

This keeps ordinary Q&A lightweight while allowing multi-tool work such as **research → infographic**, **storyboard → video**, or **image → vision → edit**.

## 🔌 API Surface

| Area | Endpoints |
| --- | --- |
| ❤️ Health | `GET /healthz` |
| 💬 OpenAI compatibility | `POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/models` |
| ⚡ Agent runs | `POST /v1/agent/runs`, run state, SSE events, cancellation |
| 🗂️ Sessions | Create, list, update, delete, message history, regeneration, and latest job state under `/v1/sessions` |
| 🖼️ Assets | `POST /v1/assets`, `POST /v1/images/generations`, `POST /v1/images/edits`, `GET /temp_file/{name}` |
| 📚 RAG | File upload, collections, document state, job SSE, `POST /v1/rag/search` |
| 💻 Coding | `POST /v1/coding/workspaces` |
| 🎙️ Audio | `WS /v1/audio/asr`, `POST /v1/audio/tts` |
| 🎬 Video | `GET /v1/sessions/{id}/video-generation` |

All endpoints except `/healthz` use:

```http
Authorization: Bearer <LUMA_API_KEY>
X-Luma-Project-ID: my-project
```

`X-Luma-Project-ID` is optional and defaults to `default`. It isolates sessions, RAG collections, assets, code workspaces, and video projects inside one deployment.

## 🧩 Configure Only What You Need

Luma's integrations are independent. Copy `backend/.env.example`, then enable a capability by filling its required settings. An unset integration is simply unavailable to the Agent; it does not break regular chat.

| Capability | Required settings | How it becomes available |
| --- | --- | --- |
| 💬 Chat + Agent | `LLM_BASE_URL`, `LLM_API_KEY`, `AGENT_MODEL` | Always available after chat configuration is valid |
| 🧠 Tool-driven Agent | `LLM_SUPPORTS_TOOLS=true` | Local tools/MCP domains are discovered automatically; set `AGENT_TOOL_SELECTOR_MODEL` only when the selector should use a smaller dedicated model |
| 👁️ Vision | `VISION_LLM_BASE_URL`, `VISION_LLM_MODEL`, `VISION_LLM_API_KEY` or `DASHSCOPE_API_KEY` | Automatically used when the request requires reading an image |
| 🎨 Images | `IMAGE_PROVIDER`, `IMAGE_BASE_URL`, `IMAGE_API_KEY`, `IMAGE_MODEL` | Image generation/editing tools become usable after a compatible provider is configured |
| 📚 RAG | `RAG_EMBEDDING_BASE_URL`, `RAG_EMBEDDING_MODEL`, `NVIDIA_API_KEY` or `LLM_API_KEY` | Upload files through `/v1/rag/files`; pgvector is included in the Compose database image |
| 🎙️ ASR / TTS | `DASHSCOPE_API_KEY` | ASR uses `/v1/audio/asr`; TTS uses `/v1/audio/tts` |
| 🔎 Web search | `TAVILY_API_KEY` | The search tool is automatically offered when a request needs current web information |
| 🌤️ Weather | `WEATHER_API_KEY` | The weather tool is automatically offered for location-based forecasts |
| 🔌 MCP | `MCP_*_ENABLED`, `MCP_*_URL`, `MCP_*_API_KEY` | Each enabled server is discovered as an MCP tool domain |
| 🎬 Video | `VIDEO_GATEWAY_BASE_URL`, `VIDEO_GATEWAY_API_KEY` | The video workflow submits only after its concept and storyboard are ready |

### Agent settings

```dotenv
# Required for the Agent loop
LLM_SUPPORTS_TOOLS=true
AGENT_MAX_ITERATIONS=8
AGENT_CONTEXT_MESSAGE_LIMIT=30
AGENT_CONTEXT_TOKEN_BUDGET=6000

# Optional: use a cheaper/faster model only for candidate tool selection.
# Leave blank to reuse AGENT_MODEL.
AGENT_TOOL_SELECTOR_MODEL=
AGENT_TOOL_SELECTOR_MAX_TOOLS=3
```

`AGENT_MAX_ITERATIONS` caps one run's model/tool loop. Raise it for workflows that genuinely need more tool turns; keep it bounded to avoid runaway requests. Tool selection is automatic: do **not** manually list tools in the request.

### Vision and image generation

```dotenv
# Dedicated vision provider; DASHSCOPE_API_KEY is reused when VISION_LLM_API_KEY is blank.
VISION_LLM_BASE_URL=https://your-provider.example/v1
VISION_LLM_API_KEY=
VISION_LLM_MODEL=your-vision-model

# Pick one image mode.
IMAGE_PROVIDER=openai_images
IMAGE_BASE_URL=https://your-image-provider.example/v1
IMAGE_API_KEY=replace-with-image-provider-key
IMAGE_MODEL=your-image-model
```

- **OpenAI-compatible images**: use `IMAGE_PROVIDER=openai_images` when the provider implements `/images/generations` and multipart `/images/edits`.
- **Qwen Image**: use `IMAGE_PROVIDER=qwen_image`, point `IMAGE_BASE_URL` at Qwen compatible mode, and set `IMAGE_MODEL=qwen-image-3.0`.
- Vision receives the public local-media URL of uploaded/generated images, so set `PUBLIC_MEDIA_BASE_URL` to an address your vision provider can reach.

### RAG and embeddings

```dotenv
RAG_CHUNKER=router
RAG_EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
RAG_EMBEDDING_MODEL=your-embedding-model
# Optional dedicated key. Blank means LLM_API_KEY is reused.
NVIDIA_API_KEY=

# Optional dedicated semantic-chunking endpoint; blank means LLM_BASE_URL.
RAG_SEMANTIC_CHUNK_BASE_URL=
RAG_SEMANTIC_CHUNK_MODEL=your-chat-model
```

Use one stable embedding model for a collection. Changing the model later requires re-ingesting documents because the stored vectors belong to the original embedding space.

### Speech, search, weather, and MCP

```dotenv
# DashScope ASR + TTS
DASHSCOPE_API_KEY=replace-with-dashscope-key
DASHSCOPE_ASR_MODEL=fun-asr-realtime
DASHSCOPE_TTS_MODEL=qwen-audio-3.0-tts-flash
DASHSCOPE_TTS_VOICE=longanhuan_v3.6

# Local research tools
TAVILY_API_KEY=
WEATHER_API_KEY=

# Optional MCP server examples
MCP_MBTI_ENABLED=false
MCP_MBTI_URL=
MCP_MBTI_API_KEY=
MCP_MCD_ENABLED=false
MCP_MCD_URL=
MCP_MCD_API_KEY=
```

MCP servers are not hard-wired into the Agent prompt. When enabled, Luma discovers their official tool schemas at runtime, keeps each server as a tool domain, and lets the Agent select them from the user request and conversation context.

### Video generation

```dotenv
VIDEO_GATEWAY_BASE_URL=https://your-video-gateway.example/video-api
VIDEO_GATEWAY_API_KEY=replace-with-video-gateway-key
VIDEO_RESOLUTION=480p
VIDEO_ASPECT_RATIO=16:9
VIDEO_INSTANCE_TYPE=ultra
VIDEO_POLL_TIMEOUT_SECONDS=900
```

Video is an opt-in integration. The Agent first creates and confirms a compact video concept, subject/scene references, and a multi-panel storyboard. It then uses the reviewed storyboard plus a production prompt to submit **one** final video job to the configured gateway. Without the gateway URL and key, planning and storyboard work can still run, but submission is unavailable.

## 📡 Agent Run Example

```bash
curl -X POST http://localhost:8001/v1/agent/runs \
  -H "Authorization: Bearer $LUMA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "demo",
    "session_id": "sess_replace_me",
    "model": "your-agent-model",
    "messages": [
      {"role":"user","content":"Search this week’s AI news and create an infographic."}
    ]
  }'
```

The request returns a `job_id`. Subscribe to `/v1/agent/runs/{job_id}/events` for live progress instead of polling repeatedly.

## 🛡️ Deployment Notes

- 🔐 Never commit `.env`, local media, coding workspaces, or database dumps.
- 🧱 PostgreSQL and Redis remain private to the Compose network by default.
- 🌐 Put TLS and cache headers in a reverse proxy before exposing the API publicly.
- 💾 Mount `./data` on durable local storage before long-running RAG or media workloads.
- 🧪 `docker compose logs -f api` shows Agent, RAG, and video reconciliation activity.

---

<div align="center">
  Built for teams that want an agent runtime they can inspect, extend, and run themselves. ✦
</div>
