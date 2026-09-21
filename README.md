# LumaNova Agent Core

LumaNova Agent Core is the backend-only extraction of the LumaNova agent
runtime. It deliberately excludes the product website, browser cookies,
accounts, passwords, SMS, payment, membership, API-key issuance, and CDN/S3
deployment wiring.

## Included capabilities

- LangChain/LangGraph-style agent loop with dynamic tool selection
- OpenAI-compatible chat and tool calling providers
- Vision context and image asset selection
- Image generation and editing
- Video concept, subject, scene, storyboard, quality review, and one-shot
  storyboard-to-video workflow
- RAG ingestion/retrieval, read-only coding analysis, web search, weather,
  MCP tools, and DashScope ASR/TTS
- PostgreSQL durable state and Redis jobs/events/cancellation

## Media and providers

This repository stores generated and uploaded assets locally under
`./data/media`. It does not contain S3, CloudFront, user account pools, or
production credentials.

Set provider credentials in `backend/.env` from `backend/.env.example`:

- Any OpenAI-compatible LLM: `LLM_BASE_URL`, `LLM_API_KEY`, `AGENT_MODEL`
- Vision: `VISION_LLM_BASE_URL`, `VISION_LLM_API_KEY`, `VISION_LLM_MODEL`
- Images: set `IMAGE_PROVIDER=openai_images` for an OpenAI image endpoint or
  ChatGPT2API; set `IMAGE_PROVIDER=qwen_image` for Qwen Image 3.
- ASR/TTS: `DASHSCOPE_API_KEY`
- Search/weather/MCP/video: enable only the corresponding keys and URLs.

`basketikun/chatgpt2api` can be deployed separately as an optional image
provider. Point `IMAGE_BASE_URL` at its `/v1` endpoint. It is not bundled
here, and deployments must comply with the upstream project and provider
terms. Qwen Image needs no ChatGPT2API installation: configure its compatible
mode Base URL, DashScope key, and `qwen-image-3.0` model instead.

The video gateway is intentionally opt-in. Set `VIDEO_GATEWAY_BASE_URL` and a
gateway key only after you have separately obtained access from that provider.

## Extraction status

The runtime, skills, tools, image-provider adapter, and local-media defaults
are extracted. `backend/sql.py` uses an independent Core Schema:
`core_clients` identifies a caller/service principal, and `core_projects`
groups that caller's sessions and RAG collections. Chat, assets, jobs, video
state and RAG rows use `client_id`; sessions and collections additionally
carry `project_id` (defaulting to `default`). There are no historical user,
password, session-cookie, payment, membership, or phone tables.

## Development setup

```bash
cp .env.example .env
cp backend/.env.example backend/.env
```

Set the same `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` values in
both files. Then fill the provider settings and a long random `CORE_API_KEY` in
`backend/.env`.

```bash
docker compose up --build -d
curl http://localhost:8001/healthz
```

`docker compose logs -f api` shows startup, agent, RAG, and video reconciliation
logs. PostgreSQL and Redis are private to the Compose network; do not add host
port mappings unless an operational need requires them.

Set `CORE_PORT` in the root `.env` when port `8001` is already used. If you do,
also set `PUBLIC_API_BASE`, `PUBLIC_IMAGE_BASE_URL`, and `PUBLIC_MEDIA_BASE_URL`
in `backend/.env` to the externally reachable address.

## Authentication and tenancy

Every non-health endpoint requires:

```http
Authorization: Bearer <CORE_API_KEY>
X-Agent-Project-ID: my-project     # optional; defaults to `default`
```

`CORE_API_KEY` authenticates one self-hosted deployment. `X-Agent-Project-ID`
isolates that deployment's sessions, RAG collections, coding workspaces, and
video work within the durable Core Schema. It is deliberately not a browser
Cookie system or a user billing system.

## HTTP and WebSocket API

The core has a complete service API rather than a website-specific facade:

| Area | Endpoints |
| --- | --- |
| Health | `GET /healthz` |
| OpenAI-compatible proxy | `POST /v1/chat/completions`, `POST /v1/responses`, `GET /v1/models`; upstream SSE is passed through unchanged |
| Agent runs | `POST /v1/agent/runs`, `GET /v1/agent/runs/{id}`, `GET /v1/agent/runs/{id}/events`, `POST /v1/agent/runs/{id}/cancel` |
| Session and history | `POST/GET /v1/sessions`, `PATCH/DELETE /v1/sessions/{id}`, `GET /v1/sessions/{id}/messages`, `GET /v1/sessions/{id}/job`, `POST /v1/sessions/{id}/messages/{message_id}/regenerate` |
| Compatibility aliases | `GET /v1/chat/jobs/{id}`, `GET /v1/chat/jobs/{id}/events`, `POST /v1/chat/jobs/{id}/cancel` |
| Assets and media | `POST /v1/assets`, `GET /temp_file/{file_name}`, `POST /v1/images/generations`, `POST /v1/images/edits` |
| RAG | `POST /v1/rag/files`, collection/document reads, job reads/SSE, `POST /v1/rag/search` |
| Coding | `POST /v1/coding/workspaces` |
| Audio | `WS /v1/audio/asr`, `POST /v1/audio/tts` |
| Video | `GET /v1/sessions/{id}/video-generation`; the Agent owns the video planning and submission workflow |

Agent and RAG executions are accepted with `202`, retained in PostgreSQL, and
published through Redis-backed SSE. A client should consume the `events`
endpoint rather than polling repeatedly. Completed video gateway work is
reconciled in the background and its local media URL is written back to the
project's chat history.

### Agent run example

```bash
curl -X POST http://localhost:8001/v1/agent/runs \
  -H "Authorization: Bearer $CORE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "demo",
    "session_id": "sess_replace_me",
    "model": "your-agent-model",
    "messages": [{"role":"user","content":"Search this week’s AI news and create an infographic."}]
  }'
```

Create the session first with `POST /v1/sessions`. The returned `job_id` can be
subscribed to at `/v1/agent/runs/{job_id}/events`.

## Provider configuration

Only configure the capabilities you intend to use. Core chat needs an
OpenAI-compatible `LLM_BASE_URL`, `LLM_API_KEY`, and `AGENT_MODEL`. Vision,
images, RAG embeddings, ASR/TTS, search/weather, MCP, and video each have
separate optional configuration in `backend/.env.example`.

For images, choose one mode:

- `IMAGE_PROVIDER=openai_images`: any endpoint implementing OpenAI image
  generation plus edits. This can point at a separately deployed compatible
  service such as `basketikun/chatgpt2api`.
- `IMAGE_PROVIDER=qwen_image`: Qwen Image compatible-mode endpoint using only
  its own base URL, API key, and model. No ChatGPT2API installation is needed.

All uploads and generated media are cached under `./data/media`; the API serves
them from `/temp_file/`. In a production deployment, mount `./data` on durable
local storage and put TLS/reverse-proxy caching in front of the API.

## Repository hygiene

- Never commit `backend/.env`, local media, workspaces, or database dumps.
- Generate a unique `CORE_API_KEY` for each deployment.
- Prefer a reverse proxy for TLS and for serving `/temp_file/` from
  `./data/media`.
