---
name: luma-agent-setup
description: "Install, configure, start, and verify the standalone Luma Agent backend from AuroraChloe/Luma-Agent using Docker Compose, PostgreSQL with pgvector, and Redis. Use for a fresh deployment or startup troubleshooting, not for changing the LumaNova production app."
---

# Install Luma Agent

Deploy the standalone backend from [AuroraChloe/Luma-Agent](https://github.com/AuroraChloe/Luma-Agent). Do not use the production LumaNova source, database, or credentials. Installing this skill does not start the service.

## Preflight

- Confirm the target machine and directory. Default to an API reachable only on that machine. Do not replace an existing deployment, expose a public port, or delete volumes without explicit approval.
- Check `git`, `docker info`, and `docker compose version` on the target. If Docker is not running, report that prerequisite. Do not install Docker or change host networking as a side effect.
- Clone the public repository into a new directory with `git clone https://github.com/AuroraChloe/Luma-Agent.git` or reuse a checkout with that verified origin. Record the commit. Do not pull over local edits.
- Read the checked-out `docker-compose.yml`, `Dockerfile`, `.env.example`, and `backend/.env.example`. The public checkout uses `LUMA_PORT`, `LUMA_API_KEY`, and `X-Luma-Project-ID`; do not substitute older `CORE_*` names.

## Configure and start

1. From the repository root, create missing `.env` and `backend/.env` from their respective examples with restrictive file permissions. Do not overwrite existing env files. Set identical `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` in both. For a fresh deployment, generate distinct random values for the database password and `LUMA_API_KEY`. Inside Compose use `POSTGRES_HOST=postgres` and `REDIS_URL=redis://redis:6379/0`.
2. Set `LLM_BASE_URL`, `LLM_API_KEY`, `DEFAULT_MODEL`, and `AGENT_MODEL` in `backend/.env` from the user's provider. For agent tool calling, use a tool-capable model and keep `LLM_SUPPORTS_TOOLS=true`. The upstream key is not the service key. Do not guess either key, print it, paste it into chat, or commit it. Without provider credentials, report the difference between infrastructure ready and Agent verified. Configure image, vision, RAG, search, audio, MCP, and video only when requested.
3. Choose an unused host `LUMA_PORT` (default 8001). Change the API mapping in the deployment checkout from `"${LUMA_PORT:-8001}:8001"` to `"127.0.0.1:${LUMA_PORT:-8001}:8001"`. Leave PostgreSQL and Redis unpublished. Update `PUBLIC_API_BASE` and `PUBLIC_IMAGE_BASE_URL` to the client-reachable base URL and `PUBLIC_MEDIA_BASE_URL` to that base plus `/temp_file`. For a remote host, verify over SSH or use a tunnel; do not expose the API directly. Review authentication separately before public deployment: the current `/temp_file/{file_name}` handler does not call `require_context`.
4. Before building, ensure the applicable Docker ignore file excludes `.env`, `backend/.env`, `data/`, and `.git/`. The Dockerfile copies the entire `backend/` tree; `.gitignore` alone does not keep secrets out of image layers. If a Dockerfile-specific ignore file exists, inspect it too. Pass environment files at runtime through Compose.
5. Run `docker compose config -q` (not an unfiltered config dump), then `docker compose up --build -d` and `docker compose ps`. Preserve the PostgreSQL and Redis named volumes and the `./data` bind mount.

## Verify in layers

- **Process:** `curl -fsS http://127.0.0.1:8001/healthz` with the actual port should return `{"status":"ok"}`. This alone does not prove the provider works.
- **Auth and storage:** request `GET /v1/sessions` without the bearer key and expect 401. Repeat with `Authorization: Bearer <LUMA_API_KEY>` and expect 200. Use an isolated smoke-test project via `X-Luma-Project-ID`.
- **Agent:** create a test session via `POST /v1/sessions`. Submit a short text-only request to `POST /v1/agent/runs` with the returned `session_id`, the same project and a `messages` array containing a user turn. Expect 202, then check `GET /v1/agent/runs/{job_id}` or its `/events` stream for a successful final result. Do not use the `/v1/chat/completions` proxy as proof of the Agent job path.
- If a check fails, inspect `docker compose ps` and a bounded `docker compose logs --tail=80 api postgres redis` locally. Report only redacted errors. Distinguish Docker startup, database/Redis, service authentication, and upstream model failures. Never run `docker compose down -v` to troubleshoot.

Report the checkout commit, endpoint, and which verification layers passed. Never include actual keys in the report.
