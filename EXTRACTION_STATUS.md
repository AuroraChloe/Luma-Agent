# Extraction Boundary

This directory was cut from the production LumaNova service without copying
the frontend, deployment secrets, user accounts, sessions, SMS, payments,
membership, or production media.

## Safe to publish now

- Agent runtime, tool schemas, prompts, skills, MCP adapter and provider adapters
- Local media persistence defaults
- Provider configuration template and optional ChatGPT2API/Qwen Image support
- Redis and PostgreSQL development service definitions

## Core persistence boundary

Persistence has been migrated to an independent `core_clients` /
`core_projects` schema. Runtime identifiers use `client_id`, while `project_id`
groups a caller's sessions and knowledge collections. Usage metering is a
no-op instrumentation hook, not a membership or billing system.

The remaining extraction step is the clean FastAPI transport surface. It will
authenticate a service key, resolve a client/project context, then expose
agent runs, assets, RAG, coding and video routes.

Do not copy a production `sql.py` or `.env` over this directory after cloning.
