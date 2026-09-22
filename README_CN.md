<div align="center">

[English](README.md) | **简体中文**

# ✦ Luma

### 面向对话、工具、媒体与内容生产工作流的自托管 AI 自动化服务。

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](#-快速开始)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](#️-系统架构)
[![LangChain](https://img.shields.io/badge/LangChain-Agent%20Runtime-1C3C3C)](#-agent-运行机制)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)](#️-系统架构)
[![Redis](https://img.shields.io/badge/Redis-Jobs%20%26%20SSE-DC382D?logo=redis&logoColor=white)](#️-系统架构)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](#-快速开始)

</div>

> **Luma** 将自然语言请求变成可观测、可追踪的工具工作流。它支持普通对话、结构化工具调用、看图理解、图片创作、RAG、代码库分析、MCP 服务、语音，以及由分镜图驱动的短视频制作。

## ✨ Luma 能做什么

| | 能力 | 用途 |
| :--: | --- | --- |
| 🧠 | **Agent 运行时** | 动态筛选工具、多步执行、结合上下文的连续对话与可靠收尾 |
| 🔎 | **检索与 MCP** | 网页搜索、天气、远程 MCP 服务，以及被封装为工具的业务工作流 |
| 👁️ | **视觉与图片** | 图片理解、资产选择、生图、修图与图片质量审查 |
| 🎬 | **视频工作流** | 概念规划、角色/场景素材、多格分镜图、视觉质检与分镜图引导的视频生成 |
| 📚 | **RAG** | 文档导入、语义切块、向量检索与聚焦的知识库问答 |
| 💻 | **代码分析** | 上传项目后进行只读的结构、链路、风险和工程实现分析 |
| 🎙️ | **语音** | WebSocket 实时 ASR 与助手最终回复的 TTS |

## 🗺️ 系统架构

```mermaid
flowchart LR
    Client[客户端 / 应用] -->|Bearer Key| API[FastAPI 服务]
    API --> Runtime[Luma Agent Runtime]
    API --> Queue[Redis 任务队列 + SSE]
    API --> DB[(PostgreSQL + pgvector)]
    API --> Media[本地媒体存储]

    Runtime --> LLM[OpenAI 兼容 LLM]
    Runtime --> Tools[本地工具 + MCP]
    Runtime --> Vision[视觉模型]
    Runtime --> Image[图像模型]
    Runtime --> Video[视频网关]
    Runtime --> Speech[ASR / TTS 服务]
```

**长期数据**保存在 PostgreSQL；**运行中的任务、取消信号和事件流**由 Redis 承担。媒体文件默认放在 `./data/media`，不依赖 S3 或 CDN 也能运行。

## 🚀 快速开始

### 1. 准备配置

```bash
git clone https://github.com/<your-account>/Luma-Agent.git
cd Luma-Agent

cp .env.example .env
cp backend/.env.example backend/.env
```

根目录与 `backend/.env` 中的数据库配置需要保持一致。随后至少在 `backend/.env` 配置：

```dotenv
LUMA_API_KEY=replace-with-a-long-random-service-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=replace-with-your-provider-key
AGENT_MODEL=your-agent-model
```

### 2. 启动服务

```bash
docker compose up --build -d
curl http://localhost:8001/healthz
```

预期返回：

```json
{"status":"ok"}
```

### 3. 创建会话

```bash
curl -X POST http://localhost:8001/v1/sessions \
  -H "Authorization: Bearer $LUMA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"demo","title":"我的第一个 Luma 会话"}'
```

> 💡 若 `8001` 端口被占用，在根目录 `.env` 中修改 `LUMA_PORT`。部署到反向代理后，也应同步调整 `backend/.env` 内的 `PUBLIC_API_BASE`、`PUBLIC_IMAGE_BASE_URL` 与 `PUBLIC_MEDIA_BASE_URL`。

## 🧠 Agent 运行机制

Luma 不会把工具调用写死成固定流程。每次请求都会经过以下阶段：

1. **构建上下文**：汇集当前对话、关联视觉资产、项目状态和可选的 Skill 指引。
2. **筛选候选能力**：轻量工具选择器读取近期历史，而非只看最后一句话。
3. **执行 Agent 循环**：主模型获得筛选后的本地工具或 MCP 服务域，决定执行顺序和参数。
4. **回看工具观察结果**：模型根据真实返回决定是否继续调用工具，直到信息充分或任务完成。
5. **保存与推送状态**：结果写入 PostgreSQL，实时状态通过 Redis 驱动的 SSE 推送。

因此，普通问答保持轻量，而“**查新闻 → 生成版图**”“**分镜图 → 视频**”“**图片 → 识图 → 修图**”这类多工具任务也可以自然完成。

## 🔌 API 一览

| 模块 | 接口 |
| --- | --- |
| ❤️ 健康检查 | `GET /healthz` |
| 💬 OpenAI 兼容层 | `POST /v1/chat/completions`、`POST /v1/responses`、`GET /v1/models` |
| ⚡ Agent 任务 | `POST /v1/agent/runs`、任务状态、SSE 事件流、取消 |
| 🗂️ 会话 | `/v1/sessions` 下的创建、列表、修改、删除、历史、重试和最新任务 |
| 🖼️ 图片与文件 | `POST /v1/assets`、`POST /v1/images/generations`、`POST /v1/images/edits`、`GET /temp_file/{name}` |
| 📚 RAG | 文件上传、知识库/文档读取、导入任务与 SSE、`POST /v1/rag/search` |
| 💻 代码分析 | `POST /v1/coding/workspaces` |
| 🎙️ 语音 | `WS /v1/audio/asr`、`POST /v1/audio/tts` |
| 🎬 视频 | `GET /v1/sessions/{id}/video-generation` |

除 `/healthz` 外，接口都需要服务密钥：

```http
Authorization: Bearer <LUMA_API_KEY>
X-Luma-Project-ID: my-project
```

`X-Luma-Project-ID` 可选，默认值为 `default`。它可在同一个部署内隔离会话、知识库、媒体资产、代码工作区和视频项目。

## 🧩 按需配置能力

Luma 的集成彼此独立。复制 `backend/.env.example` 后，只需填写某项能力所需字段即可启用；没有配置的能力不会影响正常聊天，只是不会被 Agent 选中执行。

| 能力 | 必填配置 | 如何启用 |
| --- | --- | --- |
| 💬 对话 + Agent | `LLM_BASE_URL`、`LLM_API_KEY`、`AGENT_MODEL` | 聊天模型配置可用后自动启用 |
| 🧠 工具型 Agent | `LLM_SUPPORTS_TOOLS=true` | 本地工具和 MCP 服务域自动发现；仅在想让筛选器单独使用小模型时配置 `AGENT_TOOL_SELECTOR_MODEL` |
| 👁️ Vision | `VISION_LLM_BASE_URL`、`VISION_LLM_MODEL`、`VISION_LLM_API_KEY` 或 `DASHSCOPE_API_KEY` | 请求确实需要看图时自动进入视觉链路 |
| 🎨 图像 | `IMAGE_PROVIDER`、`IMAGE_BASE_URL`、`IMAGE_API_KEY`、`IMAGE_MODEL` | 配好兼容图像服务后，Agent 可按需调用生图/修图 |
| 📚 RAG | `RAG_EMBEDDING_BASE_URL`、`RAG_EMBEDDING_MODEL`、`NVIDIA_API_KEY` 或 `LLM_API_KEY` | 通过 `/v1/rag/files` 导入文件；Compose 中已包含 pgvector |
| 🎙️ ASR / TTS | `DASHSCOPE_API_KEY` | ASR 使用 `/v1/audio/asr`，TTS 使用 `/v1/audio/tts` |
| 🔎 网页搜索 | `TAVILY_API_KEY` | 需要实时网页信息时，搜索工具会自动进入候选集合 |
| 🌤️ 天气 | `WEATHER_API_KEY` | 用户请求地点天气时，天气工具会自动进入候选集合 |
| 🔌 MCP | `MCP_*_ENABLED`、`MCP_*_URL`、`MCP_*_API_KEY` | 每个启用的 MCP 服务会被发现为独立工具域 |
| 🎬 视频 | `VIDEO_GATEWAY_BASE_URL`、`VIDEO_GATEWAY_API_KEY` | 视频概念与分镜图准备完成后才会提交最终任务 |

### Agent 配置

```dotenv
# Agent 工具循环的基础配置
LLM_SUPPORTS_TOOLS=true
AGENT_MAX_ITERATIONS=8
AGENT_CONTEXT_MESSAGE_LIMIT=30
AGENT_CONTEXT_TOKEN_BUDGET=6000

# 可选：让候选工具筛选器使用更快/更便宜的小模型。
# 留空时复用 AGENT_MODEL。
AGENT_TOOL_SELECTOR_MODEL=
AGENT_TOOL_SELECTOR_MAX_TOOLS=3
```

`AGENT_MAX_ITERATIONS` 限制单次请求中的“模型判断 → 工具调用 → 观察结果”循环次数。复杂工作流可适当调高，但建议始终保留上限，避免异常请求无限循环。工具选择本身是自动完成的，调用方不需要在请求中手工指定工具列表。

### Vision 与生图

```dotenv
# 专用视觉模型；若 VISION_LLM_API_KEY 留空，会复用 DASHSCOPE_API_KEY。
VISION_LLM_BASE_URL=https://your-provider.example/v1
VISION_LLM_API_KEY=
VISION_LLM_MODEL=your-vision-model

# 图像服务二选一。
IMAGE_PROVIDER=openai_images
IMAGE_BASE_URL=https://your-image-provider.example/v1
IMAGE_API_KEY=replace-with-image-provider-key
IMAGE_MODEL=your-image-model
```

- **OpenAI 兼容图像服务**：当上游支持 `/images/generations` 与 multipart `/images/edits` 时，设置 `IMAGE_PROVIDER=openai_images`。
- **Qwen Image**：设置 `IMAGE_PROVIDER=qwen_image`，将 `IMAGE_BASE_URL` 指向千问 compatible-mode，再设置 `IMAGE_MODEL=qwen-image-3.0`。
- Vision 会接收上传或生成图片的公开本地媒体 URL，因此 `PUBLIC_MEDIA_BASE_URL` 必须是你的视觉模型上游能够访问到的地址。

### RAG 与 Embedding

```dotenv
RAG_CHUNKER=router
RAG_EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
RAG_EMBEDDING_MODEL=your-embedding-model
# 可选的独立 Embedding Key。留空时复用 LLM_API_KEY。
NVIDIA_API_KEY=

# 可选的独立语义切块 Endpoint；留空时复用 LLM_BASE_URL。
RAG_SEMANTIC_CHUNK_BASE_URL=
RAG_SEMANTIC_CHUNK_MODEL=your-chat-model
```

同一知识库应使用同一个稳定的 Embedding 模型。模型变更后需重新导入文档，因为旧向量属于原模型的向量空间。

### 语音、搜索、天气与 MCP

```dotenv
# DashScope 的 ASR 与 TTS 共用同一个 Key
DASHSCOPE_API_KEY=replace-with-dashscope-key
DASHSCOPE_ASR_MODEL=fun-asr-realtime
DASHSCOPE_TTS_MODEL=qwen-audio-3.0-tts-flash
DASHSCOPE_TTS_VOICE=longanhuan_v3.6

# 本地检索工具
TAVILY_API_KEY=
WEATHER_API_KEY=

# 可选 MCP 服务示例
MCP_MBTI_ENABLED=false
MCP_MBTI_URL=
MCP_MBTI_API_KEY=
MCP_MCD_ENABLED=false
MCP_MCD_URL=
MCP_MCD_API_KEY=
```

MCP 服务不会被硬编码进主提示词。启用后，Luma 会在运行时读取该服务的官方 Tool Schema，将每台服务保留为一个工具域，再由 Agent 根据用户请求和对话上下文选择是否调用。

### 视频生成

```dotenv
VIDEO_GATEWAY_BASE_URL=https://your-video-gateway.example/video-api
VIDEO_GATEWAY_API_KEY=replace-with-video-gateway-key
VIDEO_RESOLUTION=480p
VIDEO_ASPECT_RATIO=16:9
VIDEO_INSTANCE_TYPE=ultra
VIDEO_POLL_TIMEOUT_SECONDS=900
```

视频是独立接入的可选能力。Agent 会先产出并确认紧凑的视频概念、角色/场景素材与多格分镜图，再基于已审查的分镜图和生产级提示词向视频网关提交**一个**最终任务。未配置网关 URL 与 Key 时，概念、素材和分镜流程仍可运行，但不能提交最终视频。

## 📡 发起一次 Agent 任务

```bash
curl -X POST http://localhost:8001/v1/agent/runs \
  -H "Authorization: Bearer $LUMA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "project_id": "demo",
    "session_id": "sess_replace_me",
    "model": "your-agent-model",
    "messages": [
      {"role":"user","content":"帮我汇总本周 AI 新闻，并生成一张信息版图。"}
    ]
  }'
```

接口会返回 `job_id`。推荐订阅 `/v1/agent/runs/{job_id}/events` 获取实时状态，而不是持续轮询。

## 🛡️ 部署说明

- 🔐 不要提交 `.env`、本地媒体文件、代码工作区和数据库导出文件。
- 🧱 PostgreSQL 与 Redis 默认只在 Docker Compose 内网中可见。
- 🌐 对外部署时，建议在 API 前加启用 TLS 的反向代理并配置缓存头。
- 💾 长期运行 RAG 或媒体任务前，请将 `./data` 挂载到持久化本地存储。
- 🧪 使用 `docker compose logs -f api` 查看 Agent、RAG 与视频任务的运行日志。

---

<div align="center">
  为希望自行检查、扩展并运行 Agent Runtime 的团队而构建。✦
</div>
