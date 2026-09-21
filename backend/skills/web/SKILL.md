---
name: web
description: Use web_search only for current or external public information.
tools:
  - web_search
---

# Web Search Skill

Use `web_search` only when the user explicitly needs online public information, the model cannot answer the current question, or there are no other available tools for the question.

Default choice: `none`.

## Use `web_search`

Call `web_search` when the current request needs information outside the model's stable knowledge, such as:

- explicit search intent: `搜索`, `搜一下`, `查一下`, `联网`, `网上`, `资料`, `source`, `link`
- latest/current/recent facts, news, prices, releases, schedules, policies, public company status, or version changes
- source-backed public facts where the user asks for verification, links, references, or official information
- a public URL, webpage, article, paper, document, repository, product, company, or person that must be checked online

## Choose `none`

Do not call `web_search` when the model can answer from reasoning, chat history, uploaded content, or selected knowledge base.

Choose `none` for normal explanation, writing, translation, math, coding help, summarization, brainstorming, image understanding, image generation/editing, weather, and knowledge-base/RAG questions.

Mentioning a company, product, technology, person, or place is not enough. Search only when the user asks for online/current/source-backed information.

## Tool Arguments

Use this shape:

```json
{
  "query": "用户当前完整问题",
  "search_depth": "advanced"
}
```

Rules:

- Keep `query` as the full current user question.
- When the request is a follow-up, complete the search query with the needed subject from recent context.
- Preserve relative time words such as "最新", "最近", "目前", and "当前" as relative time.
- Do not invent a year, month, date, quarter, or other concrete time range unless it is explicitly present in the current request or a user message in recent context.
- Use `advanced` unless the user explicitly asks for a quick/simple search.
- Normally call the tool at most once for one user request.


## Answer

Answer naturally from the search results. Do not expose raw JSON, tool names, API names, request IDs, or internal process.

If the results are weak, unrelated, or inconsistent, say so briefly instead of pretending certainty.
