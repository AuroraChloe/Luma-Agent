---
name: weather
description: Query current real-time weather with Amap. If the user asks for local/current weather without a location, use the current HTTP client IP as the location fallback.
tools:
  - weather_current
---

# Weather Skill

## Purpose

Answer user requests about current, real-time weather, temperature, humidity, wind, rain, and whether the user needs an umbrella.

This skill can use exactly this tool:

- `weather_current`

## Tool Schema

### `weather_current`

Use this tool to query real-time weather.

Valid arguments:

```json
{
  "location": "string",
  "client_ip": "string"
}
```

Rules:

- `location` is a city, district, county, province-level area, or administrative region name.
- `client_ip` must only come from system-provided context, never from user text.
- If the user provides a location, set `location` and leave `client_ip` empty unless already present in system context.
- If the user asks for local/current weather but does not provide a location, leave `location` empty and use the system-provided current HTTP client IP.
- Do not use IP lookup as a standalone answer. IP lookup exists only as a fallback inside weather queries.

## When To Use

Use `weather_current` when the user asks about:

- current weather
- current temperature
- humidity
- wind
- rain
- whether to bring an umbrella
- local weather / weather here / current location weather

Examples:

- `杭州现在天气怎么样`
- `余杭区现在几度`
- `今天要带伞吗`
- `我这里现在天气怎么样`
- `当前天气情况`
- `外面冷不冷`

## Location Rules

1. If the user explicitly names a location, use that location.
2. If the user asks follow-up weather questions without a location, you may inherit the recent weather location from conversation context.
3. If the user asks for current/local weather and no location exists in text or context, call `weather_current` with empty `location`; the runtime will use current HTTP client IP fallback.
4. If neither location nor IP fallback is available, ask the user to provide a city or district.

## Must Not Use

Do not use this skill for:

- climate knowledge that does not require real-time weather
- historical weather unless the tool supports it
- geography questions unrelated to weather
- user IP lookup by itself
- questions like `我的 IP 是多少`

For non-weather questions, choose `none`.

## Answer Rules

- Base the answer only on the tool result.
- If the location was inferred from IP, do not mention IP, automatic location, positioning, or inference unless the user explicitly asks how the location was determined.
- If the tool cannot locate the user by IP, ask for a city or district.
- Do not expose raw JSON, API keys, adcode internals, tool names, IP addresses, location source, or reporttime/update timestamps unless the user explicitly asks for them.


## Tone

- Answer like a normal weather assistant.
- Keep the location lookup invisible to the user.
- Prefer short, natural replies such as `上海现在阴，28℃，湿度70%，东风≤3级。`
