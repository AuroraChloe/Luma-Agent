---
name: mbti
description: Run an interactive MBTI questionnaire across multiple chat turns and calculate the final result.
tools:
  - start_mbti_test
  - answer_question
  - get_progress
  - calculate_mbti_result
---

# MBTI Skill

## Purpose

Run an interactive MBTI questionnaire inside the current chat session.

## When To Use

Use these tools only when the user:

- explicitly asks to start, take, continue, or restart an MBTI test
- is answering a question in an MBTI test already running in this chat
- asks for the progress or final result of that test

Do not use them for general personality discussion, explaining MBTI theory, guessing a personality type from casual conversation, or unrelated questions.

## Workflow

1. Call `start_mbti_test` once when the user starts or restarts a test.
2. Present only the returned current question and the 1-5 scoring options naturally.
3. Convert the user's answer to a score only when its meaning is clear:
   - strongly disagree = 1
   - disagree = 2
   - neutral or unsure = 3
   - agree = 4
   - strongly agree = 5
4. Call `answer_question` once for each answer. The runtime automatically supplies the saved test session.
5. Continue presenting the next question until the service reports that all questions are complete.
6. Call `calculate_mbti_result` only after completion.
7. Use `get_progress` only when the user asks about progress.

If the user's answer cannot be mapped confidently to 1-5, ask them to clarify instead of guessing.

## Answer Rules

- Do not expose MCP, tool names, JSON, API keys, internal session objects, or routing details.
- Do not print the hidden test session object.
- Keep each question easy to answer and preserve the service's original wording.
- Explain the final result as an informal personality reference, not a medical or psychological diagnosis.
