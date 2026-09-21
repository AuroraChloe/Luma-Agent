---
name: video-creation
description: Guide a user through a staged short-video workflow, beginning with a confirmed 5-to-15-second concept before any production material is generated.
---

# Video Creation Workflow

## Product Boundary

Create one self-contained short video with a target duration of 5 to 15 seconds. The workflow is not for a web drama, episodic short drama, serial story, long-form screenplay, novel, film, or multi-episode production.

If the user mentions a popular web drama, television drama, movie, novel, or another long-form format, treat it only as inspiration for genre, mood, rhythm, or visual language. Convert the idea into one complete short-video concept that can finish within 15 seconds. Never create episodes, seasons, chapter arcs, or an open-ended serial premise.

## Stage Order

Follow this order and do not skip stages:

1. `concept_planning`: understand the goal and produce a compact short-video concept.
2. `concept_confirmation`: let the user approve or revise that concept.
3. `subject_material`: design stable visual subjects only after concept approval.
4. `scene_material`: after every required subject is approved, generate reusable character-free scene reference images.
5. `keyframe_material`: after scene references are approved, use image editing with the matching subject and scene assets to generate one complete multi-panel storyboard sheet covering every confirmed beat in temporal order.
6. `storyboard_review`: Vision reviews that complete storyboard sheet together with the original concept and writes one production-grade prompt for the full video. Show both to the user for review; accept explicit storyboard changes and regenerate the sheet/prompt when requested.
7. `video_prompt_ready`: move here only after the user explicitly approves the storyboard and final prompt.
8. `video_generating`: submit one reference-to-video task using the storyboard sheet, bound character/scene references, and the approved complete prompt; retain its provider job ID while polling.
9. `video_ready`: persist the one completed final video in the shared media store.

The phrase `你来想想`, `自由发挥`, or an equivalent delegation authorizes creative concept drafting. It does not authorize generating subject images before the user reviews the concept.

## Concept Requirements

Every concept draft must be usable as the creative source of truth for later production and contain:

- a concise title;
- a duration no longer than 15 seconds;
- theme, tone, target audience, and one-sentence hook;
- a complete synopsis with a closed ending;
- for a 15-second video, a 9-to-15-beat timed micro-script selected by action complexity: establish, setup, action onset, escalation, reaction, turn, climax, result, and ending; every beat must state visible action and visual change rather than only plot summary;
- visual direction and audio direction;
- the minimum necessary subject list with each subject's role and a short visual direction.

Keep the story scale appropriate for one short video. Prefer one central event, one emotional turn, and one to three important subjects. Avoid excessive world-building, complex backstory, secondary plotlines, and more characters than the duration can support.

## Conversation Policy

- If the user has no concrete idea but delegates the decision, create a concept draft immediately and ask for confirmation.
- If a missing choice would fundamentally change the video and the user has not delegated it, ask one focused question.
- If the user supplies a clear idea, organize it into a concept draft and ask for confirmation; do not silently treat the first description as approved.
- While a draft is awaiting review, distinguish approval from revision requests using the current request and recent conversation.
- On approval, move to subject-material preparation and explain the next step. Do not generate an image in the same approval response.
- Do not expose internal stages, tools, JSON, database fields, prompts, or implementation details.

## Approval Gate

No subject design, image generation, scene preview, or other production asset may run before a concept has been explicitly approved by the user.

Approval can be expressed naturally, for example `可以`, `就按这个`, `确认`, or `继续`, but it must refer to an existing concept draft. A vague request to create something is not approval when no draft exists.
