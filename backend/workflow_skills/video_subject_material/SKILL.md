---
name: video-subject-material
description: Guide a user from a vague or complete video idea to approved, reusable subject reference material through natural conversation, subject design generation, visual inspection, revision, approval, and optional scene preview.
---

# Video Subject Material

## Purpose

Prepare stable subject reference material for a later video-generation pipeline. A subject may be a person, animal, anthropomorphic character, creature, vehicle, product, object, or another story-defining visual entity.

This workflow only completes the `subject_material` stage. Do not claim to generate scripts, scenes, shots, animation, audio, or final video.

## Conversation Policy

- Understand the user's creative goal through natural dialogue; do not run a fixed questionnaire.
- Decide whether the current information can define a visually stable subject without inventing its core identity.
- Ask one focused, high-value follow-up when identity-critical information is missing or contradictory.
- Treat phrases such as `你来设计`, `自由发挥`, or equivalent language as permission to infer non-critical art direction.
- Do not ask for details the user already supplied.
- If the request is sufficiently concrete, proceed without an unnecessary confirmation round.
- Keep separate subjects distinct and assign each a stable display name and subject reference.
- When the user asks to start or continue subject preparation without naming one subject, process every pending subject in the approved concept before returning one combined response.
- When the user explicitly names one or more subjects, limit the operation to those subjects.

## Readiness

Block generation only when one of these is unresolved and the user has not delegated it:

- what the subject fundamentally is;
- its role or intended identity in the video concept;
- the visual direction needed to recognize it consistently;
- a contradiction that would materially change the subject.

Colors, materials, clothing details, accessories, expressions, and secondary styling may be inferred when they do not redefine the core identity.

## Subject Specification

Preserve the user's intent in a structured specification. Include only relevant fields:

- identity and subject type;
- story role and personality or functional traits;
- body shape, silhouette, proportions, age impression, and species where relevant;
- face, hair, fur, surface, material, or mechanical features;
- clothing, accessories, palette, and signature identifiers;
- visual style and rendering direction;
- constraints, elements to preserve, and elements to avoid.

Do not copy conversational filler, prior evaluation, or asset-selection phrases into image instructions.
Never copy a field from one subject into another subject's specification, even when both appear in the same concept.

## Design Asset

Generate one design sheet per subject. The sheet is the production-facing subject reference and should:

- show one consistent subject on a clean neutral background;
- include a clear full-body primary view plus useful secondary angles or detail views;
- preserve one silhouette, palette, costume, material language, and signature features;
- avoid story scenes, unrelated props, duplicate characters, decorative layout noise, and unreadable labels;
- be usable as a visual consistency reference for later video generation.

After generation or revision, always inspect the result against the saved subject specification. The visual inspection is a technical quality gate, not the final aesthetic decision.

Allow at most one automatic correction for blocking structural or identity defects. The user makes the final approval decision.

## Revision And Approval

- Apply revisions to the explicitly selected or current design version.
- Convert user feedback into concrete visual changes while preserving unmentioned identity details.
- Never silently overwrite an older version; create a new version linked to its source.
- Mark exactly one approved design version as eligible for later production use.
- A new revision after approval returns the subject to `awaiting_review` until approved again.

## Scene Preview

Only generate a scene preview after the subject has an approved design and the user explicitly requests or accepts the preview offer.

- Use the approved design as the visual source.
- Preserve subject identity and place it in a simple context chosen from the user's video idea or a restrained inferred scene.
- Mark the preview as reference-only and never use it as the approved production subject asset.
- Do not generate a preview automatically after the design sheet.

## User-Facing Responses

- Do not expose internal tool names, database identifiers, prompts, paths, JSON, inspection fields, or workflow implementation.
- When information is missing, explain what decision is needed and give the user permission to delegate it.
- When a design is ready, show the image and ask whether to revise or approve it.
- After every required subject is approved, hand the project to the script-bound keyframe stage. A scene preview remains optional while subject work is still in progress and never replaces a production keyframe.
- When an operation fails, explain the visible outcome naturally and preserve the previous approved/current version.
