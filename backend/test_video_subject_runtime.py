import video_subject_runtime as runtime
from video_subject_runtime import (
    ReadinessDecision,
    SubjectUpdate,
    VideoSubjectDecision,
    _resolve_subject,
    _concept_subject_approval_state,
    _save_subject_updates,
    _seed_concept_subjects,
    _target_subjects,
)
from video_subject_tools import build_subject_design_prompt


class FakeToolbox:
    def __init__(self):
        self.rows = []

    def save_subject_spec(self, update, *, status):
        existing = next(
            (
                item
                for item in self.rows
                if item["subject_id"] == update.get("subject_id")
                or item["display_name"].lower() == str(update.get("display_name") or "").lower()
            ),
            None,
        )
        if existing:
            existing.update({
                "display_name": update.get("display_name") or existing["display_name"],
                "subject_type": update.get("subject_type") or existing["subject_type"],
                "story_role": update.get("story_role") or existing["story_role"],
                "spec": {**existing["spec"], **(update.get("spec") or {})},
                "status": status,
            })
            return dict(existing)
        row = {
            "subject_id": f"subject_{len(self.rows) + 1}",
            "ordinal": len(self.rows) + 1,
            "display_name": update["display_name"],
            "subject_type": update.get("subject_type") or "other",
            "story_role": update.get("story_role") or "",
            "spec": dict(update.get("spec") or {}),
            "status": status,
            "current_asset_id": None,
            "approved_asset_id": None,
        }
        self.rows.append(row)
        return dict(row)


def _decision(*updates, target_scope="all_pending", target_refs=None):
    return VideoSubjectDecision(
        intent="generate_design",
        readiness=ReadinessDecision(status="ready"),
        subject_updates=list(updates),
        target_scope=target_scope,
        target_subject_refs=target_refs or [],
    )


def test_explicit_unknown_reference_never_falls_back_to_only_subject():
    subjects = [{"subject_id": "subject_1", "ordinal": 1, "display_name": "Alpha"}]
    assert _resolve_subject(subjects, "Beta", None) is None
    assert _resolve_subject(subjects, "", None) == subjects[0]


def test_multiple_subject_updates_create_isolated_subjects_and_specs():
    toolbox = FakeToolbox()
    decision = _decision(
        SubjectUpdate(display_name="Alpha", subject_ref="Alpha", spec={"surface": "metal"}),
        SubjectUpdate(display_name="Beta", subject_ref="Beta", spec={"clothing": "coat"}),
        SubjectUpdate(display_name="Gamma", subject_ref="Gamma", spec={"fur": "white"}),
    )
    saved, known = _save_subject_updates(toolbox, decision, [])
    assert [item["ordinal"] for item in saved] == [1, 2, 3]
    assert [item["display_name"] for item in known] == ["Alpha", "Beta", "Gamma"]
    assert known[0]["spec"] == {"surface": "metal", "identity": "Alpha"}
    assert known[1]["spec"] == {"clothing": "coat", "identity": "Beta"}
    assert known[2]["spec"] == {"fur": "white", "identity": "Gamma"}


def test_approved_concept_seeds_stable_roster_in_concept_order():
    toolbox = FakeToolbox()
    project = {
        "project_brief": {
            "concept": {
                "subjects": [
                    {"display_name": "Alpha", "subject_type": "character", "role": "lead"},
                    {"display_name": "Beta", "subject_type": "vehicle", "role": "transport"},
                ]
            }
        }
    }
    created, known = _seed_concept_subjects(toolbox, project, [])
    assert [item["display_name"] for item in created] == ["Alpha", "Beta"]
    assert [item["ordinal"] for item in known] == [1, 2]


def test_all_pending_targets_every_subject_without_a_design():
    subjects = [
        {"subject_id": "a", "ordinal": 1, "display_name": "Alpha", "current_asset_id": "asset_a"},
        {"subject_id": "b", "ordinal": 2, "display_name": "Beta", "current_asset_id": None},
        {"subject_id": "c", "ordinal": 3, "display_name": "Gamma", "current_asset_id": None},
    ]
    targets = _target_subjects(_decision(), subjects, "a")
    assert [item["subject_id"] for item in targets] == ["b", "c"]


def test_selected_scope_respects_explicit_subject_reference():
    subjects = [
        {"subject_id": "a", "ordinal": 1, "display_name": "Alpha", "current_asset_id": None},
        {"subject_id": "b", "ordinal": 2, "display_name": "Beta", "current_asset_id": None},
    ]
    targets = _target_subjects(
        _decision(target_scope="selected", target_refs=["Beta"]),
        subjects,
        "a",
    )
    assert [item["subject_id"] for item in targets] == ["b"]


def test_prompt_only_contains_the_current_subject_spec():
    prompt = build_subject_design_prompt({
        "display_name": "Alpha",
        "subject_type": "vehicle",
        "story_role": "transport",
        "spec": {"surface": "metal", "empty": ""},
    })
    assert "metal" in prompt
    assert "empty" not in prompt
    assert "Beta" not in prompt


def test_subject_decision_has_no_project_brief_writeback_field():
    assert "project_brief" not in VideoSubjectDecision.model_fields


def test_concept_roster_requires_only_current_subjects_after_a_replacement():
    project = {
        "project_brief": {
            "concept": {
                "subjects": [
                    {"display_name": "Daybreak"},
                    {"display_name": "Opponent"},
                ]
            }
        }
    }
    subjects = [
        {"display_name": "Daybreak", "approved_asset_id": "asset_daybreak"},
        {"display_name": "Opponent", "approved_asset_id": "asset_opponent"},
    ]
    assert _concept_subject_approval_state(project, subjects) == (True, [])


def test_one_runtime_call_generates_all_pending_subjects_and_returns_one_summary():
    subjects = [
        {
            "subject_id": f"subject_{index}",
            "ordinal": index,
            "display_name": name,
            "subject_type": "other",
            "story_role": "independent role",
            "spec": {"identity": name, "feature": f"feature_{index}"},
            "status": "ready",
            "current_asset_id": None,
            "approved_asset_id": None,
        }
        for index, name in enumerate(("Alpha", "Beta", "Gamma"), start=1)
    ]
    generated_prompts = {}

    class BatchToolbox(FakeToolbox):
        def __init__(self, **_kwargs):
            self.rows = [dict(item) for item in subjects]

        def consume_image_quota(self):
            return True

        def generate_subject_design(self, subject, *, reference_image=None):
            prompt = build_subject_design_prompt(subject)
            generated_prompts[subject["display_name"]] = prompt
            return {
                "status": "1",
                "image_url": f"https://example.test/{subject['subject_id']}.png",
                "compiled_prompt": prompt,
            }

        def inspect_subject_design(self, subject, result):
            return {"report": {"passed": True, "score": 100, "blocking_issues": []}, "usage": {}}

        def persist_design(self, subject, result, *, inspection, source_asset_id=None):
            return {"link": {"version": 1}, "asset": {"public_url": result["image_url"]}}

        def reload_subject(self, subject_id):
            return next(item for item in self.rows if item["subject_id"] == subject_id)

    project = {
        "stage": "subject_material",
        "status": "concept_approved",
        "active_subject_id": None,
        "project_brief": {"concept": {"subjects": []}},
    }
    decision = _decision(target_scope="all_pending")
    originals = {
        "get_or_create_video_project": runtime.get_or_create_video_project,
        "list_video_subjects": runtime.list_video_subjects,
        "update_video_project": runtime.update_video_project,
        "_director_decision": runtime._director_decision,
        "VideoSubjectToolbox": runtime.VideoSubjectToolbox,
    }
    runtime.get_or_create_video_project = lambda *_args, **_kwargs: dict(project)
    runtime.list_video_subjects = lambda *_args, **_kwargs: [dict(item) for item in subjects]
    runtime.update_video_project = lambda *_args, **kwargs: {**project, **kwargs}
    runtime._director_decision = lambda **_kwargs: (decision, {})
    runtime.VideoSubjectToolbox = BatchToolbox
    try:
        result = runtime.run_video_subject_chat(
            1,
            session_id="session_test",
            model="model_test",
            messages=[],
            user_message="prepare every pending subject",
        )
    finally:
        for name, value in originals.items():
            setattr(runtime, name, value)

    assert result["content"].count("**主体 ") == 3
    assert [result["content"].find(name) for name in ("Alpha", "Beta", "Gamma")] == sorted(
        result["content"].find(name) for name in ("Alpha", "Beta", "Gamma")
    )
    assert set(generated_prompts) == {"Alpha", "Beta", "Gamma"}
    for name, prompt in generated_prompts.items():
        assert f"主体名称：{name}" in prompt
        for other in set(generated_prompts) - {name}:
            assert other not in prompt
