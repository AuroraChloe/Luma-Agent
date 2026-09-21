import unittest

from video_keyframe_runtime import (
    KeyframeDirection,
    _compile_plan,
    _is_explicit_storyboard_submission_request,
    _render_storyboard_review,
    _required_subjects,
)
from video_keyframe_tools import build_keyframe_prompt, build_keyframe_revision_prompt, build_storyboard_sheet_revision_prompt


def _project():
    return {
        "project_brief": {
            "concept": {
                "title": "Test",
                "subjects": [
                    {"display_name": "Alpha"},
                    {"display_name": "Beta"},
                ],
                "beats": [
                    {
                        "time_range": "0-3s",
                        "purpose": "opening",
                        "content": "Alpha enters the room",
                        "visual": "Alpha in the doorway",
                    },
                    {
                        "time_range": "3-6s",
                        "purpose": "turn",
                        "content": "Beta closes the door",
                        "visual": "Beta beside the door",
                    },
                ],
            }
        }
    }


def _subjects():
    return [
        {
            "subject_id": "subject_alpha",
            "ordinal": 1,
            "display_name": "Alpha",
            "subject_type": "character",
            "approved_asset_id": "asset_alpha",
            "spec": {"palette": "red"},
        },
        {
            "subject_id": "subject_beta",
            "ordinal": 2,
            "display_name": "Beta",
            "subject_type": "character",
            "approved_asset_id": "asset_beta",
            "spec": {"palette": "blue"},
        },
    ]


def _scenes():
    return [
        {
            "scene_id": "scene_room",
            "ordinal": 1,
            "display_name": "The Room",
            "setting": "A red room",
            "approved": True,
            "asset_id": "asset_room",
            "public_url": "https://example.test/room.png",
        }
    ]


class VideoKeyframeRuntimeTest(unittest.TestCase):
    def test_plan_is_bound_to_every_source_beat_and_resolves_subjects(self):
        rows = _compile_plan(
            _project(),
            _subjects(),
            [
                KeyframeDirection(source_beat_index=1, subject_refs=["Alpha"], shot_size="wide"),
                KeyframeDirection(source_beat_index=2, subject_refs=["subject_beta"], shot_size="close"),
            ],
        )
        self.assertEqual([item["source_beat_index"] for item in rows], [1, 2])
        self.assertEqual(rows[0]["script_content"], "Alpha enters the room")
        self.assertEqual(rows[0]["subject_ids"], ["subject_alpha"])
        self.assertEqual(rows[1]["subject_ids"], ["subject_beta"])
        self.assertEqual(len(rows[0]["source_beat_fingerprint"]), 64)

    def test_plan_rejects_missing_or_duplicate_beats(self):
        with self.assertRaises(ValueError):
            _compile_plan(
                _project(),
                _subjects(),
                [
                    KeyframeDirection(source_beat_index=1),
                    KeyframeDirection(source_beat_index=1),
                ],
            )

    def test_plan_rejects_unknown_subject_reference(self):
        with self.assertRaises(ValueError):
            _compile_plan(
                _project(),
                _subjects(),
                [
                    KeyframeDirection(source_beat_index=1, subject_refs=["Unknown"]),
                    KeyframeDirection(source_beat_index=2, subject_refs=["Beta"]),
                ],
            )

    def test_named_subject_is_forced_into_reference_set(self):
        rows = _compile_plan(
            _project(),
            _subjects(),
            [
                KeyframeDirection(source_beat_index=1, subject_refs=[]),
                KeyframeDirection(source_beat_index=2, subject_refs=[]),
            ],
        )
        self.assertEqual(rows[0]["subject_ids"], ["subject_alpha"])
        self.assertEqual(rows[1]["subject_ids"], ["subject_beta"])

    def test_required_subjects_follow_confirmed_concept_roster(self):
        required, missing = _required_subjects(_project(), _subjects())
        self.assertEqual(missing, [])
        self.assertEqual(
            [item["subject_id"] for item in required],
            ["subject_alpha", "subject_beta"],
        )

    def test_keyframe_prompt_keeps_script_and_reference_mapping(self):
        frame = {
            "time_range": "0-3s",
            "purpose": "opening",
            "script_content": "Alpha enters the room",
            "script_visual": "Alpha in the doorway",
            "plan": {"shot_size": "wide"},
        }
        prompt = build_keyframe_prompt(frame, [_subjects()[0]])
        self.assertIn("Alpha enters the room", prompt)
        self.assertIn('"reference_image":1', prompt)
        self.assertIn('"display_name":"Alpha"', prompt)
        self.assertIn("不得增加、删除、替换剧情事件", prompt)
        self.assertIn("1536x1024", prompt)

    def test_plan_and_prompt_bind_the_approved_scene_asset(self):
        rows = _compile_plan(
            _project(),
            _subjects(),
            [
                KeyframeDirection(source_beat_index=1, subject_refs=["Alpha"], scene_refs=["The Room"]),
                KeyframeDirection(source_beat_index=2, subject_refs=["Beta"], scene_refs=["scene_room"]),
            ],
            _scenes(),
        )
        self.assertEqual(rows[0]["plan"]["scene_ids"], ["scene_room"])
        prompt = build_keyframe_prompt(rows[0], [_subjects()[0]], _scenes())
        self.assertIn('"reference_kind":"scene"', prompt)
        self.assertIn("The Room", prompt)

    def test_revision_prompt_anchors_current_beat_and_rejects_timeline_leakage(self):
        frame = {
            "source_beat_index": 1,
            "time_range": "0-3s",
            "purpose": "opening",
            "script_content": "Alpha enters the room",
            "script_visual": "Alpha in the doorway",
            "plan": {"shot_size": "wide"},
        }
        prompt = build_keyframe_revision_prompt(
            frame,
            [_subjects()[0]],
            {
                "blocking_issues": ["Beta appears too early"],
                "correction_prompt": "Remove Beta",
            },
            script_timeline=_project()["project_brief"]["concept"]["beats"],
        )
        self.assertIn("Alpha enters the room", prompt)
        self.assertIn("Beta appears too early", prompt)
        self.assertIn("不要把前后节拍的事件混进当前画面", prompt)
        self.assertIn('"reference_image":2', prompt)

    def test_storyboard_review_exposes_the_final_prompt_before_submission(self):
        reply = _render_storyboard_review(
            {"public_url": "https://example.test/storyboard.png"},
            {"production_prompt": "[Picture 1] 是分镜图合集。镜头缓慢推进。"},
        )
        self.assertIn("https://example.test/storyboard.png", reply)
        self.assertIn("最终视频生成提示词（审阅稿）", reply)
        self.assertIn("确认分镜并开始生成视频", reply)

    def test_storyboard_revision_prompt_keeps_the_sheet_as_the_first_reference(self):
        rows = _compile_plan(
            _project(),
            _subjects(),
            [
                KeyframeDirection(source_beat_index=1, subject_refs=["Alpha"], scene_refs=["The Room"]),
                KeyframeDirection(source_beat_index=2, subject_refs=["Beta"], scene_refs=["The Room"]),
            ],
            _scenes(),
        )
        prompt = build_storyboard_sheet_revision_prompt(rows, _subjects(), _scenes(), "把开场改成低机位")
        self.assertIn("参考图1是一张已生成的完整连续视频分镜图合集", prompt)
        self.assertIn("把开场改成低机位", prompt)
        self.assertIn("保留用户没有要求修改的剧情", prompt)

    def test_storyboard_confirmation_is_deterministic_but_edits_are_not_submission(self):
        self.assertTrue(_is_explicit_storyboard_submission_request("确认"))
        self.assertTrue(_is_explicit_storyboard_submission_request("确认分镜并开始生成视频"))
        self.assertTrue(_is_explicit_storyboard_submission_request("生成啊开始"))
        self.assertFalse(_is_explicit_storyboard_submission_request("确认，不过把开场改成近景"))


if __name__ == "__main__":
    unittest.main()
