import video_scene_runtime as runtime


def test_scene_prompt_is_environment_only_and_reusable():
    prompt = runtime._scene_prompt(
        {
            "display_name": "Clock Tower",
            "setting": "An abandoned clock tower full of drawings",
            "visual_direction": "cold violet gaslight",
            "continuity_notes": "the desk stays by the north wall",
        },
        {"visual_direction": "painted noir"},
    )
    assert "Clock Tower" in prompt
    assert "严禁出现任何人物" in prompt
    assert "1536x1024" in prompt


def test_scene_resolution_prefers_explicit_identity_then_ordinal():
    scenes = [
        {"scene_id": "scene_a", "ordinal": 1, "display_name": "Clock Tower"},
        {"scene_id": "scene_b", "ordinal": 2, "display_name": "Rain Street"},
    ]
    assert runtime._resolve_scene(scenes, "Clock Tower") == scenes[0]
    assert runtime._resolve_scene(scenes, "scene 2") == scenes[1]
