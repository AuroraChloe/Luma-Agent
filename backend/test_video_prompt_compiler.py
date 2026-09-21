import unittest

from video_prompt_compiler import (
    PictureReference,
    TimelineCue,
    VideoPromptSpec,
    compile_h3_video_prompt,
    reference_urls,
)


def _spec(**overrides):
    values = {
        "title": "test clip",
        "duration": 5,
        "references": (
            PictureReference("https://example.com/start.png", "opening frame", "start here"),
            PictureReference("https://example.com/actor.png", "actor identity", "keep identity"),
        ),
        "visual_baseline": "cinematic horizontal animation",
        "opening_state": "use <Picture 1> as the exact opening",
        "ending_state": "the actor from <Picture 2> reaches the door",
        "timeline": (
            TimelineCue(0, 2, "wide", "slow dolly", "actor left", "looks up"),
            TimelineCue(2, 5, "medium", "track right", "actor crosses", "opens door"),
        ),
    }
    values.update(overrides)
    return VideoPromptSpec(**values)


class VideoPromptCompilerTest(unittest.TestCase):
    def test_prompt_tags_match_reference_order(self):
        spec = _spec()
        prompt = compile_h3_video_prompt(spec)
        self.assertIn("<Picture 1>：opening frame", prompt)
        self.assertIn("<Picture 2>：actor identity", prompt)
        self.assertEqual(
            reference_urls(spec),
            ["https://example.com/start.png", "https://example.com/actor.png"],
        )

    def test_timeline_must_cover_the_whole_clip(self):
        with self.assertRaisesRegex(ValueError, "timeline must be continuous"):
            compile_h3_video_prompt(
                _spec(timeline=(TimelineCue(0, 2, "wide", "static", "left", "wait"), TimelineCue(3, 5, "wide", "static", "left", "wait")))
            )

    def test_duration_obeys_gateway_limit(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 15"):
            compile_h3_video_prompt(_spec(duration=16))

    def test_reference_urls_must_be_https(self):
        with self.assertRaisesRegex(ValueError, "public HTTPS"):
            compile_h3_video_prompt(
                _spec(references=(PictureReference("http://example.com/a.png", "bad", "bad"),))
            )


if __name__ == "__main__":
    unittest.main()
