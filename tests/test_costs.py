"""The spending ledger: pricing maths, the append-only log, and the report.

Money is the one thing in this pipeline the user cannot get back, so the
numbers next to it have to be trustworthy: priced from config (never
invented), never lost to a concurrent write, and never able to fail a run
that has already spent the money.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_maker.costs import (
    KIND_CLIP,
    KIND_IMAGE,
    CostLedger,
    Pricing,
    estimate_usd,
    merge_totals,
)
from ai_video_maker.models import Storyboard


class TestPricing:
    def test_kling_clip_prices_match_the_provider_tiers(self):
        pricing = Pricing()
        assert pricing.clip(5) == 0.35
        assert pricing.clip(10) == 0.70

    def test_text_calls_are_priced_from_token_usage(self):
        pricing = Pricing(
            openai_text_input_usd_per_1m=1.0, openai_text_output_usd_per_1m=10.0
        )
        assert pricing.text(1_000_000, 0) == 1.0
        assert pricing.text(0, 100_000) == 1.0

    def test_negative_and_zero_quantities_never_produce_a_charge(self):
        pricing = Pricing()
        assert pricing.clip(-5) == 0.0
        assert pricing.image(0) == 0.0
        assert pricing.text(-10, -10) == 0.0


class TestCostLedger:
    def test_records_survive_a_reload(self, tmp_path: Path):
        path = tmp_path / "costs.json"
        CostLedger(path).record(KIND_CLIP, 0.35, "a_to_b.mp4", seconds=5)
        reloaded = CostLedger(path)
        assert reloaded.summary()["total_usd"] == 0.35
        assert reloaded.entries()[0]["detail"] == "a_to_b.mp4"
        assert reloaded.entries()[0]["seconds"] == 5

    def test_summary_groups_by_kind(self, tmp_path: Path):
        ledger = CostLedger(tmp_path / "costs.json")
        ledger.record(KIND_IMAGE, 0.19, "a.png")
        ledger.record(KIND_IMAGE, 0.19, "b.png")
        ledger.record(KIND_CLIP, 0.70, "a_to_b.mp4")
        summary = ledger.summary()
        assert summary["total_usd"] == 1.08
        assert summary["by_kind"][KIND_IMAGE] == {"usd": 0.38, "count": 2}
        assert summary["entries"] == 3

    def test_free_calls_are_not_recorded(self, tmp_path: Path):
        ledger = CostLedger(tmp_path / "costs.json")
        ledger.record(KIND_IMAGE, 0.0, "nothing")
        ledger.record(KIND_IMAGE, -1.0, "refund?")
        assert ledger.entries() == []

    def test_an_unwritable_ledger_never_raises(self, tmp_path: Path):
        # The ledger's directory is a FILE: writing is impossible. Recording
        # a cost still has to succeed from the caller's point of view — the
        # clip it describes is already rendered and paid for.
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        ledger = CostLedger(blocker / "costs.json")
        ledger.record(KIND_CLIP, 0.35, "a_to_b.mp4")  # must not raise
        # This process still knows what it spent; only the file is lost.
        assert ledger.summary()["total_usd"] == 0.35
        assert not (blocker / "costs.json").exists()

    def test_a_corrupt_ledger_is_treated_as_empty_not_fatal(self, tmp_path: Path):
        path = tmp_path / "costs.json"
        path.write_text("{not json", encoding="utf-8")
        ledger = CostLedger(path)
        assert ledger.entries() == []
        ledger.record(KIND_CLIP, 0.35, "a_to_b.mp4")
        assert ledger.summary()["total_usd"] == 0.35

    def test_empty_summary_has_every_kind(self, tmp_path: Path):
        summary = CostLedger(tmp_path / "costs.json").summary()
        assert summary["total_usd"] == 0.0
        assert set(summary["by_kind"]) == {"image", "text", "clip", "sfx"}


class TestEstimates:
    def test_estimate_prices_the_artifacts_on_disk(self):
        pricing = Pricing()
        # 2 styled frames + one 10s clip with SFX.
        assert estimate_usd(
            pricing, images=2, clip_seconds=10, sfx_clips=1
        ) == round(0.38 + 0.70 + 0.02, 6)

    def test_merge_totals_adds_projects_up(self):
        totals = merge_totals([
            {"total_usd": 1.0, "estimated_usd": 2.0,
             "by_kind": {"clip": {"usd": 1.0, "count": 1}}},
            {"total_usd": 0.5, "estimated_usd": 0.5,
             "by_kind": {"clip": {"usd": 0.5, "count": 2}}},
        ])
        assert totals["total_usd"] == 1.5
        assert totals["estimated_usd"] == 2.5
        assert totals["by_kind"]["clip"] == {"usd": 1.5, "count": 3}
        assert totals["projects"] == 2


def _storyboard_with_clip(workspace, duration: int) -> None:
    Storyboard(
        project_title="t", style="s",
        frames=[
            {"id": "a", "description": "", "image_prompt": "",
             "output_path": "styled_images/a.png"},
            {"id": "b", "description": "", "image_prompt": "",
             "output_path": "styled_images/b.png"},
        ],
        transitions=[{
            "id": "a_to_b", "start_frame": "styled_images/a.png",
            "end_frame": "styled_images/b.png", "motion_prompt": "m",
            "duration": duration, "output_path": "clips/a_to_b.mp4",
        }],
    ).save(workspace.default_storyboard_json)


class TestPipelineCostReport:
    def test_recorded_spend_is_reported_and_grouped(self, pipeline):
        pipeline.costs.record(KIND_CLIP, 0.35, "a_to_b.mp4")
        pipeline.costs.record(KIND_IMAGE, 0.19, "a.png")
        report = pipeline.cost_report()
        assert report["total_usd"] == 0.54
        assert report["currency"] == "USD"

    def test_a_project_that_predates_the_ledger_is_valued_from_disk(
        self, pipeline, workspace
    ):
        # No ledger at all, but a full movie on disk: reporting "$0 spent"
        # would be worse than an approximation, so the estimate leads and is
        # flagged as one.
        _storyboard_with_clip(workspace, 10)
        (workspace.styled_images_dir / "a.png").write_bytes(b"x")
        (workspace.styled_images_dir / "b.png").write_bytes(b"x")
        (workspace.clips_dir / "a_to_b.mp4").write_bytes(b"x")
        report = pipeline.cost_report()
        assert report["total_usd"] == 0.0
        assert report["estimated_usd"] == 1.08  # 2 images + one 10s clip
        assert report["estimated"] is True
        assert report["clip_seconds"] == 10

    def test_a_fully_tracked_project_is_not_flagged_as_estimated(
        self, pipeline, workspace
    ):
        _storyboard_with_clip(workspace, 5)
        (workspace.clips_dir / "a_to_b.mp4").write_bytes(b"x")
        pipeline.costs.record(KIND_CLIP, 0.35, "a_to_b.mp4")
        report = pipeline.cost_report()
        assert report["estimated"] is False

    def test_snapshot_carries_the_cost_report(self, pipeline):
        pipeline.costs.record(KIND_CLIP, 0.35, "a_to_b.mp4")
        assert pipeline.snapshot()["cost"]["total_usd"] == 0.35

    def test_a_dry_run_never_books_money(self, make_pipeline):
        dry = make_pipeline(dry_run=True)
        dry._record_cost(KIND_CLIP, 0.35, "a_to_b.mp4")
        assert dry.costs.entries() == []

    def test_clip_renders_are_booked_at_the_storyboard_duration(
        self, pipeline, workspace
    ):
        start = workspace.styled_images_dir / "a.png"
        end = workspace.styled_images_dir / "b.png"
        for frame in (start, end):
            frame.write_bytes(b"x")
        clip = workspace.clips_dir / "a_to_b.mp4"

        def fake_generate(s, e, motion, duration, dst, reword=None, elements=None):
            dst.write_bytes(b"rendered")

        pipeline.video_client.generate_clip = fake_generate
        pipeline._generate_clips([(start, end, "m", 10, "")], set())
        assert clip.exists()
        entry = pipeline.costs.entries()[0]
        assert entry["kind"] == KIND_CLIP
        assert entry["usd"] == 0.70
        assert entry["seconds"] == 10

    def test_openai_calls_price_themselves_into_the_project(self, pipeline):
        # The client reports its own spending (the runner never sees the
        # condense/restage/reword calls), so the wiring is what matters here.
        class _Usage:
            prompt_tokens = 1_000_000
            completion_tokens = 0

        class _Resp:
            usage = _Usage()

        pipeline.openai._note_text_usage(_Resp(), "analyze")
        assert pipeline.costs.summary()["by_kind"]["text"]["usd"] == 1.25


class TestLedgerFileShape:
    def test_the_file_is_readable_json_with_an_entries_list(self, tmp_path: Path):
        path = tmp_path / "costs.json"
        CostLedger(path).record(KIND_CLIP, 0.35, "a_to_b.mp4")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert list(data) == ["entries"]
        assert data["entries"][0]["kind"] == KIND_CLIP


class TestClipsAreValuedAtTheModelThatRenderedThem:
    """Switching the studio's video model must not re-price old movies."""

    def _project(self, make_pipeline, workspace, **config):
        from ai_video_maker.models import Frame, Storyboard, Transition
        for n in ("a", "b", "c"):
            (workspace.styled_images_dir / f"{n}.png").write_bytes(b"x")
        Storyboard(
            project_title="t", style="s",
            frames=[Frame(id=n, description="", image_prompt="",
                          output_path=f"styled_images/{n}.png")
                    for n in ("a", "b", "c")],
            transitions=[
                Transition(id=f"{x}_to_{y}", start_frame=f"styled_images/{x}.png",
                           end_frame=f"styled_images/{y}.png", motion_prompt="m",
                           duration=10, output_path=f"clips/{x}_to_{y}.mp4")
                for x, y in (("a", "b"), ("b", "c"))
            ],
        ).save(workspace.default_storyboard_json)
        for name in ("a_to_b.mp4", "b_to_c.mp4"):
            (workspace.clips_dir / name).write_bytes(b"clip")
        return make_pipeline()

    def _clip_usd(self, pipeline):
        return sum(
            pipeline._clip_rate(name) * 10 for name in ("a_to_b.mp4", "b_to_c.mp4")
        )

    def test_clips_with_no_recorded_model_are_priced_as_kling_two_five(
        self, make_pipeline, workspace
    ):
        pipeline = self._project(make_pipeline, workspace)
        pipeline.config.pricing.clip_usd_per_second = 0.112  # today's model
        assert self._clip_usd(pipeline) == pytest.approx(1.40)

    def test_a_clip_is_priced_by_the_model_recorded_for_it(
        self, make_pipeline, workspace
    ):
        pipeline = self._project(make_pipeline, workspace)
        pipeline.state.set(
            "clip:a_to_b.mp4", "done",
            model="fal-ai/kling-video/v3/pro/image-to-video",
        )
        assert pipeline._clip_rate("a_to_b.mp4") == pytest.approx(0.112)
        assert pipeline._clip_rate("b_to_c.mp4") == pytest.approx(0.07)

    def test_an_unknown_recorded_model_uses_the_configured_price(
        self, make_pipeline, workspace
    ):
        pipeline = self._project(make_pipeline, workspace)
        pipeline.config.pricing.clip_usd_per_second = 0.5
        pipeline.state.set("clip:a_to_b.mp4", "done", model="someone/new-model")
        assert pipeline._clip_rate("a_to_b.mp4") == pytest.approx(0.5)
