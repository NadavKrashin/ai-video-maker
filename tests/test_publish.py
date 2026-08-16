"""Publishing the finished movie back into the customer's Cloudinary folder.

The one invariant worth testing hard: publishing is ADDITIVE. It must never
pick a version number that was used before — not after a hand-deleted version,
not when one of Cloudinary's two listings comes back empty, and not when the
local record and the cloud disagree — because the version number is the only
thing standing between a delivery and an overwrite.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import requests

from ai_video_maker.clients.cloudinary_client import (
    CloudinaryClient,
    PublishedVideo,
    _raise_for_status,
    chunk_ranges,
    cloudinary_error_detail,
    next_publish_version,
    publish_public_id,
    publish_version,
    size_rejection_hint,
    upload_signature,
)
from ai_video_maker.errors import PipelineError
from ai_video_maker.retry import is_retryable_error
from ai_video_maker.intake import write_order_record
from ai_video_maker.publish import (
    latest_publication,
    publish_state,
    published_versions,
    read_publications,
    record_publication,
)

FOLDER = "AM-280726-XY12_Dana-Cohen-28.07.2026_10-30"


def _final_video(workspace, data: bytes = b"movie") -> Path:
    path = workspace.final_video
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _ordered(workspace, folder: str = FOLDER) -> None:
    write_order_record(workspace.order_file, order_folder=folder, photo_count=3)


class TestPublishNaming:
    def test_version_is_read_from_the_public_id_leaf(self):
        assert publish_version(f"video-orders/{FOLDER}/final_v3", "final") == 3
        # Dynamic-folder mode: the public_id may be the bare name.
        assert publish_version("final_v11", "final") == 11

    def test_other_assets_in_the_folder_are_not_ours(self):
        assert publish_version(f"video-orders/{FOLDER}/1", "final") is None
        assert publish_version(f"video-orders/{FOLDER}/final", "final") is None
        assert publish_version(f"video-orders/{FOLDER}/final_v2_copy", "final") is None

    def test_first_publish_is_v1(self):
        assert next_publish_version([]) == 1

    def test_next_version_is_past_the_highest_never_a_count(self):
        # v2 deleted by hand in the console: the next publish must still be v4,
        # or it would land on a name that already existed.
        assert next_publish_version([1, 3]) == 4

    def test_public_id_lives_inside_the_order_folder(self):
        assert publish_public_id("video-orders", FOLDER, "final", 2) == (
            f"video-orders/{FOLDER}/final_v2"
        )


class TestUploadSignature:
    def test_signs_sorted_params_with_the_secret(self):
        params = {"public_id": "a/b", "timestamp": "1700000000", "overwrite": "false"}
        expected = hashlib.sha1(
            b"overwrite=false&public_id=a/b&timestamp=1700000000secret"
        ).hexdigest()
        assert upload_signature(params, "secret") == expected

    def test_transport_params_are_never_signed(self):
        signed = {"timestamp": "1", "public_id": "x"}
        noisy = dict(signed, api_key="k", file="@movie.mp4", resource_type="video")
        assert upload_signature(noisy, "s") == upload_signature(signed, "s")


class TestChunkRanges:
    def test_small_file_is_one_inclusive_range(self):
        assert chunk_ranges(10, chunk_size=100) == [(0, 9)]

    def test_ranges_are_contiguous_and_cover_everything(self):
        ranges = chunk_ranges(250, chunk_size=100)
        assert ranges == [(0, 99), (100, 199), (200, 249)]
        assert sum(end - start + 1 for start, end in ranges) == 250

    def test_empty_file_still_makes_one_request(self):
        assert chunk_ranges(0, chunk_size=100) == [(0, 0)]


class TestPublicationRecord:
    def test_missing_or_broken_file_reads_as_nothing_published(self, tmp_path):
        assert read_publications(tmp_path / "none.json") == []
        broken = tmp_path / "published.json"
        broken.write_text("{oops", encoding="utf-8")
        assert read_publications(broken) == []

    def test_records_append_and_never_rewrite_history(self, tmp_path, workspace):
        path = tmp_path / "published.json"
        video = _final_video(workspace)
        record_publication(
            path, order_folder=FOLDER, public_id="p/final_v1",
            url="https://x/final_v1.mp4", version=1, video=video,
        )
        record_publication(
            path, order_folder=FOLDER, public_id="p/final_v2",
            url="https://x/final_v2.mp4", version=2, video=video,
        )
        entries = read_publications(path)
        assert [e["version"] for e in entries] == [1, 2]
        assert entries[0]["public_id"] == "p/final_v1"
        assert published_versions(path) == [1, 2]
        assert latest_publication(path)["version"] == 2

    def test_state_notices_the_movie_was_rebuilt(self, tmp_path, workspace):
        path = tmp_path / "published.json"
        video = _final_video(workspace, b"first cut")
        record_publication(
            path, order_folder=FOLDER, public_id="p/final_v1", url="u",
            version=1, video=video,
        )
        state = publish_state(path, video)
        assert state["count"] == 1 and state["changed_since"] is False
        assert state["latest"]["version"] == 1

        video.write_bytes(b"a different, longer cut")  # re-combined
        assert publish_state(path, video)["changed_since"] is True

    def test_nothing_published_yet(self, tmp_path, workspace):
        state = publish_state(tmp_path / "none.json", _final_video(workspace))
        assert state == {
            "count": 0, "latest": None, "versions": [], "changed_since": False,
        }


class _FakeResponse:
    """Just enough of requests.Response for raise_for_status()."""

    def __init__(self, status_code: int, text: str, payload=None) -> None:
        self.status_code = status_code
        self.text = text
        self.reason = "Bad Request" if status_code >= 400 else "OK"
        self.url = "https://api.cloudinary.com/v1_1/test/video/upload"
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: {self.reason} for url: {self.url}",
                response=self,
            )


def _cloudinary_error(message: str) -> _FakeResponse:
    return _FakeResponse(400, json.dumps({"error": {"message": message}}))


_TOO_LARGE = "File size too large. Got 523239424. Maximum is 104857600."


class TestUploadFailuresExplainThemselves:
    """A failed publish must say WHY, in Cloudinary's own words.

    A real 499 MB delivery died as a bare "400 Client Error: Bad Request for
    url: …/video/upload" on chunk 1 of 25 — requests' default message, with
    Cloudinary's actual sentence dropped on the floor along with the body.
    """

    def test_the_message_is_pulled_out_of_the_error_body(self):
        assert cloudinary_error_detail(
            json.dumps({"error": {"message": _TOO_LARGE}})
        ) == _TOO_LARGE

    def test_a_non_json_body_survives_as_a_one_line_excerpt(self):
        detail = cloudinary_error_detail("<html>\n  <body>Gateway  Timeout</body>\n")
        assert "Gateway Timeout" in detail and "\n" not in detail

    def test_nothing_to_add_stays_empty(self):
        assert cloudinary_error_detail("") == ""
        assert cloudinary_error_detail("{not json") != ""  # excerpt, not a crash

    def test_the_reraised_error_keeps_its_status_so_it_is_not_retried(self):
        with pytest.raises(requests.HTTPError) as caught:
            _raise_for_status(_cloudinary_error(_TOO_LARGE))
        exc = caught.value
        assert _TOO_LARGE in str(exc)
        assert exc.response.status_code == 400
        # The point of keeping .response: a 400 must stay permanent, or every
        # chunk would be re-sent (20 MB a time) to be refused again.
        assert is_retryable_error(exc) is False

    def test_a_body_free_failure_is_left_exactly_as_requests_raised_it(self):
        with pytest.raises(requests.HTTPError) as caught:
            _raise_for_status(_FakeResponse(500, ""))
        assert "500 Client Error" in str(caught.value) or "500" in str(caught.value)


class TestSizeRejectionsAreTranslated:
    """Chunking beats the per-request cap, not the account's file-size limit."""

    def test_only_a_size_rejection_gets_the_hint(self):
        assert size_rejection_hint(_TOO_LARGE, 523239424)
        assert size_rejection_hint("Invalid signature", 523239424) == ""

    def test_the_hint_names_the_whole_movie_not_the_chunk(self):
        hint = size_rejection_hint(_TOO_LARGE, 499 * 1024 * 1024)
        assert "499 MB" in hint
        assert "chunk" in hint.lower()

    def test_the_upload_reports_it_instead_of_a_bare_400(self, tmp_path, monkeypatch):
        movie = tmp_path / "final_video.mp4"
        movie.write_bytes(b"x" * 1024)
        calls: list[int] = []

        def _refuse(*args, **kwargs):
            calls.append(1)
            return _cloudinary_error(_TOO_LARGE)

        monkeypatch.setattr(requests, "post", _refuse)
        # Keep the run offline: publishing reads the cloud's folder mode first.
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: _FakeResponse(200, "{}", {"settings": {"folder_mode": "fixed"}}),
        )
        client = CloudinaryClient("cloud", "key", "secret", max_retries=5, base_delay=0)

        with pytest.raises(PipelineError) as caught:
            client.publish_final_video(FOLDER, movie, 1)

        message = str(caught.value)
        assert _TOO_LARGE in message          # Cloudinary's own sentence
        assert "maximum video file size" in message  # ...and what to do about it
        assert calls == [1]                   # a 400 is permanent: uploaded once


class TestOrderPhotosAreFoundHoweverTheyWereFiled:
    """Three routes into an order folder, each invisible to the other two.

    A real order's folder was recreated in the Cloudinary console; the photos
    inside carried neither the frontend's tag nor the folder in their
    public_id, so both existing lookups returned nothing and ingest reported
    "contains no images" for a folder that was plainly full. Dynamic-folder
    mode files those assets by ``asset_folder`` alone.
    """

    LEAF = "AM-160826-VKXQ_Kfir-Daniel-16.08.2026_11-23_mood"

    def _client(self, monkeypatch, pages: dict[str, list], not_found=()):
        """Serve Admin-API listings by path; record which were asked for.

        A listing with no hits answers 200 with an empty ``resources`` array —
        which is what the real order did, since ingest reported "contains no
        images" rather than failing. `not_found` covers the endpoints that
        answer 404 instead (an asset folder that does not exist).
        """
        asked: list[str] = []

        def _get(url, params=None, **kwargs):
            path = url.split("/v1_1/cloud/", 1)[1]
            asked.append(path)
            if path in not_found:
                return _FakeResponse(404, '{"error":{"message":"not found"}}')
            return _FakeResponse(200, "{}", {"resources": pages.get(path, [])})

        monkeypatch.setattr(requests, "get", _get)
        client = CloudinaryClient("cloud", "key", "secret", base_delay=0)
        return client, asked

    @staticmethod
    def _photo(public_id: str, order: str) -> dict:
        return {
            "public_id": public_id, "format": "jpg",
            "secure_url": f"https://res.cloudinary.com/{public_id}.jpg",
            "context": {"custom": {"order": order}},
        }

    def test_assets_filed_only_by_asset_folder_are_found(self, monkeypatch):
        # Console-uploaded: bare public_ids, no tag, no path.
        client, asked = self._client(monkeypatch, {
            "resources/by_asset_folder": [
                self._photo("kfir2", "2"), self._photo("kfir1", "1"),
            ],
        })
        assets = client.list_order_assets(self.LEAF)
        assert [a.position for a in assets] == [1, 2]  # still movie order
        assert asked[-1] == "resources/by_asset_folder"

    def test_the_tag_listing_still_wins_and_stops_there(self, monkeypatch):
        client, asked = self._client(monkeypatch, {
            f"resources/image/tags/{requests.utils.quote(self.LEAF, safe='')}":
                [self._photo("video-orders/x/1", "1")],
        })
        assert len(client.list_order_assets(self.LEAF)) == 1
        assert "resources/by_asset_folder" not in asked

    def test_an_empty_folder_is_still_empty_not_an_error(self, monkeypatch):
        # All three come back empty — the honest answer is "no photos", which
        # ingest turns into a message about uploads still being pending.
        client, asked = self._client(monkeypatch, {})
        assert client.list_order_assets(self.LEAF) == []
        assert "resources/by_asset_folder" in asked  # all three were tried

    def test_a_folder_cloudinary_has_never_heard_of_is_not_a_crash(
        self, monkeypatch
    ):
        # An asset folder that does not exist answers 404, not an empty list.
        client, _ = self._client(
            monkeypatch, {}, not_found={"resources/by_asset_folder"}
        )
        assert client.list_order_assets(self.LEAF) == []

    def test_that_404_is_not_logged_as_a_failure(self, monkeypatch, caplog):
        """An order with no photos yet is normal, and must read as normal.

        Absorbing the 404 by raising and catching it made the retry loop
        announce "failed with a permanent error (no retry)" for every order
        whose folder does not exist — on every panel refresh, for every such
        order, burying the errors that matter.
        """
        client, _ = self._client(
            monkeypatch, {}, not_found={"resources/by_asset_folder"}
        )
        with caplog.at_level("WARNING"):
            assert client.list_order_assets(self.LEAF) == []
        assert not [r for r in caplog.records if "permanent error" in r.message]


class _FakeCloudinary:
    """A CloudinaryClient stand-in: records the upload, invents no network."""

    orders_folder = "video-orders"
    publish_basename = "final"

    def __init__(self, published: list[int] | None = None) -> None:
        self._published = published or []
        self.uploaded: list[tuple[str, Path, int]] = []

    def list_published_videos(self, folder_leaf: str) -> list[PublishedVideo]:
        return [
            PublishedVideo(
                public_id=publish_public_id("video-orders", folder_leaf, "final", v),
                url=f"https://res.cloudinary.com/final_v{v}.mp4", version=v,
            )
            for v in self._published
        ]

    def publish_final_video(self, folder_leaf, path, version) -> PublishedVideo:
        self.uploaded.append((folder_leaf, path, version))
        public_id = publish_public_id("video-orders", folder_leaf, "final", version)
        return PublishedVideo(
            public_id=public_id, url=f"https://res.cloudinary.com/{public_id}.mp4",
            version=version, bytes=path.stat().st_size,
        )


class TestPublishPlan:
    def test_first_publish_of_an_ingested_project(self, pipeline, workspace):
        _ordered(workspace)
        _final_video(workspace)
        plan = pipeline.publish_plan(_FakeCloudinary())
        assert plan["version"] == 1
        assert plan["public_id"] == f"video-orders/{FOLDER}/final_v1"
        assert plan["filename"] == "final_v1.mp4"
        assert plan["order_folder"] == FOLDER and plan["final_video"] is True

    def test_next_version_follows_what_cloudinary_already_holds(
        self, pipeline, workspace
    ):
        _ordered(workspace)
        _final_video(workspace)
        plan = pipeline.publish_plan(_FakeCloudinary([1, 2]))
        assert plan["version"] == 3
        assert [p["version"] for p in plan["published"]] == [1, 2]

    def test_local_history_is_merged_in_when_cloudinary_forgets(
        self, pipeline, workspace
    ):
        # A listing that comes back empty (tag removed, permissions, an
        # outage) must not make the next publish reuse v1 and risk the
        # already-delivered movie.
        _ordered(workspace)
        video = _final_video(workspace)
        record_publication(
            workspace.published_file, order_folder=FOLDER,
            public_id=f"video-orders/{FOLDER}/final_v1", url="u", version=1,
            video=video,
        )
        assert pipeline.publish_plan(_FakeCloudinary([]))["version"] == 2

    def test_a_project_without_an_order_cannot_be_published(self, pipeline, workspace):
        _final_video(workspace)
        with pytest.raises(PipelineError, match="isn't tied to a Cloudinary order"):
            pipeline.publish_plan(_FakeCloudinary())


class TestCmdPublish:
    def _pipeline_with_confirm(self, make_pipeline, answer: bool, **options):
        pipeline = make_pipeline(**options)
        asked: list[list[str]] = []

        def confirm(lines, question):
            asked.append(lines + [question])
            return answer

        pipeline.confirm = confirm
        return pipeline, asked

    def test_confirmation_names_the_exact_file(self, make_pipeline, workspace,
                                               monkeypatch):
        _ordered(workspace)
        _final_video(workspace)
        fake = _FakeCloudinary([1])
        monkeypatch.setattr(CloudinaryClient, "from_config", staticmethod(lambda c: fake))
        pipeline, asked = self._pipeline_with_confirm(make_pipeline, True)

        pipeline.cmd_publish()

        shown = "\n".join(asked[0])
        assert f"video-orders/{FOLDER}/final_v2" in shown
        assert "Nothing already in Cloudinary is replaced or deleted." in shown
        assert fake.uploaded == [(FOLDER, workspace.final_video, 2)]
        assert published_versions(workspace.published_file) == [2]

    def test_declining_uploads_nothing(self, make_pipeline, workspace, monkeypatch):
        _ordered(workspace)
        _final_video(workspace)
        fake = _FakeCloudinary()
        monkeypatch.setattr(CloudinaryClient, "from_config", staticmethod(lambda c: fake))
        pipeline, _ = self._pipeline_with_confirm(make_pipeline, False)

        pipeline.cmd_publish()

        assert fake.uploaded == []
        assert not workspace.published_file.exists()

    def test_dry_run_never_uploads(self, make_pipeline, workspace, monkeypatch):
        _ordered(workspace)
        _final_video(workspace)
        fake = _FakeCloudinary()
        monkeypatch.setattr(CloudinaryClient, "from_config", staticmethod(lambda c: fake))
        pipeline, _ = self._pipeline_with_confirm(make_pipeline, True, dry_run=True)

        pipeline.cmd_publish()

        assert fake.uploaded == []

    def test_no_final_video_is_a_clear_error(self, make_pipeline, workspace):
        _ordered(workspace)
        pipeline, _ = self._pipeline_with_confirm(make_pipeline, True)
        with pytest.raises(PipelineError, match="No final video to publish"):
            pipeline.cmd_publish()

    def test_an_approved_name_that_went_stale_stops_the_upload(
        self, make_pipeline, workspace, monkeypatch
    ):
        # The panel showed final_v2 for approval; by the time the job ran,
        # someone had published v2 from elsewhere. Uploading under any other
        # name than the approved one is not what was agreed to.
        _ordered(workspace)
        _final_video(workspace)
        fake = _FakeCloudinary([1, 2])
        monkeypatch.setattr(CloudinaryClient, "from_config", staticmethod(lambda c: fake))
        pipeline, _ = self._pipeline_with_confirm(
            make_pipeline, True, publish_as=f"video-orders/{FOLDER}/final_v2"
        )
        with pytest.raises(PipelineError, match="no longer the next free"):
            pipeline.cmd_publish()
        assert fake.uploaded == []

    def test_the_approved_name_is_the_one_uploaded(self, make_pipeline, workspace,
                                                   monkeypatch):
        _ordered(workspace)
        _final_video(workspace)
        fake = _FakeCloudinary([1])
        monkeypatch.setattr(CloudinaryClient, "from_config", staticmethod(lambda c: fake))
        pipeline, _ = self._pipeline_with_confirm(
            make_pipeline, True, publish_as=f"video-orders/{FOLDER}/final_v2"
        )
        pipeline.cmd_publish()
        assert fake.uploaded == [(FOLDER, workspace.final_video, 2)]

    def test_record_is_written_for_the_panel(self, make_pipeline, workspace,
                                             monkeypatch):
        _ordered(workspace)
        _final_video(workspace)
        monkeypatch.setattr(
            CloudinaryClient, "from_config", staticmethod(lambda c: _FakeCloudinary())
        )
        pipeline, _ = self._pipeline_with_confirm(make_pipeline, True)
        pipeline.cmd_publish()

        entry = json.loads(workspace.published_file.read_text())["publications"][0]
        assert entry["version"] == 1
        assert entry["order_folder"] == FOLDER
        assert entry["url"].endswith("final_v1.mp4")


class TestDeliveredCopiesAreKeptLocally:
    """Every published version is archived as output/published/final_vN.mp4.

    output/final_video.mp4 is rebuilt IN PLACE by the next combine, so without
    this copy the bytes a customer was actually sent survive only in
    Cloudinary — there is no way to see, re-send or compare an earlier cut.
    """

    def _publish(self, make_pipeline, workspace, monkeypatch, versions=(), **options):
        monkeypatch.setattr(
            CloudinaryClient, "from_config",
            staticmethod(lambda c: _FakeCloudinary(list(versions))),
        )
        pipeline = make_pipeline(**options)
        pipeline.confirm = lambda lines, question: True
        pipeline.cmd_publish()
        return pipeline

    def test_the_delivered_bytes_are_copied_next_to_the_movie(
        self, make_pipeline, workspace, monkeypatch
    ):
        _ordered(workspace)
        _final_video(workspace, b"the first cut")
        self._publish(make_pipeline, workspace, monkeypatch)

        copy = workspace.published_dir / "final_v1.mp4"
        assert copy.read_bytes() == b"the first cut"
        entry = read_publications(workspace.published_file)[0]
        assert entry["local_file"] == "output/published/final_v1.mp4"

    def test_each_version_is_kept_side_by_side(
        self, make_pipeline, workspace, monkeypatch
    ):
        _ordered(workspace)
        _final_video(workspace, b"the first cut")
        self._publish(make_pipeline, workspace, monkeypatch)
        # A fix, a re-combine (which REPLACES final_video.mp4), another publish.
        _final_video(workspace, b"the second cut, after a fix")
        self._publish(make_pipeline, workspace, monkeypatch, versions=[1])

        assert (workspace.published_dir / "final_v1.mp4").read_bytes() == b"the first cut"
        assert (workspace.published_dir / "final_v2.mp4").read_bytes() == (
            b"the second cut, after a fix"
        )

    def test_an_existing_copy_is_never_overwritten(
        self, make_pipeline, workspace, monkeypatch
    ):
        _ordered(workspace)
        _final_video(workspace, b"new bytes")
        leftover = workspace.published_dir / "final_v1.mp4"
        leftover.parent.mkdir(parents=True, exist_ok=True)
        leftover.write_bytes(b"a leftover from a half-finished publish")
        self._publish(make_pipeline, workspace, monkeypatch)
        assert leftover.read_bytes() == b"a leftover from a half-finished publish"

    def test_a_failed_copy_never_sinks_a_delivery_that_happened(
        self, make_pipeline, workspace, monkeypatch
    ):
        # Disk full at exactly the wrong moment: the movie IS in the customer's
        # folder, so the publish must be recorded — only the archive is missing.
        _ordered(workspace)
        _final_video(workspace)

        def _no_space(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("ai_video_maker.runner.shutil.copy2", _no_space)
        self._publish(make_pipeline, workspace, monkeypatch)

        entry = read_publications(workspace.published_file)[0]
        assert entry["version"] == 1 and entry["local_file"] == ""

    def test_archiving_can_be_switched_off(self, make_pipeline, workspace, monkeypatch,
                                           config):
        _ordered(workspace)
        _final_video(workspace)
        object.__setattr__(config, "publish_keep_local_copy", False)
        self._publish(make_pipeline, workspace, monkeypatch)

        assert not workspace.published_dir.exists()
        assert read_publications(workspace.published_file)[0]["local_file"] == ""


class TestSnapshotPublished:
    def test_hand_made_projects_are_not_publishable(self, pipeline):
        published = pipeline.snapshot()["published"]
        assert published["publishable"] is False and published["count"] == 0

    def test_a_finished_order_asks_to_be_published(self, pipeline, workspace):
        _ordered(workspace)
        _final_video(workspace)
        workspace.default_storyboard_json.parent.mkdir(parents=True, exist_ok=True)
        snap = pipeline.snapshot()
        assert snap["published"]["publishable"] is True
        # No storyboard on disk here, so next_step still points at the start of
        # the pipeline — publish only becomes the next step once the movie is
        # actually built (covered below).
        assert snap["next_step"] == "storyboard"

    def test_published_and_unchanged_means_nothing_left_to_do(
        self, pipeline, workspace, monkeypatch
    ):
        _ordered(workspace)
        video = _final_video(workspace)
        record_publication(
            workspace.published_file, order_folder=FOLDER,
            public_id="p/final_v1", url="u", version=1, video=video,
        )
        snap = pipeline.snapshot()
        assert snap["published"]["latest"]["version"] == 1
        assert snap["published"]["changed_since"] is False
