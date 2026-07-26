"""Fetching the music bed from a URL (direct file or extractable page).

No network: requests.get and yt_dlp are stubbed. What matters here is the
routing decision, the guard rails (scheme, size cap, empty response) and
that both paths land on the same destination an upload writes to.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_video_maker.errors import PipelineError
from ai_video_maker.media import music_url as mu


class _FakeResponse:
    def __init__(self, chunks: list[bytes], headers: dict | None = None) -> None:
        self._chunks = chunks
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 0):
        yield from self._chunks


class TestUrlValidation:
    @pytest.mark.parametrize("bad", [
        "", "   ", "not a url", "ftp://example.com/a.mp3",
        "file:///etc/passwd", "javascript:alert(1)", "//example.com/a.mp3",
    ])
    def test_non_http_urls_are_rejected(self, bad, tmp_path):
        with pytest.raises(PipelineError, match="Not a usable music URL"):
            mu.fetch_music(bad, tmp_path / "out.mp3")

    def test_http_and_https_are_accepted(self):
        assert mu._validate("http://x.test/a.mp3") == "http://x.test/a.mp3"
        assert mu._validate(" https://x.test/a.mp3 ") == "https://x.test/a.mp3"


class TestRouting:
    @pytest.mark.parametrize("url,direct", [
        ("https://x.test/track.mp3", True),
        ("https://x.test/track.WAV", True),
        ("https://x.test/a/b/song.flac", True),
        ("https://x.test/track.mp3?token=1", True),   # query ignored
        ("https://www.youtube.com/watch?v=abc123", False),
        ("https://x.test/some/page", False),
        ("https://x.test/video.mp4", False),          # not an audio suffix
    ])
    def test_direct_audio_detection(self, url, direct):
        assert mu.looks_like_direct_audio(url) is direct

    def test_page_url_goes_to_the_extractor(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(mu, "_extract_audio",
                            lambda url, dst: seen.update(url=url) or dst.write_bytes(b"x"))
        dst = tmp_path / "music_custom.mp3"
        mu.fetch_music("https://www.youtube.com/watch?v=abc", dst)
        assert seen["url"] == "https://www.youtube.com/watch?v=abc"

    def test_direct_url_skips_the_extractor(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mu, "_extract_audio",
                            lambda *a: pytest.fail("must not extract"))
        monkeypatch.setattr(mu.requests, "get",
                            lambda *a, **k: _FakeResponse([b"audio-bytes"]))
        dst = tmp_path / "music_custom.mp3"
        assert mu.fetch_music("https://x.test/t.mp3", dst) == dst
        assert dst.read_bytes() == b"audio-bytes"


class TestDirectDownloadGuards:
    def _get(self, monkeypatch, chunks, headers=None):
        monkeypatch.setattr(mu.requests, "get",
                            lambda *a, **k: _FakeResponse(chunks, headers))

    def test_declared_size_over_the_cap_is_refused(self, tmp_path, monkeypatch):
        self._get(monkeypatch, [b"x"], {"Content-Length": str(mu._MAX_BYTES + 1)})
        with pytest.raises(PipelineError, match="the limit is"):
            mu.fetch_music("https://x.test/t.mp3", tmp_path / "o.mp3")

    def test_a_lying_server_is_still_capped_mid_stream(self, tmp_path, monkeypatch):
        # No Content-Length, then far more bytes than allowed.
        chunk = b"y" * (1024 * 1024)
        self._get(monkeypatch, [chunk] * (mu._MAX_BYTES // len(chunk) + 2))
        dst = tmp_path / "o.mp3"
        with pytest.raises(PipelineError, match="exceeded"):
            mu.fetch_music("https://x.test/t.mp3", dst)
        assert not dst.exists()  # the partial file is cleaned up

    def test_empty_response_is_an_error_not_a_silent_zero_byte_track(
        self, tmp_path, monkeypatch
    ):
        self._get(monkeypatch, [])
        dst = tmp_path / "o.mp3"
        with pytest.raises(PipelineError, match="Nothing was downloaded"):
            mu.fetch_music("https://x.test/t.mp3", dst)
        assert not dst.exists()

    def test_parent_directory_is_created(self, tmp_path, monkeypatch):
        self._get(monkeypatch, [b"audio"])
        dst = tmp_path / "nested" / "output" / "music_custom.mp3"
        mu.fetch_music("https://x.test/t.mp3", dst)
        assert dst.read_bytes() == b"audio"


class TestExtractorFailures:
    def test_extractor_error_is_actionable(self, tmp_path, monkeypatch):
        class _Boom:
            def __init__(self, *_a, **_k): pass
            def __enter__(self): return self
            def __exit__(self, *_e): return False
            def download(self, _urls): raise RuntimeError("player changed")

        import yt_dlp
        monkeypatch.setattr(yt_dlp, "YoutubeDL", _Boom)
        with pytest.raises(PipelineError, match="pip install -U yt-dlp"):
            mu.fetch_music("https://www.youtube.com/watch?v=x", tmp_path / "o.mp3")

    def test_extractor_producing_nothing_is_reported(self, tmp_path, monkeypatch):
        class _Silent:
            def __init__(self, *_a, **_k): pass
            def __enter__(self): return self
            def __exit__(self, *_e): return False
            def download(self, _urls): return 0  # writes no file

        import yt_dlp
        monkeypatch.setattr(yt_dlp, "YoutubeDL", _Silent)
        with pytest.raises(PipelineError, match="No audio track came back"):
            mu.fetch_music("https://www.youtube.com/watch?v=x", tmp_path / "o.mp3")


class TestPipelineIntegration:
    """--music-url lands in the custom slot, ahead of every other source."""

    def test_music_url_wins_and_is_stored_as_the_custom_track(
        self, make_pipeline, workspace, monkeypatch
    ):
        calls = {}

        def fake_fetch(url, dst):
            calls["url"], calls["dst"] = url, dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"fetched")
            return dst

        monkeypatch.setattr("ai_video_maker.runner.fetch_music", fake_fetch)
        (workspace.output_dir).mkdir(parents=True, exist_ok=True)
        workspace.music_file.write_bytes(b"older")
        p = make_pipeline(music_url="https://x.test/t.mp3")
        assert p._resolve_music_file() == workspace.custom_music_file
        assert calls["url"] == "https://x.test/t.mp3"
        # Stored where an upload goes, so later runs reuse it without re-fetching.
        assert calls["dst"] == workspace.custom_music_file
        assert workspace.custom_music_file.read_bytes() == b"fetched"

    def test_without_music_url_nothing_is_fetched(self, make_pipeline, workspace):
        p = make_pipeline()
        assert p._resolve_music_file() is None
