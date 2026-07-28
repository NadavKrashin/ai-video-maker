"""Recognising an uploaded track by content (media/audio_files.py).

The upload path used to trust the filename's extension, and a phone upload
of a real mp3 named "Lights(chosic.com)" was rejected because of it.
"""
from __future__ import annotations

from ai_video_maker.media.audio_files import looks_like_audio


class TestMagicBytes:
    def test_id3_tagged_mp3(self):
        assert looks_like_audio(b"ID3\x04\x00\x00" + b"\x00" * 32)

    def test_bare_mp3_frame_sync(self):
        assert looks_like_audio(b"\xff\xfb\x90\x64" + b"\x00" * 32)

    def test_wav_ogg_flac_and_m4a(self):
        assert looks_like_audio(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        assert looks_like_audio(b"OggS\x00\x02" + b"\x00" * 32)
        assert looks_like_audio(b"fLaC\x00\x00\x00\x22")
        assert looks_like_audio(b"\x00\x00\x00\x20ftypM4A ")

    def test_content_wins_over_a_missing_extension(self):
        # The exact shape of the upload that failed.
        assert looks_like_audio(
            b"ID3\x04\x00\x00" + b"\x00" * 32,
            filename="Lights(chosic.com)",
            content_type="application/octet-stream",
        )


class TestOtherSignals:
    def test_declared_audio_mime(self):
        assert looks_like_audio(b"\x00" * 64, content_type="audio/mpeg")

    def test_video_mime_counts_because_m4a_is_labelled_that_way(self):
        assert looks_like_audio(b"\x00" * 64, content_type="video/mp4")

    def test_mime_with_parameters(self):
        assert looks_like_audio(b"\x00" * 64, content_type="audio/mpeg; charset=binary")

    def test_extension_as_the_last_resort(self):
        assert looks_like_audio(b"\x00" * 64, filename="track.M4A")


class TestRefusals:
    def test_a_pdf_is_not_audio(self):
        assert not looks_like_audio(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3", "invoice.pdf")

    def test_a_png_is_not_audio(self):
        assert not looks_like_audio(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "photo.png")

    def test_plain_text_is_not_audio(self):
        assert not looks_like_audio(b"just some notes\n", "notes.txt")

    def test_empty_upload_is_not_audio(self):
        assert not looks_like_audio(b"", "", "")

    def test_frame_sync_is_not_hunted_deep_into_the_file(self):
        # 0xFF 0xFB appears in plenty of binaries; only the start counts.
        assert not looks_like_audio(b"\x00" * 40 + b"\xff\xfb" + b"\x00" * 40, "x.bin")
