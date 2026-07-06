"""Per-movie workspace paths.

Every per-movie artifact (input/frames/clips/output/state/logs) lives under a
single base directory, so separate movies never collide. This replaces what used
to be a set of module-level globals mutated in place at startup: paths are now
explicit, immutable, and passed around, which is what an API serving several
projects concurrently needs.

`.env` and config.json are NOT per-movie — they stay shared at the repo root
(`PROJECT_ROOT`), and the CLI/API resolve them there.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import InvalidProjectName

# Repo root = parent of this package directory. Shared (non per-movie) files
# such as .env and config.json live here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = PROJECT_ROOT / "projects"


@dataclass(frozen=True)
class Workspace:
    """All per-movie paths, derived from a single base directory (`root`)."""

    root: Path

    @classmethod
    def for_project(cls, name: str) -> "Workspace":
        """Build the workspace for ``projects/<name>/``, validating the name.

        A name is required — there is no default project. It must be a single
        safe path segment (non-empty, no slashes, not "." / "..").
        """
        cleaned = (name or "").strip().strip("/")
        if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
            raise InvalidProjectName(f"Invalid project name: {name!r}")
        return cls(PROJECTS_DIR / cleaned)

    # --- directories ------------------------------------------------------- #
    @property
    def input_images_dir(self) -> Path:
        return self.root / "input_images"

    @property
    def generated_frames_dir(self) -> Path:
        return self.root / "generated_frames"

    @property
    def styled_images_dir(self) -> Path:
        return self.root / "styled_images"

    @property
    def storyboard_dir(self) -> Path:
        return self.root / "storyboard"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def failed_jobs_dir(self) -> Path:
        return self.root / "failed_jobs"

    # --- files ------------------------------------------------------------- #
    @property
    def final_video(self) -> Path:
        return self.output_dir / "final_video.mp4"

    @property
    def music_file(self) -> Path:
        return self.output_dir / "music.mp3"

    @property
    def state_file(self) -> Path:
        return self.logs_dir / "state.json"

    @property
    def failed_jobs_file(self) -> Path:
        return self.failed_jobs_dir / "failed_jobs.json"

    @property
    def default_storyboard_json(self) -> Path:
        return self.storyboard_dir / "storyboard.json"

    @property
    def storyboard_md(self) -> Path:
        return self.storyboard_dir / "storyboard.md"

    @property
    def storyboard_preview(self) -> Path:
        return self.storyboard_dir / "preview.html"

    # --- helpers ----------------------------------------------------------- #
    @property
    def all_dirs(self) -> list[Path]:
        return [
            self.input_images_dir,
            self.generated_frames_dir,
            self.styled_images_dir,
            self.storyboard_dir,
            self.clips_dir,
            self.output_dir,
            self.logs_dir,
            self.failed_jobs_dir,
        ]

    def mkdirs(self) -> None:
        """Create every per-movie directory (idempotent)."""
        for d in self.all_dirs:
            d.mkdir(parents=True, exist_ok=True)
