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

def lessons_file() -> Path:
    """Where the studio's learned planning rules live (see feedback.py).

    Studio-wide (not per-movie) state that is still USER DATA, so it belongs
    under projects/ rather than in the checkout: the dev tree symlinks
    projects/ to the production one, and a deploy's `git checkout` never
    touches it.

    A function, not a module constant, on purpose: several modules need this
    path, and a constant would freeze ``PROJECTS_DIR`` at import time in each
    of them independently — one redirected copy and the panel would read a
    different lessons file than the pipeline writes.
    """
    return PROJECTS_DIR / "_learning" / "lessons.json"


# Names under projects/ that are NOT movies. Reserved rather than filtered by
# a leading underscore: `_smoketest` is a legitimate throwaway project, so the
# rule is an explicit list of one, not a naming convention.
RESERVED_PROJECT_NAMES = frozenset({"_learning"})


def is_project_dir(path: Path) -> bool:
    """Is this entry under projects/ an actual movie workspace?

    Every place that enumerates projects (the CLI, the API listing, order
    intake) goes through here, so studio-wide state stored alongside them
    never shows up as a project with no photos in it.
    """
    return path.is_dir() and path.name not in RESERVED_PROJECT_NAMES


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
        if cleaned in RESERVED_PROJECT_NAMES:
            raise InvalidProjectName(
                f"'{cleaned}' is reserved for studio-wide data, not a project."
            )
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
    def cast_refs_dir(self) -> Path:
        """One canonical styled portrait per cast member (see media/faces.py).

        Cut for free out of frames already styled, so this directory is
        DERIVED data: deleting it costs nothing but a rebuild, and nothing
        downstream may fail because it is empty. Not created by mkdirs —
        a project with no tagged people never grows one."""
        return self.root / "cast_refs"

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
    def published_dir(self) -> Path:
        """Archive of every movie version delivered to the customer.

        `final_video.mp4` is rebuilt in place by each combine, so without a
        copy taken at publish time the bytes a customer was actually sent
        survive only in Cloudinary. Created on the first publish (not by
        mkdirs) so untouched projects stay clean."""
        return self.output_dir / "published"

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
    def custom_music_file(self) -> Path:
        """A user-supplied music track (uploaded via the panel / --music-file).

        Kept separate from the AI-generated ``music.mp3`` so it always wins and
        is never overwritten by a regeneration, even under ``--force``."""
        return self.output_dir / "music_custom.mp3"

    @property
    def state_file(self) -> Path:
        return self.logs_dir / "state.json"

    @property
    def failed_jobs_file(self) -> Path:
        return self.failed_jobs_dir / "failed_jobs.json"

    @property
    def costs_file(self) -> Path:
        """What this movie has cost so far, one line per paid API call.

        Lives with the other runtime bookkeeping (logs/) rather than at the
        project root: it is a record OF the run, like state.json."""
        return self.logs_dir / "costs.json"

    @property
    def feedback_file(self) -> Path:
        """Human judgements of this movie's rendered clips (see feedback.py).

        Project-level, next to order.json/published.json, because it is
        content about the movie — the LESSONS distilled from it are global."""
        return self.root / "feedback.json"

    @property
    def letter_file(self) -> Path:
        """User-written closing-letter text (used when closing_letter is on)."""
        return self.root / "letter.txt"

    @property
    def order_file(self) -> Path:
        """Which Cloudinary web order this project came from (written by
        `ingest`); absent for hand-made projects."""
        return self.root / "order.json"

    @property
    def published_file(self) -> Path:
        """Delivery history: every movie version published back to the order's
        Cloudinary folder (written by `publish`). See publish.py."""
        return self.root / "published.json"

    @property
    def default_storyboard_json(self) -> Path:
        return self.storyboard_dir / "storyboard.json"

    @property
    def face_audit_file(self) -> Path:
        """The last face-consistency audit's findings (see cmd_faces).

        Persisted because the audit is the one part of the cast-faces feature
        that costs money: the panel must be able to show yesterday's answer
        without buying it again. Derived and disposable — deleting it loses
        nothing but the report."""
        return self.storyboard_dir / "face_audit.json"

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
