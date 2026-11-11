"""deploy/backup.sh — the only thing standing between projects/ and a
permanent loss. Tested by RUNNING it against throwaway folders, and each
refusal asserts what survives, not just the exit code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "backup.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None or shutil.which("bash") is None,
    reason="needs bash + rsync",
)


def _run(src: Path, dest: Path, stamp: str, keep: int = 30):
    env = {
        **os.environ,
        "PROJECTS_DIR": str(src),
        "BACKUP_DEST": str(dest),
        "BACKUP_STAMP": stamp,
        "BACKUP_KEEP": str(keep),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture
def projects(tmp_path: Path) -> Path:
    src = tmp_path / "projects"
    (src / "Matan" / "clips").mkdir(parents=True)
    (src / "Matan" / "clips" / "a_to_b.mp4").write_bytes(b"clip-bytes")
    (src / "Matan" / "storyboard.json").write_text('{"frames": []}')
    (src / "Hila").mkdir()
    (src / "Hila" / "final_video.mp4").write_bytes(b"movie")
    return src


@pytest.fixture
def dest(tmp_path: Path) -> Path:
    d = tmp_path / "backup-disk"
    d.mkdir()
    return d


def test_a_snapshot_is_a_complete_copy(projects, dest):
    result = _run(projects, dest, "20260101-000000")
    assert result.returncode == 0, result.stderr
    snap = dest / "projects-20260101-000000"
    assert _tree(snap) == _tree(projects)
    assert not list(dest.glob("*.partial"))


def test_a_project_can_be_restored_from_a_snapshot(projects, dest, tmp_path):
    _run(projects, dest, "20260101-000000")
    before = _tree(projects / "Matan")
    shutil.rmtree(projects / "Matan")  # the disaster

    restored = tmp_path / "restored" / "Matan"
    shutil.copytree(dest / "projects-20260101-000000" / "Matan", restored)
    assert _tree(restored) == before


def test_unchanged_files_are_hard_links_not_second_copies(projects, dest):
    _run(projects, dest, "20260101-000000")
    (projects / "Hila" / "final_video.mp4").write_bytes(b"movie, rebuilt")
    result = _run(projects, dest, "20260102-000000")
    assert result.returncode == 0, result.stderr

    old, new = (dest / f"projects-2026010{n}-000000" for n in (1, 2))
    same = "Matan/clips/a_to_b.mp4"
    assert (old / same).stat().st_ino == (new / same).stat().st_ino
    # The changed file is a real new copy, and the OLD snapshot still holds
    # the old bytes — a later backup must never rewrite an earlier one.
    assert (new / "Hila/final_video.mp4").read_bytes() == b"movie, rebuilt"
    assert (old / "Hila/final_video.mp4").read_bytes() == b"movie"


def test_an_unmounted_destination_is_refused_not_created(projects, tmp_path):
    missing = tmp_path / "Volumes" / "Backup"
    result = _run(projects, missing, "20260101-000000")
    assert result.returncode != 0
    assert "does not exist" in result.stderr
    assert not missing.exists()


def test_an_empty_source_never_becomes_a_snapshot(projects, dest, tmp_path):
    _run(projects, dest, "20260101-000000")
    empty = tmp_path / "wiped"
    empty.mkdir()
    result = _run(empty, dest, "20260102-000000", keep=1)
    assert result.returncode != 0
    assert "no files" in result.stderr
    # The good snapshot is still there: a wiped source must not be able to
    # push it out through pruning.
    assert [p.name for p in dest.iterdir()] == ["projects-20260101-000000"]


def test_a_destination_inside_the_source_is_refused(projects):
    inside = projects / "backups"
    inside.mkdir()
    result = _run(projects, inside, "20260101-000000")
    assert result.returncode != 0
    assert "inside the source" in result.stderr
    assert not list(inside.iterdir())


def test_pruning_keeps_the_newest_and_never_touches_other_folders(projects, dest):
    (dest / "family-photos").mkdir()
    (dest / "family-photos" / "x.jpg").write_bytes(b"not ours")
    (dest / "projects-20250101-000000.partial").mkdir()  # an interrupted run
    for day in (1, 2, 3, 4):
        result = _run(projects, dest, f"2026010{day}-000000", keep=2)
        assert result.returncode == 0, result.stderr

    names = sorted(p.name for p in dest.iterdir())
    assert names == [
        "family-photos",
        "projects-20250101-000000.partial",
        "projects-20260103-000000",
        "projects-20260104-000000",
    ]
    assert (dest / "family-photos" / "x.jpg").read_bytes() == b"not ours"
    # Hard links mean deleting old snapshots cannot hollow out the new ones.
    assert _tree(dest / "projects-20260104-000000") == _tree(projects)


def test_the_source_is_never_modified(projects, dest):
    before = _tree(projects)
    _run(projects, dest, "20260101-000000")
    _run(projects, dest, "20260102-000000", keep=1)
    assert _tree(projects) == before


def test_the_same_snapshot_name_is_never_overwritten(projects, dest):
    _run(projects, dest, "20260101-000000")
    (projects / "Hila" / "final_video.mp4").write_bytes(b"changed")
    result = _run(projects, dest, "20260101-000000")
    assert result.returncode != 0
    assert "already exists" in result.stderr
    snap = dest / "projects-20260101-000000"
    assert (snap / "Hila/final_video.mp4").read_bytes() == b"movie"
