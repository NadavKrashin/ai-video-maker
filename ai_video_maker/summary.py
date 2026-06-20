"""End-of-run summary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .workspace import Workspace


@dataclass
class RunSummary:
    input_count: int = 0
    styled_created: int = 0
    styled_skipped: int = 0
    styled_failed: int = 0
    videos_created: int = 0
    videos_skipped: int = 0
    videos_failed: int = 0
    sfx_created: int = 0
    sfx_skipped: int = 0
    sfx_failed: int = 0
    music_added: bool = False
    final_video: Optional[Path] = None

    def print(self, workspace: Workspace) -> None:
        line = "=" * 60
        print(f"\n{line}\nRUN SUMMARY\n{line}")
        print(f"  Input/generated images : {self.input_count}")
        print(f"  Styled/frames created  : {self.styled_created}")
        print(f"  Styled/frames skipped  : {self.styled_skipped}")
        print(f"  Styled/frames failed   : {self.styled_failed}")
        print(f"  Videos created         : {self.videos_created}")
        print(f"  Videos skipped         : {self.videos_skipped}")
        print(f"  Videos failed          : {self.videos_failed}")
        if self.sfx_created or self.sfx_skipped or self.sfx_failed:
            print(f"  Clip SFX created       : {self.sfx_created}")
            print(f"  Clip SFX skipped       : {self.sfx_skipped}")
            print(f"  Clip SFX failed        : {self.sfx_failed}")
        if self.music_added:
            print("  Music bed              : added")
        if self.final_video:
            print(f"  Final video            : {self.final_video}")
        print("\n  Output folders:")
        print(f"    Styled images   : {workspace.styled_images_dir}")
        print(f"    Generated frames: {workspace.generated_frames_dir}")
        print(f"    Clips           : {workspace.clips_dir}")
        print(f"    Logs            : {workspace.logs_dir}")
        print(f"    Failed jobs     : {workspace.failed_jobs_file}")
        print(line)
