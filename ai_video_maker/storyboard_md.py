"""Render a storyboard to human-readable markdown for review."""
from __future__ import annotations

from pathlib import Path

from .models import Storyboard


def write_storyboard_markdown(storyboard: Storyboard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# {storyboard.project_title}\n")
    lines.append(f"**Style:** {storyboard.style}\n")
    if storyboard.concept:
        lines.append(f"**Concept:** {storyboard.concept}\n")
    if storyboard.music_prompt:
        lines.append(f"**Music:** {storyboard.music_prompt}\n")
    durs = sorted({tr.duration for tr in storyboard.transitions})
    if len(durs) > 1:
        dur_desc = "mixed clip lengths (" + "/".join(f"{d}s" for d in durs) + ")"
    else:
        dur_desc = f"{(durs[0] if durs else storyboard.duration_per_clip)}s per clip"
    lines.append(
        f"**Output:** {storyboard.target_width}x{storyboard.target_height}, "
        f"{dur_desc}\n"
    )

    if storyboard.scenes:
        lines.append("## Scenes\n")
        for i, scene in enumerate(storyboard.scenes, start=1):
            lines.append(f"{i}. {scene}")
        lines.append("")

    lines.append("## Frames\n")
    for fr in storyboard.frames:
        lines.append(f"### Frame {fr.id}")
        lines.append(f"- **Description:** {fr.description}")
        lines.append(f"- **Image prompt:** {fr.image_prompt}")
        if fr.negative_prompt:
            lines.append(f"- **Negative prompt:** {fr.negative_prompt}")
        lines.append(f"- **Output:** `{fr.output_path}`")
        lines.append("")

    if storyboard.transitions:
        lines.append("## Transitions (clips)\n")
        for tr in storyboard.transitions:
            lines.append(f"### {tr.id}  ({tr.duration}s)")
            lines.append(f"- **Start:** `{tr.start_frame}`")
            lines.append(f"- **End:** `{tr.end_frame}`")
            lines.append(f"- **Motion prompt:** {tr.motion_prompt}")
            if tr.sound_prompt:
                lines.append(f"- **Sound prompt:** {tr.sound_prompt}")
            lines.append(f"- **Output:** `{tr.output_path}`")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
