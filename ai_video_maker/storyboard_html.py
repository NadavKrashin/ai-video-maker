"""Render a storyboard to a self-contained HTML contact sheet.

The preview sits next to the storyboard (storyboard/preview.html) and shows
each planned clip as start frame -> motion/duration/sound -> end frame, so the
whole movie can be eyeballed in a browser before any clip credits are spent.
Frames that aren't on disk yet (idea-based projects before `render`) fall back
to showing their image prompt.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

from .models import Storyboard

_CSS = """
  body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem;
         background: #14161a; color: #e8e8e8; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; color: #9ecbff; }
  .meta { color: #aaa; margin-bottom: 1.5rem; max-width: 60rem; }
  .clip { display: flex; align-items: stretch; gap: 1rem; margin: 1.2rem 0;
          padding: 1rem; background: #1d2026; border-radius: 10px; }
  .frame { flex: 0 0 30%; }
  .frame img { width: 100%; border-radius: 6px; display: block; }
  .frame .label { font-size: .8rem; color: #888; margin-top: .3rem; }
  .frame .prompt { font-size: .8rem; color: #bbb; background: #23262d;
                   border-radius: 6px; padding: .6rem; }
  .arrow { flex: 1; display: flex; flex-direction: column; justify-content: center; }
  .arrow .dur { font-weight: 700; color: #9ecbff; margin-bottom: .4rem; }
  .arrow .motion { font-size: .9rem; line-height: 1.35; }
  .arrow .sound { font-size: .8rem; color: #8fb98f; margin-top: .5rem; }
"""


def _frame_cell(root: Path, out_dir: Path, output_path: str, label: str,
                fallback_text: str) -> str:
    file = root / output_path
    if file.exists():
        src = html.escape(os.path.relpath(file, out_dir).replace(os.sep, "/"))
        body = f'<img src="{src}" alt="{html.escape(label)}">'
    else:
        body = (
            '<div class="prompt">(not generated yet)<br><br>'
            f"{html.escape(fallback_text or 'no image prompt')}</div>"
        )
    return f'<div class="frame">{body}<div class="label">{html.escape(label)}</div></div>'


def write_storyboard_preview(storyboard: Storyboard, root: Path, path: Path) -> None:
    """Write the contact sheet for `storyboard` (frame paths relative to `root`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = path.parent
    frames_by_path = {f.output_path: f for f in storyboard.frames}

    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(storyboard.project_title)} — storyboard</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(storyboard.project_title)}</h1>",
        f"<div class='meta'><b>Style:</b> {html.escape(storyboard.style)}",
    ]
    if storyboard.concept:
        parts.append(f"<br><b>Concept:</b> {html.escape(storyboard.concept)}")
    if storyboard.music_prompt:
        parts.append(f"<br><b>Music:</b> {html.escape(storyboard.music_prompt)}")
    if storyboard.global_motion_prompt:
        parts.append(
            "<br><b>Global motion (every clip):</b> "
            f"{html.escape(storyboard.global_motion_prompt)}"
        )
    parts.append(
        f"<br><b>Output:</b> {storyboard.target_width}x{storyboard.target_height}, "
        f"{len(storyboard.transitions)} clip(s)</div>"
    )

    for tr in storyboard.transitions:
        start = frames_by_path.get(tr.start_frame)
        end = frames_by_path.get(tr.end_frame)
        parts.append('<div class="clip">')
        parts.append(_frame_cell(
            root, out_dir, tr.start_frame, Path(tr.start_frame).name,
            start.image_prompt if start else "",
        ))
        sound = (
            f'<div class="sound">&#128266; {html.escape(tr.sound_prompt)}</div>'
            if tr.sound_prompt else ""
        )
        parts.append(
            '<div class="arrow">'
            f'<div class="dur">{html.escape(tr.id)} &rarr; {tr.duration}s</div>'
            f'<div class="motion">{html.escape(tr.motion_prompt)}</div>'
            f"{sound}</div>"
        )
        parts.append(_frame_cell(
            root, out_dir, tr.end_frame, Path(tr.end_frame).name,
            end.image_prompt if end else "",
        ))
        parts.append("</div>")

    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")
