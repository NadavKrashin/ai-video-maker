"""The handful of config keys a project may override from the panel.

Config is a big surface and most of it breaks renders in ways that are hard
to diagnose from a browser — a wrong `fal_end_frame_field` silently drops the
end frame, a rewritten `style_prompt` quietly undoes years of accumulated
rules. So this is NOT a general config editor. It is an explicit allowlist of
the knobs that exist to be EXPERIMENTED with: numbers whose right value is a
matter of taste that only watching renders can settle.

Every one of them was already tunable — by editing
`projects/<name>/config.json` on the machine. Behind the Cloudflare tunnel
that means a shell on the mini, which made "just try 2/3 on one project"
advice nobody could act on. Making them config was only half the job.

The registry is the single source of truth: the server validates against it
and the panel renders its inputs from it, so a new knob is one entry here
rather than a schema kept in step across two languages.

Nothing here spends money or changes anything already on disk. Each key is a
PLAN-TIME or RENDER-TIME input, so it lands on the next storyboard/render —
the same rule as the cast and the photo tags.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .style_realism import STYLE_REALISM_LABELS, STYLE_REALISM_LEVELS


@dataclass(frozen=True)
class ProjectSetting:
    """One overridable config key, and what the panel should show for it."""

    key: str
    label: str
    kind: str                 # "int" | "bool" | "choice"
    help: str
    # Where the value takes effect, so the card can say what to run next
    # instead of leaving the user to guess why nothing changed.
    applies: str
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    # For kind "choice": (value, label) pairs, in the order to show them.
    choices: tuple[tuple[str, str], ...] = ()

    def coerce(self, value: Any) -> Any:
        """Validate one incoming value. Raises ValueError with a reason."""
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.key} must be true or false")
            return value
        if self.kind == "choice":
            allowed = [v for v, _ in self.choices]
            if value not in allowed:
                raise ValueError(
                    f"{self.key} must be one of: {', '.join(allowed)}"
                )
            return value
        # bool is a subclass of int in Python, and `True` arriving for a
        # threshold is a caller mistake worth naming rather than storing as 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{self.key} must be a whole number")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.key} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.key} must be at most {self.maximum}")
        return value


PROJECT_SETTINGS: tuple[ProjectSetting, ...] = (
    ProjectSetting(
        key="unstageable_mover_budget",
        label="Mover budget",
        kind="int",
        minimum=0,
        maximum=20,
        applies="re-plan",
        help=(
            "How many people may leave or arrive between two frames and still "
            "be choreographed one by one. Above it the pair gets a camera "
            "transition instead. Lower it if people distort while travelling "
            "between frames; raise it for more person-to-person action."
        ),
    ),
    ProjectSetting(
        key="unstageable_crowd_limit",
        label="Crowd limit",
        kind="int",
        minimum=0,
        maximum=30,
        applies="re-plan",
        help=(
            "Above this many people who matter in the shot (clearly visible "
            "individuals; small background figures don't count) in either "
            "frame, ANY change to who is in the group hands the pair to the "
            "camera — in a crowd an epithet stops picking anyone out. A group "
            "that stays the same on both sides is never affected."
        ),
    ),
    ProjectSetting(
        key="cast_references_enabled",
        label="Match faces to the cast references",
        kind="bool",
        applies="restyle",
        help=(
            "Send each person's canonical styled face along when styling a "
            "photo of them, so the same person stays the same character. Turn "
            "it off to go back to styling every photo on its own."
        ),
    ),
    ProjectSetting(
        key="fal_elements_partial",
        label="Pin a face even when not everyone fits",
        kind="bool",
        applies="render",
        help=(
            "Only matters on a model that accepts face references. When a clip "
            "shares more people than the model can take, OFF sends none (one "
            "crisp face among several drifting ones can read worse than all of "
            "them drifting together) and ON rescues the most recurring face. "
            "Worth A/B-ing on a single clip."
        ),
    ),
)

# The look is a CHOICE rather than a bool so a further level can be added
# without changing what is stored on disk.
PROJECT_SETTINGS = PROJECT_SETTINGS + (
    ProjectSetting(
        key="style_realism",
        label="Look",
        kind="choice",
        applies="restyle",
        choices=tuple(
            (level, STYLE_REALISM_LABELS[level]) for level in STYLE_REALISM_LEVELS
        ),
        help=(
            "How cartoon the styled frames come out, on the same image model. "
            "Lifelike keeps people closest to their real faces (the current "
            "look). Classic cartoon renders skin, hair, clothes and light the "
            "way the studio's earlier movies looked, without changing faces. "
            "Frames already styled keep their look until re-styled: re-run "
            "Storyboard with \u201cForce re-style all photos\u201d to redo every "
            "photo in the new look (spends image credits)."
        ),
    ),
)

_BY_KEY = {s.key: s for s in PROJECT_SETTINGS}


def get_setting(key: str) -> Optional[ProjectSetting]:
    return _BY_KEY.get((key or "").strip())


def setting_keys() -> list[str]:
    return [s.key for s in PROJECT_SETTINGS]


def coerce_settings(values: dict[str, Any]) -> dict[str, Any]:
    """Validate an incoming {key: value} patch against the allowlist.

    A ``None`` value means "stop overriding this" and is passed through, so
    the caller can delete the key and let the shared config decide again —
    which is a different thing from pinning the shared value, and has to stay
    expressible or a project could never be un-pinned.

    Unknown keys are REFUSED rather than ignored: silently dropping one would
    let the panel believe it saved something it did not.
    """
    out: dict[str, Any] = {}
    for key, value in values.items():
        setting = get_setting(key)
        if setting is None:
            raise ValueError(
                f"{key!r} is not an editable project setting. Editable: "
                + ", ".join(setting_keys())
            )
        out[key] = None if value is None else setting.coerce(value)
    return out


def describe_settings(config: Any, pinned: dict[str, Any]) -> list[dict[str, Any]]:
    """The registry plus this project's live answers, for the panel.

    ``value`` is what the project actually runs with, ``pinned`` says whether
    that came from this project's own config.json or was inherited from the
    shared one. The distinction matters: a number shown without it looks like
    a per-project choice somebody made, when it may be the studio default
    about to change under them.
    """
    return [
        {
            "key": s.key,
            "label": s.label,
            "kind": s.kind,
            "help": s.help,
            "applies": s.applies,
            "minimum": s.minimum,
            "maximum": s.maximum,
            "choices": [{"value": v, "label": label} for v, label in s.choices],
            "value": getattr(config, s.key),
            "pinned": s.key in pinned,
        }
        for s in PROJECT_SETTINGS
    ]
