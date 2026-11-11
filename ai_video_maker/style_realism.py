"""How cartoon or how lifelike a project's styled frames come out.

Added 2026-11-12 (user request). Moving styling to gpt-image-2.6-sunburst
made frames noticeably more lifelike than the gpt-image-2 frames before it.
The SAME style prompt that used to produce a strong cartoon now produces
something closer to a portrait. The user chose that look for faithful
likeness, but customers may want the classic cartoon look the studio
shipped before. So the look is a per-project choice on the SAME, latest
model: one prompt layer per level, chosen in the panel's Project settings
(`style_realism`).

Two invariants keep this safe:

- "lifelike" adds NOTHING. The base style prompt is the approved look on
  this model ("rewording them would ship a look nobody has seen"), so the
  default level must reach the model byte-for-byte as it did before this
  existed. A test pins that.
- The cartoon layer changes RENDERING, never the PEOPLE. Every likeness rule
  in the base prompt still governs. The user rejected enlarged eyes, slimmed
  faces and "Disney-princess" beautification long before rejecting too much
  realism, so the extra cartoon comes from materials, light and
  simplification, not from proportions.
"""
from __future__ import annotations

# Order = from most stylized to most lifelike, for any UI that lists them.
STYLE_REALISM_LEVELS: tuple[str, ...] = ("cartoon", "lifelike")
DEFAULT_STYLE_REALISM = "lifelike"

STYLE_REALISM_LABELS: dict[str, str] = {
    "cartoon": "Classic cartoon",
    "lifelike": "Lifelike (current)",
}

CARTOON_LOOK_LAYER = (
    "LOOK — CLASSIC CARTOON. This movie wants the classic animated-feature "
    "look, so push the STYLIZATION of the rendering clearly further than a "
    "lightly stylized portrait: every frame must read at a glance as a still "
    "from a 3D animated film, never as a retouched or filtered photograph. "
    "Render skin as smooth, matte, softly lit cartoon skin with a gentle glow "
    "— no pores, fine wrinkles, blemishes or photographic skin texture. "
    "Render hair as soft, sculpted, clumped shapes with simple highlights, "
    "never as individual photographic strands. Render clothing and fabric as "
    "clean, simplified shapes with broad soft folds and flat, saturated "
    "colour, dropping fine weave, small print detail and photographic "
    "texture. Use bright, warm, saturated animated-film lighting with soft "
    "bounce light, and simplify backgrounds into rounded, painterly forms. "
    "This is a change of RENDERING ONLY, never of the people: every likeness "
    "rule above still governs. Keep each person's real face shape, eye size "
    "and spacing, nose, hairline and hair exactly as shown (including "
    "baldness), and their skin tone, age, body type, glasses and facial hair. "
    "Do NOT enlarge eyes, shrink noses, slim faces, smooth away "
    "distinguishing features or exaggerate proportions to look more cartoon. "
    "The cartoon look comes from materials, light and simplification, never "
    "from changing who someone is."
)

_LAYERS: dict[str, str] = {
    "cartoon": CARTOON_LOOK_LAYER,
    "lifelike": "",
}


def with_style_realism(prompt: str, level: str) -> str:
    """The style prompt for one styling call, at the project's chosen look.

    An unknown level is treated as the default rather than raised: config
    validation already refuses one, so reaching here with a bad value means
    a caller bypassed it, and styling should degrade to the approved look,
    not fail an order.
    """
    layer = _LAYERS.get(level, "")
    if not layer:
        return prompt
    return f"{prompt}\n\n{layer}"
