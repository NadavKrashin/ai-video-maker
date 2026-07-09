"""Small shared constants used across the package."""
from __future__ import annotations

# Clip durations the pipeline (and the video providers) support, in seconds.
# Veo 3.1 accepts 4s, 6s or 8s clips: 4 for quick, subtle changes, 8 for
# bigger/slower transitions, 6 in between.
VALID_DURATIONS = {4, 6, 8}
