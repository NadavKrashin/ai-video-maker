"""Small shared constants used across the package."""
from __future__ import annotations

# Clip durations the pipeline (and the video providers) support, in seconds.
# Veo 3.1 accepts 4s, 6s or 8s clips; the pipeline uses 4 (quick, subtle
# changes) and 8 (bigger/slower transitions).
VALID_DURATIONS = {4, 8}
