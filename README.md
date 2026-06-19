# AI Video Maker

A local Python pipeline that turns images (or a raw idea) into short, consistent
1920×1080 video clips using **OpenAI** (image generation/editing + storyboard
planning) and **KlingAI** (image-to-video).

It produces individual clips between consecutive key frames — **it does not
combine them**. You assemble the final cut yourself in **Premiere Pro**.

---

## Two modes

### Mode A — Image-to-video from your images (default)
1. Put source images in `input_images/`.
2. Every image is styled into a consistent 1920×1080 look.
3. Each **consecutive styled pair** is sent to KlingAI:
   image 1 = start frame, image 2 = end frame → one 5s or 10s clip.
4. `n` images → `n − 1` clips, written to `clips/`.

### Mode B — Generate from scratch (`--from-scratch`)
1. You provide an idea/prompt.
2. OpenAI writes a full storyboard (concept, scenes, frames, per-frame image
   prompts, per-transition motion prompts).
3. The plan is saved to `storyboard/storyboard.json` and `storyboard/storyboard.md`,
   then the app **stops and asks you to review/approve it**.
4. After you approve, it generates every key frame at 1920×1080.
5. Then it sends consecutive frame pairs to KlingAI → `n − 1` clips.

---

## Requirements

- Python **3.11+**
- An OpenAI API key
- KlingAI access key + secret key

## Setup

```bash
# 1. (recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

### Create your `.env`

Copy the example and fill in your real credentials:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...

# Kling auth — use ONE scheme (auto-detected):
# (a) single API key (most common; console shows one "api-key-kling-..." value)
KLING_API_KEY=api-key-kling-...
# (b) OR an Access Key + Secret Key pair (leave KLING_API_KEY blank if you use this)
# KLING_ACCESS_KEY=...
# KLING_SECRET_KEY=...
```

> **Kling auth:** if your Kling console shows a single **API Key**, set
> `KLING_API_KEY` and leave the AK/SK pair blank — it's sent directly as a Bearer
> token. If it shows an **Access Key + Secret Key** pair, set those two instead
> and the app signs a short-lived JWT. If both are set, the single API key wins.

Keys are loaded from `.env` via `python-dotenv` — they are **never hardcoded**,
and `.env` is git-ignored.

### Configure (optional)

Edit `config.json` to change style prompts, motion prompt, default duration,
model names, the Kling base URL, retry/poll settings, etc. It is validated on
startup, so typos are caught early.

---

## Mode A — using existing images

Place your images in `input_images/` (supported: `.jpg`, `.jpeg`, `.png`,
`.webp`). Ordering is by natural filename order (`img2` before `img10`).

```bash
# Preview the plan without spending any API credits
python pipeline.py --dry-run

# Full run: style images, then generate clips
python pipeline.py

# Only style the images (no video)
python pipeline.py --only-style

# Only generate videos from already-styled images in styled_images/
python pipeline.py --only-video

# Use 10-second clips
python pipeline.py --duration 10

# Re-do everything, ignoring previously completed outputs
python pipeline.py --force
```

Styled images are written as `styled_images/001_styled.png`,
`styled_images/002_styled.png`, … and clips as `clips/001_to_002.mp4`, …

---

## Mode B — generate from a raw idea

### Step 1 — create the storyboard (and stop)

```bash
python pipeline.py --from-scratch --create-storyboard \
  --idea "A cute sea lion explores a futuristic training base"
```

This writes `storyboard/storyboard.json` and `storyboard/storyboard.md`, then
prints:

> Storyboard created. Review storyboard/storyboard.md or storyboard/storyboard.json,
> edit if needed, then run with --approve-storyboard.

**Nothing is generated yet.** The app never jumps straight from an idea to a
full video.

### Step 2 — review & edit

Open `storyboard/storyboard.md` for a readable overview, or edit
`storyboard/storyboard.json` directly — it is designed to be human-editable.
You can tweak:
- each frame's `image_prompt` / `negative_prompt`
- each transition's `motion_prompt` and `duration`

### Step 3 — approve and generate

```bash
python pipeline.py --from-scratch --approve-storyboard \
  --storyboard-file storyboard/storyboard.json
```

This generates every key frame into `generated_frames/001.png`, … (each
normalized to exactly 1920×1080), then renders the clips using the
**per-transition** motion prompts from the storyboard.

---

## All command-line flags

| Flag | Description |
|------|-------------|
| `--config config.json` | Path to the config file. |
| `--force` | Redo outputs even if already completed. |
| `--dry-run` | Print planned work; spend no API credits. |
| `--only-style` | Only style/generate images; skip video. |
| `--only-video` | Only generate videos from existing images. |
| `--duration 5` / `--duration 10` | Clip length in seconds. |
| `--motion-prompt "..."` | Override the global/per-transition motion prompt. |
| `--style-prompt "..."` | Override the global style prompt (Mode A). |
| `--idea "..."` | The video idea (Mode B). |
| `--from-scratch` | Use Mode B. |
| `--create-storyboard` | Mode B: create storyboard and stop. |
| `--approve-storyboard` | Mode B: generate after review. |
| `--storyboard-file ...` | Storyboard JSON path for approval. |

---

## Resuming after interruption / failures

- Job status is stored in `logs/state.json`. Completed images and clips are
  **skipped automatically** on the next run — just re-run the same command to
  resume where it stopped.
- Use `--force` to ignore saved state and redo everything.
- Anything that failed is written to `failed_jobs/failed_jobs.json` with the
  error message and context. Fix the cause (e.g. a bad prompt, rate limit) and
  re-run; only the unfinished/failed jobs will be retried.
- Detailed logs for every run are written to `logs/pipeline_<timestamp>.log`.

The pipeline also has built-in retry with exponential backoff for transient API
errors, and it polls Kling jobs until they complete, fail, or time out.

---

## Output & final assembly

| Folder | Contents |
|--------|----------|
| `styled_images/` | Mode A styled frames (`001_styled.png`, …) |
| `generated_frames/` | Mode B generated frames (`001.png`, …) |
| `clips/` | Rendered clips (`001_to_002.mp4`, …) |
| `storyboard/` | `storyboard.json` + `storyboard.md` (Mode B) |
| `logs/` | Run logs + `state.json` |
| `failed_jobs/` | `failed_jobs.json` |

**The clips are intentionally NOT merged.** Import the `clips/` folder into
Premiere Pro and arrange them on the timeline to build your final video.

---

## Notes on the Kling client

All KlingAI-specific details (auth/JWT signing, the submit endpoint, the polling
status fields, and the result download) are isolated in the `KlingClient` class
in `pipeline.py`, with comments marking each spot that may need adjustment to
match the **official Kling API docs** if endpoints or field names change.
