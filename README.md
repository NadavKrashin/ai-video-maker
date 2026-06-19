# AI Video Maker

A local Python pipeline that turns images (or a raw idea) into short, consistent
1920×1080 video clips using **OpenAI** (image generation/editing + storyboard
planning) and **fal** or **Higgsfield** (image-to-video; selectable, default fal).

It produces individual clips between consecutive key frames — **it does not
combine them**. You assemble the final cut yourself in **Premiere Pro**.

---

## Two modes

### Mode A — Image-to-video from your images (default)
1. Put source images in `input_images/`.
2. Every image is styled into a consistent 1920×1080 look.
3. Each **consecutive styled pair** is sent to the video provider (default: fal,
   model Kling 3.0): image 1 = start frame, image 2 = end frame → one 5s or
   10s clip. (Provider/model/end-frame are configurable — see below.)
4. `n` images → `n − 1` clips, written to `clips/`.

### Mode B — Generate from scratch (`--from-scratch`)
1. You provide an idea/prompt.
2. OpenAI writes a full storyboard (concept, scenes, frames, per-frame image
   prompts, per-transition motion prompts).
3. The plan is saved to `storyboard/storyboard.json` and `storyboard/storyboard.md`,
   then the app **stops and asks you to review/approve it**.
4. After you approve, it generates every key frame at 1920×1080.
5. Then it sends consecutive frame pairs to the video provider → `n − 1` clips.

---

## Requirements

- Python **3.11+**
- An OpenAI API key
- A video-provider key: **fal** (default — from https://fal.ai/dashboard/keys),
  or Higgsfield (from https://cloud.higgsfield.ai)

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

# Video provider (default fal) — one key from https://fal.ai/dashboard/keys:
FAL_KEY=your-fal-key

# Only if you set "video_provider": "higgsfield" in config.json:
# HF_KEY=your-api-key:your-api-secret   # or HF_API_KEY + HF_API_SECRET
```

> **Video auth:** the pipeline uses the provider set by `video_provider` in
> `config.json` (default `fal`). For fal, get a key at
> [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) and set `FAL_KEY`. For
> Higgsfield, set `HF_KEY` (or `HF_API_KEY` + `HF_API_SECRET`). Either way, local
> images are uploaded to the provider automatically — you don't host them.

Keys are loaded from `.env` via `python-dotenv` — they are **never hardcoded**,
and `.env` is git-ignored.

### Configure (optional)

Edit `config.json` to change the video provider, style prompts, motion prompt,
default duration, the model id, retry settings, etc. It is validated on startup,
so typos are caught early.

### Provider, model & start/end frames (important)

The pipeline is built around **consecutive frame pairs** (start → end). Pick the
provider with `video_provider` (`"fal"` or `"higgsfield"`). Each provider has its
own block of settings, so you can keep both configured and just flip the switch.

**Default — fal + Kling 3.0** (supports start + end frame, so each clip
interpolates from one styled frame to the next):

```json
"video_provider": "fal",
"fal_model_id": "fal-ai/kling-video/v3/pro/image-to-video",
"fal_start_frame_field": "start_image_url",
"fal_end_frame_field": "end_image_url",
"fal_duration_as_string": true
```

- **Start frame** → `*_start_frame_field` (`start_image_url` for v3). **End
  frame** → `*_end_frame_field` (`end_image_url` for v3); set it to `""` for
  start-frame-only.
- `duration` format differs by provider: **fal** wants a string enum (so
  `fal_duration_as_string: true`); **Higgsfield** wants an integer
  (`higgsfield_duration_as_string: false`). These defaults are correct.
- **Cheaper option — Kling v2.1 Pro on fal:** set `fal_model_id` to
  `"fal-ai/kling-video/v2.1/pro/image-to-video"`, `fal_start_frame_field` to
  `"image_url"`, and `fal_end_frame_field` to `"tail_image_url"`.
- **Why fal is the default:** fal documents the Kling schema and validates
  inputs, so the end frame is actually applied (and unknown fields error instead
  of being silently ignored). Higgsfield's public API accepted the request but
  ignored the end frame.
- Add extra model-specific args via `fal_extra_arguments`
  (e.g. `{"negative_prompt": "blur, distortion, low quality"}`).

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
errors, and it waits on provider jobs until they complete, fail, or time out.

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

## Notes on the video clients

Both providers share a base class (`SubscribeVideoClient`) in `pipeline.py`,
with thin subclasses `FalClient` and `HiggsfieldClient` (selected by
`make_video_client` via `video_provider`). Each uses that provider's official
SDK — [`fal-client`](https://pypi.org/project/fal-client/) /
[`higgsfield-client`](https://pypi.org/project/higgsfield-client/) — which handle
authentication and uploading local images to hosted URLs. The model id, frame
field names, duration format, and extra arguments are all driven by `config.json`
(`fal_*` / `higgsfield_*`), so swapping models or providers needs no code
changes. See the [fal docs](https://docs.fal.ai) /
[Higgsfield docs](https://docs.higgsfield.ai) for model-specific parameters.
