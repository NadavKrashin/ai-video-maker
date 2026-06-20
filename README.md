# AI Video Maker

A local Python pipeline that turns images (or a raw idea) into short, consistent
1920×1080 video clips using **OpenAI** (image generation/editing + storyboard
planning) and **fal** or **Higgsfield** (image-to-video; selectable, default fal).

Clips are silent by default; an opt-in [audio step](#audio-sound) can add
motion-matched sound effects per clip plus a background-music bed (via **fal**).

It produces individual clips between consecutive key frames and then
**concatenates them, in order, into `output/final_video.mp4`** using `ffmpeg`
(requires `ffmpeg` on your `PATH`). You can disable that step with `--no-combine`
and assemble the cut yourself, or run `--combine` to stitch existing clips
without regenerating anything.

---

## Two modes

### Mode A — Image-to-video from your images (default)
1. Put source images in `input_images/`.
2. Every image is styled into a consistent 1920×1080 look.
3. Each **consecutive styled pair** is sent to the video provider (default: fal,
   model Kling v2.1 Pro): image 1 = start frame, image 2 = end frame → one 5s or
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
- **ffmpeg** on your `PATH` (used to combine clips; e.g. `brew install ffmpeg`).
  Only needed for the final-combine step — skip it with `--no-combine`.
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

**Default — fal + Kling v2.1 Pro** (supports start + end frame, so each clip
interpolates from one styled frame to the next):

```json
"video_provider": "fal",
"fal_model_id": "fal-ai/kling-video/v2.1/pro/image-to-video",
"fal_start_frame_field": "image_url",
"fal_end_frame_field": "tail_image_url",
"fal_duration_as_string": true
```

- **Start frame** → `*_start_frame_field` (`image_url`). **End frame** →
  `*_end_frame_field` (`tail_image_url`); set it to `""` for start-frame-only.
- `duration` format differs by provider: **fal** wants a string enum (so
  `fal_duration_as_string: true`); **Higgsfield** wants an integer
  (`higgsfield_duration_as_string: false`). These defaults are correct.
- **Kling 3.0 on fal:** set `fal_model_id` to
  `"fal-ai/kling-video/v3/pro/image-to-video"`, `fal_start_frame_field` to
  `"start_image_url"`, and `fal_end_frame_field` to `"end_image_url"`.
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

# Use 10-second clips (Mode A uses one length for every clip)
python pipeline.py --duration 10

# Generate clips but skip building output/final_video.mp4
python pipeline.py --no-combine

# Re-do everything, ignoring previously completed outputs
python pipeline.py --force

# Combine the existing clips/ into output/final_video.mp4 (no generation)
python pipeline.py --combine
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

**Controlling how many frames/clips you get.** By default the storyboard uses
`default_frame_count` from `config.json` (8 frames → 7 clips). To change it:

```bash
# Fixed number of frames (e.g. 5 frames -> 4 clips)
python pipeline.py --from-scratch --create-storyboard --idea "..." --frame-count 5

# Let the model choose the count based on YOUR content (no fixed padding)
python pipeline.py --from-scratch --create-storyboard --idea "..." --frame-count 0

# Pass long / structured pasted data from a file instead of --idea
python pipeline.py --from-scratch --create-storyboard --idea-file my_script.txt --frame-count 0
```

- `--frame-count N` overrides the default; `--frame-count 0` tells the model to
  pick the number of frames that fits the material (each beat/scene maps to one
  or more frames, no padding to a fixed number).
- `--idea-file PATH` reads the idea/source material from a text file — best when
  pasting a lot of text or a structured outline (avoids shell-quoting issues).
  It takes precedence over `--idea`.
- You can also **skip AI generation entirely** and author `storyboard/storyboard.json`
  by hand with any number of frames/transitions, then go straight to Step 3.
  See the structure in [Step 2](#step-2--review--edit).

### Step 2 — review & edit

Open `storyboard/storyboard.md` for a readable overview, or edit
`storyboard/storyboard.json` directly — it is designed to be human-editable.
You can tweak:
- each frame's `image_prompt` / `negative_prompt`
- each transition's `motion_prompt` and `duration`

**Mixing clip lengths:** in Mode B each clip can be a different length. When the
storyboard is created the model picks a `duration` of `5` or `10` per transition
(longer for bigger/slower motion, shorter for subtle changes), so a single video
can mix 5s and 10s clips. Override any clip by editing its transition `duration`
in the JSON. Passing `--duration 5` or `--duration 10` forces **every** clip to
that one length (at create time and at approve time); omit it to keep the mix.

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
| `--combine` | Only concatenate existing `clips/` into `output/final_video.mp4`; no generation. |
| `--no-combine` | Skip building the final combined video at the end of a run. |
| `--add-audio` | Force audio on for this run (per-clip SFX + music bed), ignoring `audio_mode`. |
| `--no-audio` | Force audio off for this run, even if `audio_mode` is `"post"`. |
| `--audio-only` | Add SFX + music to existing `clips/` and rebuild `output/final_video.mp4` (no generation). |
| `--music-prompt "..."` | Override the background-music prompt for this run. |
| `--duration 5` / `--duration 10` | Force every clip to this length. Omit in Mode B to let clips mix 5s/10s. |
| `--concurrency N` | Run N image/clip/SFX API jobs in parallel (overrides `max_parallel_requests`). `1` = sequential. |
| `--motion-prompt "..."` | Override the global/per-transition motion prompt. |
| `--style-prompt "..."` | Override the global style prompt (Mode A). |
| `--idea "..."` | The video idea (Mode B). |
| `--idea-file PATH` | Read the idea/source material from a file (Mode B); overrides `--idea`. |
| `--frame-count N` | Mode B: number of key frames (overrides config). `0` = let the model decide. |
| `--from-scratch` | Use Mode B. |
| `--create-storyboard` | Mode B: create storyboard and stop. |
| `--approve-storyboard` | Mode B: generate after review. |
| `--storyboard-file ...` | Storyboard JSON path for approval. |

---

## Parallelism (speed)

Image styling, frame generation, and clip+SFX rendering are I/O-bound (most of
the time is spent waiting on the provider), so they run **in parallel** across a
small thread pool. Control it with `max_parallel_requests` in `config.json`
(default `4`) or `--concurrency N` per run:

```bash
python pipeline.py --concurrency 8     # render up to 8 clips at once
python pipeline.py --concurrency 1     # fully sequential (old behaviour)
```

Each clip's SFX and edge-fade run inside that clip's worker, so audio is
parallelised too. Job state (`logs/state.json`) and failure tracking are
thread-safe, so resume/skip and `failed_jobs.json` work exactly as before.
Higher concurrency is faster but more likely to hit provider **rate limits**;
transient 429s are retried with backoff, but if you see a lot of them, lower the
number. Dry-runs always run sequentially so the planned-work log stays ordered.
The final `ffmpeg` concatenation and the music bed run once, after all clips.

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
- **Missing frames are bridged automatically.** If a frame fails to generate
  (e.g. frame 4), the clip step doesn't leave a hole — it pairs the nearest
  surviving neighbours directly (…`3→5`…) so the final video stays continuous.
  The bridged clip is named after the frames it actually joins (`003_to_005.mp4`)
  and inherits the motion/sound/duration of the surviving start frame's
  transition. It's logged as a warning so you know it happened; fix the frame and
  re-run to get the original `3→4`/`4→5` clips back. (A bridged `3→5` is a bigger
  jump, so that one clip interpolates a larger change.)

The pipeline also has built-in retry with exponential backoff for transient API
errors, and it waits on provider jobs until they complete, fail, or time out.

---

## Output & final assembly

| Folder | Contents |
|--------|----------|
| `styled_images/` | Mode A styled frames (`001_styled.png`, …) |
| `generated_frames/` | Mode B generated frames (`001.png`, …) |
| `clips/` | Rendered clips (`001_to_002.mp4`, …) |
| `output/` | Combined `final_video.mp4` (all clips concatenated in order) + `music.mp3` (the generated bed, when audio is on) |
| `storyboard/` | `storyboard.json` + `storyboard.md` (Mode B) |
| `logs/` | Run logs + `state.json` |
| `failed_jobs/` | `failed_jobs.json` |

By default the clips are concatenated, in filename order, into
`output/final_video.mp4` with `ffmpeg` (lossless stream-copy when the clips
share a codec, otherwise a re-encode fallback). Use `--no-combine` to skip this
and assemble the cut yourself in an editor, or `--combine` to (re)build the final
video from whatever is already in `clips/`. Re-running won't overwrite an
existing `final_video.mp4` unless you pass `--force`.

---

## Audio (sound)

The video providers above output **silent** clips. Sound is added in a separate,
opt-in step that runs entirely through **fal** (same `FAL_KEY`, no extra account),
so it works no matter which provider rendered the clips. Two independent layers:

1. **Per-clip SFX / ambient** — each silent clip is sent to a *video→audio* model
   (default `fal-ai/mmaudio-v2`), which watches the clip and returns the **same
   clip with synchronized sound muxed in**. Because it reads the actual pixels,
   every clip gets its own motion-matched audio even when they share a prompt.
2. **Music bed** — one instrumental track (default `fal-ai/elevenlabs/music`) is
   generated from a single prompt and mixed, **ducked**, under the SFX across the
   whole `output/final_video.mp4`. The track is looped/trimmed to the video length.

### Turning it on

It is **off by default** (`"audio_mode": "none"`) so existing runs are unchanged.
Enable it either permanently in `config.json` (`"audio_mode": "post"`) or per-run:

```bash
# Generate clips AND add sound in one run
python pipeline.py --add-audio

# Mode B with sound (the storyboard also plans per-clip + music prompts)
python pipeline.py --from-scratch --approve-storyboard --add-audio \
  --storyboard-file storyboard/storyboard.json

# Already have silent clips? Add SFX + music and rebuild the final video only:
python pipeline.py --audio-only

# Override just the music bed for a run:
python pipeline.py --add-audio --music-prompt "Upbeat playful ukulele, no vocals"

# Force-off for one run even if config has audio_mode: post
python pipeline.py --no-audio
```

Cost is roughly **$0.20–0.50 per full video** (MMAudio is ~$0.001/s; music is one
short call). Requires `ffmpeg`/`ffprobe` on your `PATH` for the music mix.

### Where the prompts come from

- **Mode A:** every clip uses `default_sfx_prompt`; the bed uses `music_prompt`
  (from `config.json`). Audio still differs per clip because MMAudio reads the video.
- **Mode B:** the storyboard step asks OpenAI to also write a `sound_to_next` per
  transition and one `music_prompt` for the whole video. These land in
  `storyboard.json` (and are shown in `storyboard.md`) and are **fully editable**
  before you approve — same as the motion prompts. Blank ones fall back to config.

### Config keys

| Key | Meaning |
|-----|---------|
| `audio_mode` | `"none"` (silent, default) or `"post"` (add sound). |
| `sfx_model_id` | fal video→audio model. Default `fal-ai/mmaudio-v2`. |
| `sfx_num_steps` | MMAudio sampling steps. |
| `default_sfx_prompt` | SFX prompt for Mode A and as the Mode B fallback. |
| `sfx_negative_prompt` | What the SFX model should avoid (music/speech). |
| `sfx_extra_arguments` | Extra model-specific args merged into each SFX call. |
| `sfx_fade_seconds` | Fade each clip's SFX in/out by this many seconds so hard cuts aren't abrupt (the music bed carries the dip). Sync-preserving; `0` disables. Default `0.2`. |
| `music_model_id` | fal text→music model. Default `fal-ai/elevenlabs/music`. |
| `music_prompt` | Background-music description (Mode A / fallback). |
| `music_volume` | `0..1`, how loud the bed sits under the SFX (default `0.25`). |
| `music_extra_arguments` | Extra model-specific args for the music call. |

Swap the SFX or music model by changing the id (e.g. `fal-ai/lyria2`,
`cassetteai/music-generator`, `fal-ai/thinksound`) — no code changes. SFX and
music are state-tracked like every other stage, so interrupted runs resume and
finished clips are skipped (`--force` redoes them).

### Smoothing clip-to-clip sound

Each clip's SFX is generated independently, so the ambience can change abruptly
at a cut. `sfx_fade_seconds` softens that by fading every clip's audio in/out at
its edges; the continuous music bed fills the brief dip, so the result is smooth
without overlapping clips (your hard cuts and A/V sync are untouched). The fade
is tracked separately from SFX (`fade:<clip>` in `state.json`), so if you already
generated SFX on a set of clips you can add fades **without re-paying for SFX** —
just run `python pipeline.py --audio-only` again (SFX is skipped, fades apply, the
existing `music.mp3` is reused). Set `sfx_fade_seconds: 0` to disable.

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
