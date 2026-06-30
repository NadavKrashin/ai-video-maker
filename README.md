# AI Video Maker

A local Python pipeline that turns images (or a raw idea) into short, consistent
1920×1080 video clips using **OpenAI** (image generation/editing + storyboard
planning) and **fal.ai** (image-to-video).

It renders one clip between each pair of consecutive key frames, then
**concatenates them in order into `output/final_video.mp4`** with `ffmpeg`. Skip
that with `--no-combine` to cut it yourself, or `--combine` to stitch existing
clips without regenerating. Clips are silent by default; an opt-in
[audio step](#audio-sound) adds per-clip sound effects plus a music bed.

---

## Projects — one folder per movie (`--project`)

**Every movie lives in its own workspace under `projects/<name>/`**, with its own
`input_images/`, `generated_frames/`, `styled_images/`, `clips/`, `output/`,
`storyboard/`, and `logs/state.json`. Movies never collide and each resumes
independently — no `--force` or manual cleanup between them.

`--project` is **required** (there is no default project); the workspace is
created on first use. `config.json` and `.env` stay shared at the repo root, and
`projects/` is git-ignored.

```bash
# Mode A — set up the workspace, add images, then generate
python pipeline.py --project sealion --init   # create projects/sealion/ and stop
# ...copy your images into projects/sealion/input_images/...
python pipeline.py --project sealion

# Mode B — generate from an idea (same --project for every step)
python pipeline.py --project robots --from-scratch --create-storyboard --idea "..."
```

> Folder paths in this README (`input_images/`, `clips/`, `output/final_video.mp4`)
> are **relative to the project workspace**, i.e. `projects/<name>/input_images/`.

---

## Two modes

### Mode A — Image-to-video from your images (default)
1. Put source images in `input_images/`.
2. Every image is styled into a consistent 1920×1080 look.
3. The styled frames are analysed by the vision model, which writes a storyboard
   (`storyboard/storyboard.json` + `.md`) planning, for each consecutive pair, a
   tailored **motion prompt** (so the start→end interpolation is smooth) and a
   **per-clip duration** (5 or 10s, varied to suit each transition).
4. Each **consecutive styled pair** is sent to the video provider (default: fal,
   model Kling v2.1 Pro): image 1 = start frame, image 2 = end frame → one clip,
   using that pair's planned motion prompt and duration. (Provider/model/end-frame
   are configurable — see below.)
5. `n` images → `n − 1` clips, written to `clips/`.

Pass `--no-analyze` to skip step 3 and use a single global motion prompt with one
duration for every clip (the old behaviour). `--duration 5|10` forces one length
for all clips even with analysis on; `--motion-prompt` overrides the planned
prompts.

By default Mode A runs in one shot (it pauses to confirm before spending clip
credits). For an explicit **review/approve** gate — same as Mode B — split it
into two commands so you can hand-edit the storyboard first:

```bash
python pipeline.py --project sealion --create-storyboard   # style + analyse, then stop
#   ...review/edit projects/sealion/storyboard/storyboard.json...
python pipeline.py --project sealion --approve-storyboard   # render clips from it
```

Whenever a step stops, the app prints the exact command for the next step, so you
don't have to come back here for it.

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
- **ffmpeg** on your `PATH` (used to combine clips). Install with
  `winget install Gyan.FFmpeg` (Windows), `brew install ffmpeg` (macOS), or
  `apt install ffmpeg` (Linux), then open a new terminal. Only needed for the
  final-combine step — skip it with `--no-combine`.
- An OpenAI API key
- A **fal.ai** key (image-to-video + audio) — from https://fal.ai/dashboard/keys

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

# fal.ai key (image-to-video + audio) — from https://fal.ai/dashboard/keys:
FAL_KEY=your-fal-key
```

> **Auth:** OpenAI generates/edits the images and plans the storyboard; fal.ai
> renders every clip and all audio. Local images are uploaded to fal
> automatically — you don't host them.

Keys are loaded from `.env` via `python-dotenv` — they are **never hardcoded**,
and `.env` is git-ignored.

### Configure (optional)

Edit `config.json` to change the style prompts, motion prompt, default duration,
the fal model id, retry settings, etc. It is validated on startup (pydantic), so
typos are caught early.

### Model & start/end frames (important)

The pipeline is built around **consecutive frame pairs** (start → end). The video
model and the exact request shape come from the `fal_*` config fields, so you can
swap models without touching code.

**Default — Kling v2.1 Pro on fal** (supports start + end frame, so each clip
interpolates from one styled frame to the next):

```json
"fal_model_id": "fal-ai/kling-video/v2.1/pro/image-to-video",
"fal_start_frame_field": "image_url",
"fal_end_frame_field": "tail_image_url",
"fal_duration_as_string": true
```

- **Start frame** → `fal_start_frame_field` (`image_url`). **End frame** →
  `fal_end_frame_field` (`tail_image_url`); set it to `""` for start-frame-only.
- `fal_duration_as_string: true` because fal's Kling expects a string enum
  (`"5"`/`"10"`); set it `false` for a model that wants an integer.
- **Kling 3.0 on fal:** set `fal_model_id` to
  `"fal-ai/kling-video/v3/pro/image-to-video"`, `fal_start_frame_field` to
  `"start_image_url"`, and `fal_end_frame_field` to `"end_image_url"`.
- Add extra model-specific args via `fal_extra_arguments`
  (e.g. `{"negative_prompt": "blur, distortion, low quality"}`).

---

## Mode A — using existing images

Run `--init`, drop your images into `input_images/` (`.jpg`, `.jpeg`, `.png`,
`.webp`; ordered by natural filename order, so `img2` before `img10`), then run
the project. Styled images are written as `styled_images/001_styled.png`, … and
clips as `clips/001_to_002.mp4`, …

```bash
python pipeline.py --project sealion --init   # create the workspace, then add images
python pipeline.py --project sealion           # style → analyse → confirm → clips → combine
python pipeline.py --project sealion --dry-run # preview the plan, spend no credits
```

A normal run pauses to confirm before clips and before combining. For an explicit
review gate where you can hand-edit the storyboard first, split it in two
(see [the two-step flow](#mode-a--image-to-video-from-your-images-default) above).
Every other behaviour is a flag — see [All command-line flags](#all-command-line-flags)
(e.g. `--only-style`, `--only-video`, `--no-analyze`, `--duration`, `--combine`).

---

## Mode B — generate from a raw idea

> Pass the **same** `--project` to every step below.

### Step 1 — create the storyboard (and stop)

```bash
python pipeline.py --project sea-lion --from-scratch --create-storyboard \
  --idea "A cute sea lion explores a futuristic training base"
```

This writes `storyboard/storyboard.json` and `storyboard/storyboard.md`, then
prints the exact `--approve-storyboard` command to run next. **Nothing is
generated yet** — the app never jumps straight from an idea to a full video.

**Controlling how many frames/clips you get.** By default the storyboard uses
`default_frame_count` from `config.json` (8 frames → 7 clips). To change it:

```bash
# Fixed number of frames (e.g. 5 frames -> 4 clips)
python pipeline.py --project sea-lion --from-scratch --create-storyboard --idea "..." --frame-count 5

# Let the model choose the count based on YOUR content (no fixed padding)
python pipeline.py --project sea-lion --from-scratch --create-storyboard --idea "..." --frame-count 0

# Pass long / structured pasted data from a file instead of --idea
python pipeline.py --project sea-lion --from-scratch --create-storyboard --idea-file my_script.txt --frame-count 0
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
python pipeline.py --project sea-lion --from-scratch --approve-storyboard \
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
| `--project NAME` | **Required.** Movie workspace under `projects/NAME/` (own input, frames, clips, output, storyboard, state). The run errors out if omitted. |
| `--init` | Create the project workspace (`input_images/`, `clips/`, …) and exit without generating anything. Use it to set up a Mode A project before adding images. |
| `--force` | Redo outputs even if already completed. |
| `--dry-run` | Print planned work; spend no API credits. |
| `-y`, `--yes` | Skip the interactive confirmations (before clip generation and before combining); proceed automatically. |
| `--only-style` | Only style/generate images; skip video. |
| `--only-video` | Only generate videos from existing images (also resumes clip generation if you declined the confirmation prompt). |
| `--combine` | Only concatenate existing `clips/` into `output/final_video.mp4`; no generation. |
| `--no-combine` | Skip building the final combined video at the end of a run. |
| `--add-audio` | Force audio on for this run (per-clip SFX + music bed), ignoring `audio_mode`. |
| `--no-audio` | Force audio off for this run, even if `audio_mode` is `"post"`. |
| `--audio-only` | Add SFX + music to existing `clips/` and rebuild `output/final_video.mp4` (no generation). |
| `--music-prompt "..."` | Override the background-music prompt for this run. |
| `--duration 5` / `--duration 10` | Force every clip to this length. Omit to let clips mix 5s/10s (Mode A analysis or Mode B storyboard picks per clip). |
| `--concurrency N` | Run N image/clip/SFX API jobs in parallel (overrides `max_parallel_requests`). `1` = sequential. |
| `--motion-prompt "..."` | Override the global/per-transition motion prompt. |
| `--style-prompt "..."` | Override the global style prompt (Mode A). |
| `--no-analyze` | Mode A: skip the vision analysis of the styled frames; use one global motion prompt and one duration for every clip (the old behaviour). |
| `--idea "..."` | The video idea (Mode B). |
| `--idea-file PATH` | Read the idea/source material from a file (Mode B); overrides `--idea`. |
| `--frame-count N` | Mode B: number of key frames (overrides config). `0` = let the model decide. |
| `--from-scratch` | Use Mode B. |
| `--create-storyboard` | Create the storyboard and stop for review. Mode B builds it from your idea; Mode A styles + analyses your images. Prints the approve command when done. |
| `--approve-storyboard` | Generate the video from an approved storyboard. Mode B regenerates frames first; Mode A renders clips from the existing styled images. |
| `--storyboard-file ...` | Storyboard JSON path used by `--approve-storyboard`. |

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

Everything below lives inside the project workspace, `projects/<name>/`:

| Folder (under `projects/<name>/`) | Contents |
|--------|----------|
| `styled_images/` | Mode A styled frames (`001_styled.png`, …) |
| `generated_frames/` | Mode B generated frames (`001.png`, …) |
| `clips/` | Rendered clips (`001_to_002.mp4`, …) |
| `output/` | Combined `final_video.mp4` (all clips concatenated in order) + `music.mp3` (the generated bed, when audio is on) |
| `storyboard/` | `storyboard.json` + `storyboard.md` (both modes) |
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
   generated from a single prompt and mixed across the whole
   `output/final_video.mp4` **louder than the clip SFX** (the SFX is ducked under
   it). Tune the balance with `music_volume` / `sfx_volume` in `config.json`. The
   track is looped/trimmed to the video length.

   **Choosing the music (interactive).** When audio is on, just before the clips
   are combined you're asked which music file to use — Enter accepts the default
   `output/music.mp3`, or you can type a path to your own track, or `g` to
   generate one with ElevenLabs. If the file you pick isn't found, you're offered
   another file, generation, or skipping the music entirely. This prompt is
   skipped (and the old behaviour kept — reuse `music.mp3` if present, else
   generate) under `--yes`, in a non-interactive shell, or during `--dry-run`.

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

- **Mode A:** the frame analysis also writes a per-clip `sound_prompt` into each
  transition (used for that clip's SFX); clips with no planned sound — and every
  clip when `--no-analyze` is set — fall back to `default_sfx_prompt`. The music
  bed uses `music_prompt` (from `config.json`).
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
| `sfx_volume` | `0..1`, how loud the per-clip SFX sits **under** the music when both play (default `0.35`). |
| `music_model_id` | fal text→music model. Default `fal-ai/elevenlabs/music`. |
| `music_prompt` | Background-music description (Mode A / fallback). |
| `music_volume` | `0..1`, how loud the background bed plays — louder than `sfx_volume` so the music dominates (default `0.85`). |
| `music_extra_arguments` | Extra model-specific args for the music call. |

Swap the SFX or music model by changing the id (e.g. `fal-ai/lyria2`,
`cassetteai/music-generator`, `fal-ai/thinksound`) — no code changes. SFX and
music are state-tracked like every other stage, so interrupted runs resume and
finished clips are skipped (`--force` redoes them).

Each clip's SFX is generated independently, so the ambience can jump at a cut.
`sfx_fade_seconds` fades every clip's audio in/out at its edges (the music bed
fills the dip; hard cuts and A/V sync are untouched). The fade is tracked
separately from SFX (`fade:<clip>` in `state.json`), so re-running `--audio-only`
adds fades to already-sounded clips **without re-paying for SFX**.

---

## Code layout

The pipeline lives in the `ai_video_maker/` package; `pipeline.py` at the repo
root is a thin shim that calls into it, so every `python pipeline.py …` command
above works unchanged. After `pip install -e .` you can also run the
`ai-video-maker` console command.

```
ai_video_maker/
  cli.py             # argument parsing + main() entry point
  config.py          # Config — validated config.json (pydantic)
  workspace.py       # Workspace — all per-movie paths, derived from one base dir
  options.py         # RunOptions — one run's choices (CLI flags or an API request)
  runner.py          # Pipeline — orchestration (Mode A / Mode B / combine / audio)
  summary.py         # RunSummary — end-of-run report
  models.py          # Frame / Transition / Storyboard
  storyboard_md.py   # storyboard -> markdown for review
  state.py           # StateStore (resume) + FailedJobStore
  retry.py           # exponential-backoff retry helper
  errors.py          # PipelineError / ConfigError / StoryboardError
  constants.py       # shared constants
  media/
    images.py        # Pillow normalisation + image listing
    ffmpeg.py        # concat, ffprobe, edge fades, music mux
  clients/
    openai_client.py # image generation/editing + storyboard text (OpenAI)
    fal.py           # shared fal session: upload + subscribe + result parsing
    download.py      # shared atomic streaming download
    video.py         # VideoClient — image-to-video (fal)
    audio.py         # AudioClient — SFX + music (fal)
pipeline.py          # backwards-compatible entry-point shim
pyproject.toml       # package metadata, deps, `ai-video-maker` console script
```

The pipeline is built from three explicit inputs — `Config`, `Workspace`, and
`RunOptions` — and reads no global state, so the same `Pipeline(config,
workspace, options)` orchestration can later be driven by an API instead of the
CLI (each request builds its own `Workspace` + `RunOptions`).
