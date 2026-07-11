# AI Video Maker

A local Python pipeline that turns images (or a raw idea) into short, consistent
1920×1080 videos using **OpenAI** (image generation/editing + storyboard
planning) and **fal.ai** (image-to-video + audio).

It renders one clip between each pair of consecutive key frames, then
concatenates them in order into `output/final_video.mp4` with `ffmpeg`. Clips
are silent by default; an opt-in [audio step](#audio-sound) adds per-clip sound
effects plus a music bed.

---

## The lifecycle

Every movie is a **project** under `projects/<name>/` with its own images,
storyboard, clips, output and state. The CLI is one command per step:

```bash
python pipeline.py init myfilm            # 1. create projects/myfilm/, then add images
python pipeline.py storyboard myfilm      # 2. style images + plan clips, stop for review
#    ...review/edit projects/myfilm/storyboard/storyboard.json...
python pipeline.py render myfilm          # 3. generate the clips from the storyboard
python pipeline.py combine myfilm         # 4. stitch clips into output/final_video.mp4

python pipeline.py status myfilm          # where am I? what's next?
python pipeline.py run myfilm             # or: everything in one go (with confirmations)
```

**The storyboard is the source of truth.** `storyboard` writes it, you edit it,
`render` executes it exactly. Re-running `storyboard` never overwrites your
edits (it reuses the saved storyboard while your images are unchanged); pass
`--force` to redo styling + analysis from scratch.

Whenever a step finishes, the app prints the exact command for the next step,
and `status` will always tell you where you stand.

### Editing cookbook

Styled frames and clips are **keyed by your input filenames** (input
`beach.jpg` → `styled_images/beach.png` → `clips/beach_to_party.mp4`), and
`storyboard` only re-plans what actually changed — everything untouched
(including your hand edits to the JSON) is carried over. That makes every edit
surgical. The recipes below assume a project called `myfilm` with images
`1.jpg`, `2.jpg`, `3.jpg`, …

**Naming inputs:** order is natural filename order (`2` before `10`, no
zero-padding needed). Plain `1.jpg, 2.jpg, 3.jpg` is fine — insert between 2
and 3 later by naming the new file `2a.jpg`. **Never renumber existing files
to make room**: to the pipeline a rename is a different image, so a full
renumber re-styles and re-renders almost everything. If you expect many
insertions, number in tens (`10.jpg, 20.jpg, 30.jpg`).

| Edit | Commands | Cost |
|------|----------|------|
| Regenerate one clip (e.g. after tweaking its `motion_prompt` in `storyboard.json`) | `python pipeline.py render myfilm --clip 2_to_3` | 1 clip |
| Change one clip's sound (edit its `sound_prompt` first) | `python pipeline.py audio myfilm --clip 2_to_3` | ~1¢ |
| Add an image between 2 and 3 | copy it in as `input_images/2a.jpg`, then:<br>`python pipeline.py storyboard myfilm`<br>`python pipeline.py render myfilm` | 1 styling + 2 clips |
| Remove image 2 | `rm projects/myfilm/input_images/2.jpg projects/myfilm/styled_images/2.png`, then:<br>`python pipeline.py storyboard myfilm`<br>`python pipeline.py render myfilm` | 1 clip |
| Swap image 2 for a different photo | overwrite `input_images/2.jpg` with the new file, then:<br>`python pipeline.py storyboard myfilm` (asks before re-styling)<br>`python pipeline.py render myfilm` | 1 styling + 2 clips |
| Re-style one image (new roll of the styling dice) | `rm projects/myfilm/styled_images/2.png`, then:<br>`python pipeline.py storyboard myfilm`<br>`python pipeline.py render myfilm` | 1 styling + 2 clips |
| Rebuild the movie after any of the above | `python pipeline.py combine myfilm --force` | free (local) |
| Redo all clips (e.g. after big storyboard edits) | `python pipeline.py render myfilm --force -y` | all clips |
| Redo styling + analysis from scratch | `python pipeline.py storyboard myfilm --force` | all stylings + 1 analysis |

Notes:

- `--clip` is repeatable (`--clip 2_to_3 --clip 3_to_4`) and always
  *regenerates* the named clips, resetting their SFX/fade state so redone
  clips get fresh audio.
- After add/remove/swap, `storyboard` re-plans **only the affected
  transitions** (a small vision call with just those frames), keeps everything
  else verbatim, and deletes clips whose frames changed so `render` redoes
  exactly those. Old clips that no longer match the storyboard become
  "strays" — `combine` ignores them and `status` lists them for deletion.
- **Preview before spending:** `render` prints a per-clip plan (render vs
  skip, durations, motion prompts) and asks before spending clip credits;
  `--dry-run` on any command prints the plan and spends nothing;
  `status` shows changed frames, missing clips, and the suggested next step.

> **Older projects** (with `styled_images/NNN_styled.png` files) keep their
> positional naming so nothing breaks — but positional names can't survive
> middle insertions/removals safely; the pipeline detects shifted sources and
> asks before re-styling. To migrate a project to filename-keyed naming,
> delete its `styled_images/` and `storyboard/` and re-run `storyboard`
> (re-styles everything once).

### From an idea instead of images

Pass `--idea` (or `--idea-file` for long/structured material) to `storyboard`:

```bash
python pipeline.py init robots
python pipeline.py storyboard robots --idea "A cute sea lion explores a futuristic base"
#    ...review/edit the storyboard (image prompts, motion, durations)...
python pipeline.py render robots          # generates the key frames, then the clips
```

- `--frame-count N` fixes the number of key frames; `--frame-count 0` lets the
  model pick a count that fits the material. Default: `default_frame_count`
  from `config.json`.
- You can also **skip AI planning entirely** and author
  `storyboard/storyboard.json` by hand, then run `render`.

---

## Requirements

- Python **3.11+**
- **ffmpeg** on your `PATH` (used to combine clips and mix audio). Install with
  `winget install Gyan.FFmpeg` (Windows), `brew install ffmpeg` (macOS), or
  `apt install ffmpeg` (Linux), then open a new terminal.
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

**Per-project overrides:** drop a `config.json` inside a project
(`projects/<name>/config.json`) with just the keys you want to change for that
movie — e.g. its own `style_prompt` or a different `fal_model_id`. It is merged
key-over-key on top of the shared config, so different movies can use different
looks/models without touching the global file.

### Model & start/end frames (important)

The pipeline is built around **consecutive frame pairs** (start → end). The video
model and the exact request shape come from the `fal_*` config fields, so you can
swap models without touching code.

**Default — Kling v2.5 Turbo Pro on fal** (supports start + end frame, so each
clip interpolates from one styled frame to the next):

```json
"fal_model_id": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
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

## How each step works

### `storyboard` (from images — the main flow)

1. Put source images in `input_images/` (`.jpg`, `.jpeg`, `.png`, `.webp`;
   ordered by natural filename order, so `img2` before `img10`).
2. Every image is styled into a consistent 1920×1080 look
   (`styled_images/001_styled.png`, …). Already-styled images are skipped on
   re-runs.
3. The styled frames are analysed by the vision model, which plans — for each
   consecutive pair — a tailored **motion prompt**, a **per-clip duration**
   (5 or 10s, leaning 5s; 10s is reserved for genuinely hard transitions),
   and a **sound prompt**, and writes
   `storyboard/storyboard.json` + `storyboard.md`.

`--no-analyze` skips step 3's vision call and uses the single global motion
prompt with one duration for every clip. `--duration 5|10` forces one length
for all clips even with analysis on.

### `render`

Reads the storyboard, generates any missing key frames (idea-based projects
only), then sends each consecutive frame pair to the video provider: start
frame + end frame + that pair's motion prompt → one clip in `clips/`
(`001_to_002.mp4`, …). `n` frames → `n − 1` clips.

Per-run overrides: `--motion-prompt` (all clips), `--duration` (all clips),
`--clip ID` (only those clips, force-regenerated).

### `combine`

Concatenates the storyboard's clips, in order, into `output/final_video.mp4`
(lossless stream-copy when the clips share a codec, otherwise a re-encode
fallback; clips with mixed audio presence are joined with silent padding).
Clips in `clips/` that don't belong to the current storyboard are ignored with
a warning, so stale files never leak into the movie. An existing final video
is only rebuilt with `--force`.

**Presentation extras (optional, off by default).** All pure local ffmpeg,
no API cost:

- `--final`: shorthand for the full presentation package — `--intro
  --credits-photos` in one flag. An explicit `--no-intro` /
  `--no-credits-photos` still wins over it.
- `--intro` (config: `intro_clip`): drop your own intro video at `intro.mp4`
  in the repo root — it's **shared by every project** — and it plays before
  everything else, scaled to fit the movie's frame size on black pads — never
  cropped. Its own audio is kept, with the music bed mixed over it. The
  `intro_file` config key relocates it (repo-root relative or absolute), and
  a per-project `config.json` can override it for one movie.
- `--credits-photos` (config: `credits_photos`): after the last clip, the
  original photos play as an end-credits montage, ~1.5s each
  (`credits_seconds_per_photo`), in movie order, under the same music bed.
- `--letter` (config: `closing_letter`): write a letter in
  `projects/<name>/letter.txt` (plain text; Hebrew and RTL are fully
  supported) and it rolls credits-style at the very end. **When
  `--credits-photos` is also on, the letter scrolls OVER the photo montage**
  (photos dimmed under a dark scrim so the text stays readable), with both
  paced to end together — photos never flash faster than configured and the
  letter never scrolls faster than configured. With the letter alone it
  scrolls over a plain dark background. Empty lines become paragraph gaps;
  long lines wrap. Font is auto-detected (override with `letter_font_path`),
  size via `letter_font_size` (default 64), pace via
  `letter_seconds_per_screen` (default 7.0 — higher is slower).
- **End fade** (config: `end_fade_seconds`, default 1.5): the video's last
  moments fade to black and the audio — music bed and SFX — fades out with
  them. Set `0` to disable.

Portrait photos are fitted whole onto a blurred background — nothing gets
cropped. The photos come from the storyboard's recorded sources
(`source_path`), so the montage stays in sync with edits automatically. Use
`--no-intro` / `--no-credits-photos` / `--no-letter` to override config for
one run.

Rendered segments live in `output/segments/` and are **reused** on the next
combine as long as their inputs (the intro video, the photos, `letter.txt`,
the config files) haven't changed since; edit any input and only the affected
segment is re-rendered. Delete `output/segments/` to force a full redo.

### `run`

`storyboard` (reused if saved) → confirmation → `render` → confirmation →
`combine`, in one command. `-y` skips the confirmations; `--no-combine` stops
after the clips.

---

## All commands & flags

Global: `--config config.json` (before the command). Every command takes the
project name as its first argument.

| Command | Flags |
|---------|-------|
| `init` | — |
| `storyboard` | `--force`, `--dry-run`, `--concurrency N`, `--style-prompt`, `--no-analyze`, `--duration 5\|10`, `--idea`, `--idea-file PATH`, `--frame-count N` |
| `render` | `--force`, `--dry-run`, `--concurrency N`, `-y/--yes`, `--clip ID` (repeatable), `--motion-prompt`, `--duration 5\|10`, `--add-audio`, `--no-audio` |
| `audio` | `--force`, `--dry-run`, `--concurrency N`, `--clip ID` (repeatable; redo that clip's audio), `--music-prompt`, `--music-file PATH` |
| `combine` | `--force`, `--dry-run`, `--music-file PATH`, `--add-audio`, `--no-audio`, `--final`, `--[no-]intro`, `--[no-]credits-photos`, `--[no-]letter` |
| `status` | — |
| `run` | everything above except `--clip`, plus `--no-combine` |

Shared flag meanings:

- `--force` — redo outputs even if already completed (for `storyboard` this
  re-styles the images **and** re-analyses; delete `storyboard/storyboard.json`
  instead to re-analyse only).
- `--dry-run` — print planned work; spend no API credits.
- `--concurrency N` — run N image/clip/SFX API jobs in parallel (overrides
  `max_parallel_requests`). `1` = sequential.
- `--add-audio` / `--no-audio` — force the audio layer on/off for this run,
  overriding `config.audio_mode`.

---

## Parallelism (speed)

Image styling, frame generation, and clip+SFX rendering are I/O-bound (most of
the time is spent waiting on the provider), so they run **in parallel** across a
small thread pool. Control it with `max_parallel_requests` in `config.json`
(default `4`) or `--concurrency N` per run.

Each clip's SFX and edge-fade run inside that clip's worker, so audio is
parallelised too. Job state (`logs/state.json`) and failure tracking are
thread-safe, so resume/skip and `failed_jobs.json` work exactly as before.
Higher concurrency is faster but more likely to hit provider **rate limits**;
transient 429s are retried with backoff, but if you see a lot of them, lower the
number. Dry-runs always run sequentially so the planned-work log stays ordered.

---

## Resuming after interruption / failures

- Job status is stored in `logs/state.json`. Completed images and clips are
  **skipped automatically** on the next run — just re-run the same command to
  resume where it stopped.
- Use `--force` to ignore saved state and redo everything, or
  `render --clip ID` to redo specific clips.
- Anything that failed is written to `failed_jobs/failed_jobs.json` with the
  error message and context (a clean run clears it). Fix the cause and re-run;
  only the unfinished/failed jobs are retried.
- Detailed logs for every run are written to `logs/pipeline_<timestamp>.log`.
- **Missing frames are bridged automatically.** If a frame fails to generate
  (e.g. frame 4), the clip step doesn't leave a hole — it pairs the nearest
  surviving neighbours directly (…`3→5`…) so the final video stays continuous.
  The bridged clip is named after the frames it actually joins
  (`003_to_005.mp4`). Fix the frame and re-run to get the original `3→4`/`4→5`
  clips back; the leftover bridged clip is then ignored by `combine` (it warns
  about strays so you can delete them).

The pipeline also has built-in retry with exponential backoff for transient API
errors, and it waits on provider jobs until they complete, fail, or time out.

**Content-filter false positives are retried with a reworded prompt.** Both
OpenAI (image styling/generation) and fal/Kling (clip rendering) sometimes flag
innocent prompts — family photos, affectionate moments, words like "shot".
When that happens the pipeline asks the text model to rephrase the prompt
(same scene and action, unambiguous wording) and resubmits, up to
`moderation_reword_attempts` times (config, default 3). The log shows the
reworded prompt that succeeded; your storyboard keeps the original, so paste
the reworded text into that transition's `motion_prompt` if you want it to
stick for future re-renders. If every reword of a clip's motion prompt is
still blocked, the clip is tried one final time with a generic safe fallback
prompt — the start/end frames still drive the motion, so you get a usable
(if less directed) clip instead of a failed render.

---

## Output layout

Everything below lives inside the project workspace, `projects/<name>/`:

| Folder | Contents |
|--------|----------|
| `input_images/` | Your source images (image-based projects) |
| `styled_images/` | Styled frames (`001_styled.png`, …) |
| `generated_frames/` | Idea-based generated frames (`001.png`, …) |
| `clips/` | Rendered clips (`001_to_002.mp4`, …) |
| `output/` | `final_video.mp4` + `music.mp3` (the generated bed, when audio is on) |
| `storyboard/` | `storyboard.json` (editable source of truth), `storyboard.md` (readable view), `preview.html` (visual contact sheet — open it in a browser) |
| `logs/` | Run logs + `state.json` |
| `failed_jobs/` | `failed_jobs.json` |

---

## Audio (sound)

The video providers above output **silent** clips. Sound is added in a separate,
opt-in step that runs entirely through **fal** (same `FAL_KEY`, no extra
account). Two independent layers:

1. **Per-clip SFX / ambient** — each silent clip is sent to a *video→audio*
   model (default `fal-ai/mmaudio-v2`), which watches the clip and returns the
   **same clip with synchronized sound muxed in**. Because it reads the actual
   pixels, every clip gets its own motion-matched audio.
2. **Music bed** — one instrumental track (default `fal-ai/elevenlabs/music`)
   is generated from a single prompt and mixed across the whole final video,
   **louder than the clip SFX** (the SFX is ducked under it). Tune the balance
   with `music_volume` / `sfx_volume` in `config.json`.

### Turning it on

Off by default (`"audio_mode": "none"`). Enable it permanently in `config.json`
(`"audio_mode": "post"`), per-run with `--add-audio`, or retrofit existing
clips:

```bash
# Render clips AND add sound in one run
python pipeline.py run myfilm --add-audio

# Already have silent clips? Add SFX + music and rebuild the final video:
python pipeline.py audio myfilm

# Use your own music track instead of generating one:
python pipeline.py audio myfilm --music-file ~/Music/mytrack.mp3

# Override just the music prompt:
python pipeline.py audio myfilm --music-prompt "Upbeat playful ukulele, no vocals"

# Force-off for one run even if config has audio_mode: post
python pipeline.py run myfilm --no-audio
```

The music bed comes from `--music-file` if given, else the project's existing
`output/music.mp3`, else it is generated from the music prompt (storyboard's
`music_prompt`, or `--music-prompt`, or config).

Cost is roughly **$0.20–0.50 per full video** (MMAudio is ~$0.001/s; music is
one short call). Requires `ffmpeg`/`ffprobe` on your `PATH`.

### Where the prompts come from

- **Image-based projects:** the frame analysis writes a per-clip `sound_prompt`
  into each transition; blank ones fall back to `default_sfx_prompt`. The music
  bed prompt comes from `config.json` (or `--music-prompt`).
- **Idea-based projects:** the storyboard also plans a `sound_prompt` per
  transition and one `music_prompt` for the whole video — all editable in
  `storyboard.json` before rendering.

### Config keys

| Key | Meaning |
|-----|---------|
| `audio_mode` | `"none"` (silent, default) or `"post"` (add sound). |
| `sfx_model_id` | fal video→audio model. Default `fal-ai/mmaudio-v2`. |
| `sfx_num_steps` | MMAudio sampling steps. |
| `default_sfx_prompt` | Fallback SFX prompt when a transition has none. |
| `sfx_negative_prompt` | What the SFX model should avoid (music/speech). |
| `sfx_extra_arguments` | Extra model-specific args merged into each SFX call. |
| `sfx_fade_seconds` | Fade each clip's SFX in/out at its edges so hard cuts aren't abrupt (the music bed carries the dip). Sync-preserving; `0` disables. Default `0.2`. |
| `sfx_volume` | `0..1`, how loud the per-clip SFX sits **under** the music (default `0.35`). |
| `music_model_id` | fal text→music model. Default `fal-ai/elevenlabs/music`. |
| `music_prompt` | Background-music description (fallback). |
| `music_volume` | `0..1`, how loud the background bed plays (default `0.85`). |
| `music_loop` | `false` (default): the track plays once; if the video is longer, the rest continues with SFX only. `true`: the track repeats for the whole video. A track longer than the video is trimmed either way. |
| `music_extra_arguments` | Extra model-specific args for the music call. |

Swap the SFX or music model by changing the id (e.g. `fal-ai/lyria2`,
`cassetteai/music-generator`, `fal-ai/thinksound`) — no code changes. SFX and
music are state-tracked like every other stage, so interrupted runs resume,
finished clips are skipped, and a regenerated clip automatically gets fresh
audio.

---

## Tests

The pure pipeline logic (frame bridging, clip planning/selection, resume
state, storyboard round-trips, image utilities) is covered by unit tests — no
network or API keys needed:

```bash
pip install -e ".[dev]"
pytest
```

---

## Code layout

The pipeline lives in the `ai_video_maker/` package; `pipeline.py` at the repo
root is a thin shim that calls into it. After `pip install -e .` you can also
run the `ai-video-maker` console command.

```
ai_video_maker/
  cli.py             # subcommand parsing + main() entry point (all interactivity)
  config.py          # Config — validated config.json (pydantic)
  workspace.py       # Workspace — all per-movie paths, derived from one base dir
  options.py         # RunOptions — one run's knobs (CLI flags or an API request)
  runner.py          # Pipeline — one cmd_* method per lifecycle command
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
pipeline.py          # entry-point shim
pyproject.toml       # package metadata, deps, `ai-video-maker` console script
```

The pipeline is built from three explicit inputs — `Config`, `Workspace`, and
`RunOptions` — plus an injected `confirm` callback for the interactive gates,
and reads no global state or stdin. Each CLI subcommand maps 1:1 onto a
`Pipeline.cmd_*` method, which is exactly the surface a future web API will
expose (each request builds its own `Workspace` + `RunOptions` and calls one
command).
