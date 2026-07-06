# CLAUDE.md — project instructions for AI sessions

You may edit this file yourself whenever you learn something future sessions
should know (a new convention, a gotcha, a decision the user made). Keep it
short and current; remove rules that no longer apply.

## What this project is

A Python CLI pipeline that turns a folder of images (or a text idea) into a
short 1920×1080 movie: OpenAI styles the images / plans a storyboard, fal.ai
(Kling) renders a clip between each consecutive frame pair, optional fal audio
adds per-clip SFX + a music bed, ffmpeg concatenates everything. It is being
evolved into the backend of a future web app, so every CLI subcommand maps 1:1
onto a `Pipeline.cmd_*` method (`ai_video_maker/runner.py`) that will become an
API endpoint. Priorities, in order: workflow smoothness > code quality > cost >
speed > scalability.

Core design rules:
- **The storyboard (`projects/<name>/storyboard/storyboard.json`) is the source
  of truth.** Users hand-edit it between steps; never regenerate or overwrite
  it implicitly (only on explicit request / `--force`).
- **No `input()` or other stdin use inside `ai_video_maker/`** — all
  interactivity lives in `cli.py` via the injected `confirm` callback.
- Resume is existence-based for files (styled images, clips) and
  state-based (`logs/state.json`) for in-place work (SFX, fades). When a clip
  is regenerated its `sfx:`/`fade:` state entries must be cleared.
- Transition motion prompts must describe in-world subject action, not camera
  moves (see `_MODE_A_SYSTEM` in `clients/openai_client.py`); the user
  explicitly rejects "zoom/pan/pull-back" slideshow-style prompts.

## Working rules (the user's standing instructions)

1. **Before committing anything, ask which branch to work on** (unless the
   user already said in this session). Do not assume `main`.
2. **Commit after each completed feature/fix** — small, focused commits with
   messages explaining *why*, not just *what*.
3. **Run the tests before every commit:** `.venv/bin/python -m pytest tests/ -q`
   (offline, <1s, no API keys needed). A commit with failing tests is not
   allowed.
4. **Add/extend tests when you change behaviour** in the pure logic (pair
   building, bridging, clip selection, state, config, media utilities). New
   features in `runner.py` logic should come with tests in `tests/`.
5. **Keep `README.md` in sync** — if you change CLI flags, commands, config
   keys, defaults, or workflow behaviour, update the README in the same
   commit.
6. **Never spend API credits without asking.** `storyboard`/`render`/`audio`
   on a real project costs real money (a Kling clip is roughly $0.35–0.70).
   Use `--dry-run`, the unit tests, or a disposable fixture project (create
   with `init _smoketest`, hand-write a storyboard.json + fake files, delete
   after) for verification. Real-credit test runs happen only with explicit
   user approval, preferably on a single clip via `render <proj> --clip ID`.
7. Never read, print, or commit `.env`.

## Commands you'll need

```bash
.venv/bin/python -m pytest tests/ -q          # tests (always before commit)
.venv/bin/python -m pyflakes ai_video_maker    # lint
.venv/bin/python pipeline.py --help            # CLI overview
.venv/bin/python pipeline.py status <project>  # project state + next step
```

Lifecycle: `init` → `storyboard` (stops for review; writes json/md/preview.html)
→ `render` (plan + confirm; `--clip ID` redoes one clip) → `audio` → `combine`;
`run` chains them with confirmation gates.

## Gotchas / facts sessions keep rediscovering

- `projects/` is gitignored and holds the user's real movies (e.g. `Matan`,
  `Entebbe`, `Hila`) — treat their contents as user data: don't delete,
  regenerate, or overwrite without asking. `ai_video_maker.egg-info/` is
  generated packaging metadata; ignore it.
- `config.json` at the repo root is the user's live shared config. Current
  model choices are deliberate: Kling v2.5 Turbo Pro (`fal_model_id`),
  `gpt-image-2` for images (user wants OpenAI images), `gpt-5.1` for
  text/vision planning. The gpt-5 model line rejects non-default `temperature`
  — don't add temperature params to chat calls.
- Per-project overrides: a `projects/<name>/config.json` is merged key-over-key
  on top of the shared config.
- OpenAI's safety filter sometimes false-positives on family photos during
  styling; `_image_with_moderation_recovery` rewords and retries — that's
  expected behaviour, not a bug to "fix".
- fal Kling durations are the string enum "5"/"10" (`fal_duration_as_string`);
  valid clip durations live in `constants.VALID_DURATIONS`.
- Clips are named `<startid>_to_<endid>.mp4`; bridged clips (a missing middle
  frame) get non-consecutive names like `003_to_005.mp4` and become "stray"
  once the frame is restored — `combine` ignores strays by design.
- ffmpeg concat: mixed silent/sounded clips must go through the concat-filter
  path with silent padding (`_combine_clips_mixed_audio`), never the demuxer.
