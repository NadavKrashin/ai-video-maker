# CLAUDE.md — project instructions for AI sessions

**Maintaining this file is part of your job.** Before ending a task, check:
did the user state a new preference or standing decision? Did you hit a gotcha
that cost you time and isn't written here? Did a rule below become stale? If
yes, update this file (no need to ask permission — but mention the update in
your response). Keep it short and current; remove rules that no longer apply.

## What this project is

A Python CLI pipeline that turns a folder of images (or a text idea) into a
short movie: OpenAI styles the images (1920×1080) / plans a storyboard, fal.ai
(currently Seedance 2.0, previously Kling — the video model is config-swappable
via the `fal_*` keys) renders a clip between each consecutive frame pair,
`fal_resolution` caps the movie's resolution (720p on Seedance), optional fal audio
adds per-clip SFX + a music bed, ffmpeg concatenates everything. It is being
evolved into the backend of a future web app, so every CLI subcommand maps 1:1
onto a `Pipeline.cmd_*` method (`ai_video_maker/runner.py`) that will become an
API endpoint. Priorities, in order: workflow smoothness > code quality > cost >
speed > scalability.

Core design rules:
- **The storyboard (`projects/<name>/storyboard/storyboard.json`) is the source
  of truth.** Users hand-edit it between steps. `storyboard` RECONCILES rather
  than regenerates: it keeps every transition whose frames are unchanged and
  re-plans only dirty pairs (frame inserted/removed/re-styled) via a targeted
  vision call (`_reconcile_storyboard` / `_plan_pairs` in runner.py). Never
  regress to overwriting the whole storyboard implicitly.
- **Styled frames are keyed by input filename** (`beach.jpg` →
  `styled_images/beach.png` → `clips/beach_to_party.mp4`), so input edits stay
  surgical. Projects that still contain `NNN_styled.png` files run in legacy
  positional mode — auto-detected in `_styled_targets`; don't break it.
- A styled image is re-styled when its source is newer or when
  `Frame.source_path` shows it came from a different file — gated on a
  confirmation because it spends credits. When a frame changes, its adjacent
  clips are deleted (stale) so `render` redoes them.
- **No `input()` or other stdin use inside `ai_video_maker/`** — all
  interactivity lives in `cli.py` via the injected `confirm` callback.
- Resume is existence-based for files (styled images, clips) and
  state-based (`logs/state.json`) for in-place work (SFX, fades). When a clip
  is regenerated its `sfx:`/`fade:` state entries must be cleared.
- Transition motion prompts must describe in-world subject action, not camera
  moves (see `_MODE_A_SYSTEM` in `clients/openai_client.py`); the user
  explicitly rejects "zoom/pan/pull-back" slideshow-style prompts. Since the
  Seedance swap they must also end with a stability clause + "One continuous
  shot, no cuts." — Seedance is multi-shot-capable and will cut mid-clip on
  scene-change-ish prompts otherwise (Kling couldn't). Don't remove those
  closing clauses.
- When a pair's two frames show DIFFERENT people, the motion prompt must stage
  an exit-and-entrance (or reveal) — never continuous identity: Kling morphs
  one person into the other otherwise, which the user finds creepy. Same
  person at a different age/in different clothes may animate continuously.
  Enforced in `_MODE_A_SYSTEM` and the fallback `motion_prompt` in config.json.
- Styling must preserve LIKENESS: cartoon rendering, real facial geometry.
  The user rejected "Disney-princess-ified" faces (enlarged eyes, slimmed,
  beautified) — people must stay recognizable as themselves. Encoded in
  `style_prompt` (config.json); keep that likeness language when touching it.

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
   on a real project costs real money (a Seedance 2.0 clip is roughly
   $1.50–3.00; Kling was $0.35–0.70).
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
  model choices are deliberate: Seedance 2.0 (`bytedance/seedance-2.0/
  image-to-video`, swapped from Kling v2.5 Turbo Pro at the user's request —
  the Kling recipe stays documented in the README for switching back),
  `gpt-image-2` for images (user wants OpenAI images), `gpt-5.1` for
  text/vision planning. The gpt-5 model line rejects non-default `temperature`
  — don't add temperature params to chat calls.
- Seedance 2.0 specifics: end frame field is `end_image_url`, durations are
  strings `"4"`…`"15"` (so the `"5"`/`"10"` enum still fits), documented max
  resolution 720p (final movie is 1280×720; some fal pages claim 1080p on the
  standard tier — unverified), and it generates native audio unless
  `generate_audio: false` is sent — config.json disables it via
  `fal_extra_arguments` so the pipeline's own `audio` step stays in charge.
  Don't mix 720p Seedance clips with 1080p Kling clips in one project.
- Per-project overrides: a `projects/<name>/config.json` is merged key-over-key
  on top of the shared config.
- Content filters false-positive on family content: OpenAI during styling,
  and fal/Kling on clip motion prompts (`content_policy_violation`, e.g.
  affection + "bed" wording). Both recover via reword-and-retry
  (`with_reword_recovery` in retry.py; motion prompts reworded by
  `reword_motion_prompt`). Expected behaviour, not a bug to "fix".
- fal durations are sent as strings (`fal_duration_as_string`): Kling takes
  the enum "5"/"10", Seedance takes "4"…"15"; valid clip durations live in
  `constants.VALID_DURATIONS`.
- Clips are named `<startid>_to_<endid>.mp4`; bridged clips (a missing middle
  frame) get non-consecutive names like `003_to_005.mp4` and become "stray"
  once the frame is restored — `combine` ignores strays by design.
- ffmpeg concat: mixed silent/sounded clips must go through the concat-filter
  path with silent padding (`_combine_clips_mixed_audio`), never the demuxer.
