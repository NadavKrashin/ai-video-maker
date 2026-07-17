# CLAUDE.md — project instructions for AI sessions

**Maintaining this file is part of your job.** Before ending a task, check:
did the user state a new preference or standing decision? Did you hit a gotcha
that cost you time and isn't written here? Did a rule below become stale? If
yes, update this file (no need to ask permission — but mention the update in
your response). Keep it short and current; remove rules that no longer apply.

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
  clips are deleted (stale) so `render` redoes them — but staleness is
  decided by CONTENT HASH (`Frame.styled_hash`), never by mtime, and the
  deletion is confirm-gated: a cloud-sync client once bumped every styled
  image's mtime and the old mtime heuristic silently wiped a project's
  rendered clips (real money). Keep both protections.
- **No `input()` or other stdin use inside `ai_video_maker/`** — all
  interactivity lives in `cli.py` via the injected `confirm` callback.
- Resume is existence-based for files (styled images, clips) and
  state-based (`logs/state.json`) for in-place work (SFX, fades). When a clip
  is regenerated its `sfx:`/`fade:` state entries must be cleared.
- Transition motion prompts must describe in-world subject action, not camera
  moves (see `_MODE_A_SYSTEM` in `clients/openai_client.py`); the user
  explicitly rejects "zoom/pan/pull-back" slideshow-style prompts.
- Motion prompts are BEAT-BUDGETED to the clip length: 5s = exactly one
  continuous action, 10s = max two beats. User-verified on a real clip: an
  overloaded 5s prompt (lift-carry-seat-examine) made Kling swap in another
  kid, while a rewritten single-action prompt ("kid runs to the patio ...")
  rendered fine. Kling also can't show cross-ground travel when the subject
  fills both frames — stage exit-past-camera + re-entry, or world morphs
  around a steady subject, instead.
- Beat budgets need WORD CAPS enforced in code, not just prompt rules: a
  real plan under the beat-budget rule alone still wrote 79–113 words for
  every 5s clip, and an 84-word 5s prompt rendered as a whip-pan blur (a
  disguised cut). Caps: 5s ≤ 35 words, 10s ≤ 60 (`_MOTION_WORD_LIMITS`);
  over-budget prompts get a targeted condense call in
  `_coerce_transition_plans` (`_condense_motion_prompt`, falls back to the
  original on failure).
- NO OFF-SCREEN HANDS in motion prompts: an action done TO the subject by
  someone not visible in the frames ("as if being lifted out", "is gently
  set down") makes Kling levitate the subject — a real clip had the toddler
  flying out of his stroller into glowing light. Subjects act under their
  own power, or hold steady while the world transforms. Enforced in
  `_MODE_A_SYSTEM` and `_CONDENSE_MOTION_SYSTEM`.
- When a pair's two frames show DIFFERENT people, the motion prompt must stage
  an exit-and-entrance (or reveal) — never continuous identity: Kling morphs
  one person into the other otherwise, which the user finds creepy. Same
  person at a different age/in different clothes may animate continuously.
  Enforced in `_MODE_A_SYSTEM` and the fallback `motion_prompt` in config.json.
- The flip side: when both frames show the SAME person, the motion prompt
  must say so explicitly ("the same little boy, now in ...") using singular
  he/she — never singular "they/their" and never handover phrasing like
  "the scene shifts to a toddler ...". Both made Kling swap in a
  different-looking kid mid-story on a real project. Enforced in
  `_MODE_A_SYSTEM` ("SAME PERSON, ONE PROTAGONIST").
- Clip durations must LEAN SHORT, and prompt-side bias alone cannot deliver
  it: real plans came back all-5s under "prefer 5" and all-10s under a
  hard-transition checklist (in a photo-album movie nearly every pair
  changes setting or outfit). The planner therefore only RATES each pair's
  difficulty 1-5 (3 = one major change, 4 = two at once OR any transition
  that can't physically play out in 5s, 5 = shares almost nothing /
  different people) and code derives durations: >=4 → 10s, capped
  at 1/3 of clips (`_LONG_CLIP_MAX_FRACTION`, `_select_long_clips` in
  clients/openai_client.py). A hard pair squeezed into 5s visibly teleports
  (seen in a real render), so keep both sides of the balance.
- A RELOCATION WITHIN ONE SETTING (high chair → couch across the same room,
  subject prominent in both frames) is a difficulty-4 pair even though the
  setting/outfit don't change: exit + crossing + arrival can't compress
  into one 5s action, so Kling degrades it into a disguised cut (real
  render: baby drops out of frame, camera slides to the couch — no
  transition at all). Rate it 4 → 10s and stage the full journey with a
  motivated camera following the subject.
- With ~20 frames in one vision call the planner can slip a pair mid-array
  (a transition describing the PREVIOUS pair — happened on a real plan).
  Every transition therefore declares a `pair_index` that code re-aligns by
  (`_realign_by_pair_index`), and motion prompts must END at exactly what
  the end frame shows. Don't remove either guard.
- Styling must preserve LIKENESS while being an unmistakable Pixar-style
  cartoon: the whole scene (people, clothing, background) renders as a
  stylized 3D animated film still — never near-photorealistic — but with
  real facial geometry. The user rejected both "Disney-princess-ified" faces
  (enlarged eyes, slimmed, beautified) AND outputs that stayed too realistic;
  people must stay recognizable as themselves inside a full cartoon look.
  Encoded in `style_prompt` / `scratch_style_prompt` (config.json); keep both
  the likeness and full-scene-cartoon language when touching them.

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

Lifecycle: (`orders` → `ingest` for paid web orders, or `init` + manual
images) → `storyboard` (stops for review; writes json/md/preview.html)
→ `render` (plan + confirm; `--clip ID` redoes one clip) → `audio` → `combine`;
`run` chains them with confirmation gates.

## The web frontend (animoments) & order intake

- The customer-facing web app lives in a SEPARATE repo:
  `../animoments` (github.com/itaycohen1010/animoments). Standing rule:
  frontend changes go on a **separate branch** there; this pipeline repo
  works on `main`.
- After payment the frontend uploads each order's photos to Cloudinary
  (cloud `dmxkoz4jo`, unsigned preset `videoOrders`), one folder per order:
  `video-orders/<ORDER-ID>_<customer>-<dd.mm.yyyy_HH-MM>/` with photos named
  `1, 2, ...` = their position in the movie, each asset tagged with the
  folder leaf name and carrying `context.order`. Customer PII is NOT stored
  in Cloudinary (email only). The order id (`AM-...`) reaches the user by
  confirmation email.
- `pipeline.py orders` / `pipeline.py ingest <project> <order>` are the
  intake commands (`clients/cloudinary_client.py`; Admin API, basic auth via
  CLOUDINARY_API_KEY/SECRET in .env). Asset listing is BY TAG first (works
  in both Cloudinary folder modes), public_id-prefix as fallback. `orders`
  is the one project-less CLI command — special-cased in cli.py before
  workspace resolution.
- `pipeline.py serve` (`server.py`) is the admin API + order watcher: FastAPI,
  token auth via ADMIN_API_TOKEN in .env (Bearer header or `?token=` for
  media tags), serial background JobRunner (one pipeline command at a time,
  whitelisted commands/options), and a watcher thread that auto-ingests a
  new order once uploads have been quiet `watch_quiet_minutes` (the frontend
  confirms payment BEFORE photos finish uploading — never ingest on folder
  existence alone). `watch_auto_storyboard` (default on) spends OpenAI
  credits automatically per paid order — deliberate, user-approved. The
  review surface is the animoments admin panel (frontend repo), which polls
  this API; the user explicitly chose the panel over Telegram notifications.
- `Pipeline.snapshot()` is the single source of truth for project status
  (cmd_status prints it, the API returns it). `projects/<name>/order.json`
  (written by ingest) ties a project to its Cloudinary order folder; the
  watcher/`orders` listing use it to know what's already handled.
- The user's `.venv` is a split install: `.venv/bin/python` → an anaconda
  python3.12 with its own site-packages, `.venv/bin/pip` → python3.11.
  Install packages with `.venv/bin/python -m pip`, never `.venv/bin/pip`.
- Long-term direction (agreed 2026-07-16): remaining phases — payment
  webhook from the frontend (exact photo_count completeness), delivery step
  (upload final + customer email), later move off the Mac to a VPS behind a
  tunnel. Keep review gates human; automate only the plumbing.

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
- Content filters false-positive on family content: OpenAI during styling,
  and fal/Kling on clip motion prompts (`content_policy_violation`). Kling's
  worst trigger is a baby/child being physically handled (lifted, bounced,
  settled) plus bed/crib/blanket wording. Recovery is reword-and-retry
  (`with_reword_recovery` in retry.py; each rewrite drops risky detail rather
  than paraphrasing), and clip generation ends with one `last_resort` try
  using the generic `SAFE_FALLBACK_MOTION_PROMPT` (clients/video.py) so a
  false positive degrades the prompt, not the render. Expected behaviour,
  not a bug to "fix".
- Customer phone photos must NEVER be uploaded to OpenAI as-is: iPhone HDR
  shots are MPO containers (JPEG + embedded gain-map image; `file` still says
  "JPEG", PIL says format MPO) and the image API rejects them wholesale with
  400 `invalid_image_file` — a real paid order lost 21/30 frames this way.
  `prepare_image_for_upload` (media/images.py) round-trips every style input
  through Pillow (primary frame only, EXIF orientation baked, long side
  capped) before upload; keep it in the path.
- The org's OpenAI image quota is 5 input-images/min, so a whole-order
  styling batch WILL 429 on the tail. Rate limits get their own patient
  retry budget in `with_retries` (separate from `max_retries`, honours the
  server's "try again in Xs" hint) — don't collapse it back into the
  exponential-backoff attempt count. BUT `insufficient_quota` (account out
  of credits) also arrives as HTTP 429 and waiting never fixes it —
  `is_quota_exhausted_error` classifies it permanent; a real order burned
  ~6 min per planning call retrying it before that guard existed.
- When a storyboard planning call fails, the affected transitions get
  config `motion_prompt` as a PLACEHOLDER ("a planning hiccup never sinks
  the run"). Reconcile treats `motion_prompt == config.motion_prompt` as
  never-planned and re-plans it on every storyboard run; when a real plan
  replaces a placeholder, the pair's rendered clip goes into the
  confirm-gated stale list. `snapshot()["storyboard"]
  ["placeholder_transitions"]` surfaces them. A real order rendered 26/29
  clips with the generic prompt before these guards existed.
- fal Kling durations are the string enum "5"/"10" (`fal_duration_as_string`);
  valid clip durations live in `constants.VALID_DURATIONS`.
- Clips are named `<startid>_to_<endid>.mp4`; bridged clips (a missing middle
  frame) get non-consecutive names like `003_to_005.mp4` and become "stray"
  once the frame is restored — `combine` ignores strays by design.
- ffmpeg concat: mixed silent/sounded clips must go through the concat-filter
  path with silent padding (`_combine_clips_mixed_audio`), never the demuxer.
