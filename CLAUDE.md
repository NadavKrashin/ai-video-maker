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
adds per-clip SFX, ffmpeg concatenates everything and mixes in a
user-supplied music bed. It is being
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
  confirmation because it spends credits. When a frame changes (decided by
  CONTENT HASH `Frame.styled_hash`, never mtime — a cloud-sync client once
  bumped every mtime and the old heuristic wiped a project's clips), the
  adjacent rendered clips are only MARKED outdated (state `stale:<clip>`,
  shown by status/snapshot/panel). **Clips are NEVER auto-deleted or
  auto-re-rendered**: confirm gates don't protect server jobs (the admin
  API's confirm auto-answers yes — that combination deleted 26 rendered
  clips on a real order), so redoing a clip is always an explicit, named
  action (`render --clip ID`, repeatable), which also clears the stale mark.
  The panel's "Generate everything that needs it" batches outdated +
  never-rendered clips into ONE render job, but stays inside that rule: the
  clip ids are enumerated client-side from the snapshot and listed
  individually in the confirmation modal — never a blanket "re-render all".
- Saving a storyboard through the API (`PUT /api/projects/<n>/storyboard`)
  DIFFS against the copy on disk and marks the rendered clips whose plan
  changed as stale (`changed_transition_ids` in models.py). Motion prompt /
  duration / start_frame / end_frame invalidate the clip; `sound_prompt`
  does NOT (that only feeds the audio step); editing `global_motion_prompt`
  invalidates every clip, because it is prepended to each prompt at render
  time. Without this a hand edit lived only in the browser tab that made it,
  so nothing downstream could know which clips to redo.
- **No `input()` or other stdin use inside `ai_video_maker/`** — all
  interactivity lives in `cli.py` via the injected `confirm` callback.
- Resume is existence-based for files (styled images, clips) and
  state-based (`logs/state.json`) for in-place work (SFX, fades). When a clip
  is regenerated its `sfx:`/`fade:` state entries must be cleared.
- **Music is NEVER generated** (removed 2026-07-26 at the user's request).
  The bed is always a track the user supplies: `--music-file`, or an upload
  in the panel's Audio or Combine step, a URL (`--music-url` /
  `POST /api/projects/<n>/music/url`) — all of which write
  `output/music_custom.mp3`. No music model, no music prompt, no
  `Storyboard.music_prompt` — don't reintroduce them. No track simply means
  a movie with SFX only, which is a normal outcome. URL fetching
  (`media/music_url.py`) takes a direct audio link via requests or extracts
  a page's audio with yt-dlp; 60 MB cap, http(s) only. yt-dlp breaks when
  YouTube changes and needs `pip install -U yt-dlp`. LICENSING is the
  user's call and the panel says so at the input — these movies are sold.
- Transition motion prompts must describe in-world subject action, not camera
  moves (see `_MODE_A_SYSTEM` in `clients/openai_client.py`); the user
  explicitly rejects "zoom/pan/pull-back" slideshow-style prompts.
- MOTION PROMPTS DESCRIBE THE SUBJECTS ONLY — never the scene (user call,
  2026-07-28). The two frames already fix the setting/light/outfits and
  Kling interpolates the background by itself, so naming the location or
  narrating the world ("the green Andean peaks rise behind them", "the room
  gives way to open sky") buys nothing, spends the word cap, and pulls the
  model off the people — the one thing it gets wrong unaided. The user does
  not care HOW the scenery changes. What the prompt owes instead is a
  PATH: an ordinary, walkable route from where each person stands in the
  start frame to where they stand in the end frame — no appearing,
  materialising, floating or sliding into place. The old "WORLD FLOW
  SECOND" priority (light shifts, weather rolls in, environment transforms)
  is GONE; its replacement is "if they cannot travel, they stay put" —
  people stand, turn in place, share a look, and the frames carry the rest.
  Also pinned now: an exit past the camera is only HALF the staging (the
  walk back in must be written too), and a prompt's last beat is never
  people LEAVING, because the clip must land on the end frame. Triggered by
  a real plan for a salt-flat → Machu Picchu pair (swap + setting change):
  "…squeeze a little closer, share a quiet smile, then slowly step toward
  the camera and past it" — hold-steady staging on swapped people, ending
  on the exit. Pinned by TestMotionPromptsDescribeSubjectsOnly.
- EVERY PERSON WHO MOVES GETS A CONCRETE PHYSICAL VERB (user call,
  2026-07-28, same thread). Kling animates BODIES, so the prompt must name
  a movement you could act out — walk, step, turn, crouch, kneel, climb,
  jump, crawl, slide, run, swim, paddle, ride, hug, wave, push, carry... —
  and abstractions are forbidden: "squeeze a little closer", "share a
  moment", "move together", "shift", "settle into", "they connect". An
  abstraction gives the model nothing to render, so it fills the clip with
  a morph or a drift. Expressions (smiling) may hang off a movement, never
  BE the movement, and "they hold the moment" is not a movement. Balanced
  against the pace rule by MATCH THE VERB TO THE REAL PACE AND GROUND:
  run/jog/sprint only when the shot really is running AND the ground is
  short (never as a way to cross a route faster), climb/wade/ski only when
  the frames show that terrain. `_CONDENSE_MOTION_SYSTEM` must keep the
  verb rather than generalise it back to "move"/"share a moment". Pinned by
  the same test class.
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
- THE CAST LIST (`Storyboard.characters`, added 2026-07-26) pins each
  person's epithet for the WHOLE movie. Identity rules were enforced
  per-planning-call, but reconcile re-plans only dirty pairs, so a targeted
  `_plan_pairs` call saw two frames and invented its own epithet for
  someone the rest of the film named differently. The saved cast is now
  handed to every planning call (fixed, used verbatim) and merged back
  (`_merge_cast` in clients/openai_client.py — existing entries are NEVER
  rewritten, since their wording is already baked into planned prompts).
  Epithets are baked in at PLAN time, not render time: editing the cast
  must never mark clips stale (unlike `global_motion_prompt`, which is
  prepended at render time and does). Surfaced in storyboard.md, the
  preview HTML, and the panel's Cast editor.
- REFER TO PEOPLE BY APPEARANCE ONLY in motion prompts: never names or
  relationship/role words ("the son", "the dad", "the couple") — the video
  model sees only pixels and guesses who is who (a real clip prompted "the
  son splashes the water" with two men in frame). Every person gets a short
  visible-appearance epithet ("the bald man", "the younger brown-haired
  man"), reused consistently; an action belonging to one person names that
  person, never a bare collective "they". Enforced in `_MODE_A_SYSTEM`,
  `_CONDENSE_MOTION_SYSTEM` (condense swaps relationship words for
  epithets), and `_REWORD_MOTION_SYSTEM` (identity anchors are not "risky
  detail" to drop); pinned by TestIdentityPromptRules.
- IDENTITY IS ITS OWN STEP, BETWEEN PLANNING AND RENDERING (user call,
  2026-08-02). The flow is PLAN → CORRECT → RE-PLAN → render: `storyboard`
  styles, plans, builds the cast AND ends by proposing who is in each
  untagged photo (`_propose_tags`, skippable with `--no-tag`) — the cast
  only exists once the planner has built it, so that is the earliest a tag
  can point at anything. The panel's stepper has a matching step 2,
  "People", holding the fragile-epithet warning, the tag count, the AI
  proposal and `--replan-all`. Nothing already planned uses corrected tags
  or epithets until its pair is planned AGAIN, and a plain `storyboard` run
  won't do it (it reconciles), so `--replan-all` / "Re-plan all with these"
  is the action that closes the loop. Keep the tag pass best-effort (a
  failed proposal must never lose the plan that was just written) and keep
  the step OPTIONAL — an untagged movie is a normal movie, so `next_step`
  must not start demanding tags on every existing project.
- WHO IS IN EACH FRAME IS A HUMAN'S CALL, NOT THE PLANNER'S (user request,
  2026-08-02). The cast fixes what a person is CALLED; which of them stands
  in a given photo is a separate judgement and the one the planner is worst
  at — it decides from pixels whether the child in frame 7 is the child from
  frame 6, and a wrong guess is what makes Kling morph two people together
  (two siblings at the same age is the classic trap). `Frame.people`
  (`FramePerson`: cast id + normalised x/y) records the answer, tagged by
  clicking faces in the panel's "Who's in each photo" section. `x` gives the
  left-to-right order the video model works in, so tags OVERRIDE the
  planner's own `start_order`/`end_order` in `_coerce_transition_plans` —
  swap detection is only as good as the positions it is given.
  `_tagged_people_block` states the roster per frame AND the per-pair
  consequence as fact (same people → continuous; disjoint sets → exit +
  entrance, never a morph; partial → who stays/leaves/arrives). `tag`
  (`cmd_tag`, panel "Let the AI propose…") is a DRAFT first pass: it fills
  only untagged frames (`--retag` overrides), an empty answer leaves a frame
  untagged rather than recording "nobody", and `_coerce_frame_people` drops
  invented ids/duplicates. Tags are plan-time input like the cast: they
  NEVER mark a clip stale (re-plan to apply), and `_reconcile_storyboard`
  carries them over by output_path or a storyboard re-run would silently
  destroy the hand work. Pinned by TestConfirmedIdentities /
  TestTagsDriveSwapDetection / TestIdentifyPeopleCoercion /
  TestFrameTagsSurviveStoryboardRuns / TestTaggedPeople.
- AN EPITHET MUST OUTLIVE THE PHOTO IT CAME FROM (user call, 2026-08-02).
  The cast is pinned once and reused for the WHOLE movie, but the movie is
  built from photos taken months or years apart — so a clothing-anchored
  epithet ("the boy in the striped shirt") describes one frame perfectly
  and matches nobody in the next, where Kling hunts for a striped shirt,
  fails, and puts the action on whoever is nearest. `_MODE_A_SYSTEM` had
  only a soft "prefer stable traits" line and the planner ignored it: a
  real cast came back as man in blue shirt / girl in purple dress / boy in
  yellow shirt / boy in striped shirt. Epithets must now name what a person
  CARRIES between photos (age band, relative size, hair or its absence,
  facial hair, glasses, build), two look-alikes are separated by RELATIVE
  size ("the taller boy" / "the smaller boy"), and a passing garment may
  only be added inline inside one clip's prompt, never be the epithet.
  ENFORCED IN CODE, like the swap rule: every character also reports a
  `durable_epithet`, and `repair_cast_epithets` swaps it in (plus every
  prompt in that same response) when `is_clothing_anchored` fires —
  refusing a replacement that is itself clothing-anchored or would
  duplicate another cast member. Existing casts are still NEVER rewritten
  (`_merge_cast`), so old projects keep their wording; `snapshot()`'s
  `fragile_epithets` flags them for a hand edit in the panel's Cast editor.
  Pinned by TestCastEpithetsSurviveTheMovie / TestCastPromptForbidsClothing.
- ARRANGEMENT SWAP: when the same people appear in both frames but trade
  left-right positions, hold-steady/world-morphs staging is FORBIDDEN — the
  interpolator maps left onto left, so pinned-in-place swapped people morph
  into each other (real clip 03_to_04 on order am-160726-bd2c grew hair on
  the bald man mid-shot). Stage explicit motion instead: exit past the
  camera + re-enter one at a time (preferred when the setting also
  changes), or one person visibly crossing in front of/behind the other.
  A swap rates difficulty ≥4; swap + setting change = 5. Enforced in
  `_MODE_A_SYSTEM` and the difficulty rubric.
- ...AND THE SWAP RULE IS ENFORCED IN CODE, because prompt text did not
  hold (2026-07-28). The planner missed the same swap twice on real orders
  even with the rule spelled out — a couple trading sides between a boat
  and the salt flat got "stand up ... step closer together, and come to
  rest standing side by side" (rated easy, 5s, no crossing). It SEES
  positions reliably but does not ACT on them, so each transition now
  reports `start_order`/`end_order` — everyone in that frame, left to
  right — as DATA, and code decides: `is_arrangement_swap` (token matching,
  strictest rule first, ambiguity always answers "no swap") forces the pair
  to difficulty ≥4 (10s, since exit + re-entry cannot play in 5s) and, when
  `stages_a_crossing` finds no crossing/exit wording in the prompt, fires a
  targeted `_restage_swapped_pair` rewrite. The rewrite is accepted only if
  it actually contains a crossing; any failure keeps the original prompt.
  Same shape as the word-cap condense: model for perception, code for the
  decision. Pinned by TestArrangementSwapDetection /
  TestSwappedPairsAreForcedLong / TestSwappedPairsAreRestaged.
- PACE AND DISTANCE: a motion prompt that spans a ROUTE makes Kling cover
  that route inside the clip — the people sprint or skate across the ground.
  Two real clips died this way ("stroll side by side along the meadow trail
  until they stop beside the bench"; "stroll down toward a green resort
  lawn"). A gentler verb does NOT fix it — "stroll"/"amble"/"wander" still
  say "cover this whole route now"; the DISTANCE has to shrink. Prompts
  describe the ARRIVAL only ("takes the last few unhurried steps to the
  bench and settles onto it") and name the pace. Genuinely far-apart frames
  aren't a walking shot at all: rate 4-5 and stage exit-past-camera +
  re-entry, or a world morph around a steady subject. Enforced in
  `_MODE_A_SYSTEM`, `_CONDENSE_MOTION_SYSTEM` and `_REWORD_MOTION_SYSTEM`
  (pace anchors are not "risky detail" to drop).
- The WORD CAP DOES NOT CATCH BEAT OVERLOAD: a real 10s prompt ("they
  laugh, step back from the edge, and stroll down toward a green resort
  lawn where they change into white robes, set backpacks aside, and move
  together into a relaxed selfie") was only 46 words — inside the 60-word
  cap — but SIX beats, and rendered as a frantic sprint. `_MODE_A_SYSTEM`
  therefore tells the planner to COUNT beats explicitly (chained
  and/then/where/until = another beat) on top of the word cap. Beats are
  not mechanically enforced — only word counts are.
- Clip durations must LEAN SHORT, and prompt-side bias alone cannot deliver
  it: real plans came back all-5s under "prefer 5" and all-10s under a
  hard-transition checklist (in a photo-album movie nearly every pair
  changes setting or outfit). The planner therefore only RATES each pair's
  difficulty 1-5 (3 = one major change, 4 = two at once OR any transition
  that can't physically play out in 5s, 5 = shares almost nothing /
  different people) and code derives durations: >=4 → 10s, capped at
  `config.long_clip_max_fraction` of the clips (default 0.5, was a hardcoded
  1/3 until 2026-07-26 — the cap was demoting genuinely hard pairs to 5s;
  `_select_long_clips` in clients/openai_client.py). A hard pair squeezed
  into 5s visibly teleports (seen in a real render), so keep both sides of
  the balance — raising the fraction also raises cost, since a 10s clip is
  about twice a 5s one.
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
- SUBJECTS OUTRANK BACKGROUND in styling (user call, 2026-07-26). A real
  4-person photo against a stone wall + woodpile came back with the wall
  and logs lovingly detailed while the faces went generic and one man's
  wraparound sunglasses changed shape and colour. `style_prompt` now says
  the detail budget goes to faces, each face in a group gets solo-portrait
  care, and "approximate background + exact faces = SUCCESS; detailed
  background + generic faces = FAILURE". It also tells the model to CROP IN
  rather than invent scenery when reshaping a portrait photo to 16:9 —
  extending the sides is what shrank the people and spent the budget on
  background in the first place (`_IMAGE_API_SIZE` is 1536x1024, so a
  portrait source is always reshaped).
- ...BUT "CROP IN" ALONE CHOPPED HEADS OFF (user report, 2026-07-28). A
  vertical photo of a couple on a salt flat came back with the man's scalp
  cut away at the top edge: a portrait source must lose height to become
  16:9, and nothing said WHERE that height comes from. `style_prompt`'s
  FRAMING section now opens with "NEVER CUT A PERSON OFF" and says it
  OUTRANKS the keep-them-large/crop-in bias, that the height comes off the
  BOTTOM (feet/legs/foreground) and off empty sky only down to a band of
  headroom, and that the last resort is pulling the camera BACK (smaller
  people are acceptable, a cropped head is not). It also protects sources
  that are ALREADY cropped — the rule must never become "hallucinate the
  missing top of this head". Pinned by TestStylingKeepsWholeSubjectsInFrame
  against the live config.json. If prompt-side framing proves unreliable,
  the deterministic fix is padding a portrait source to 3:2 before upload
  in `prepare_image_for_upload` — not weakening this text.
- THE APP LEARNS FROM RENDERED CLIPS (2026-07-31). The planner writes each
  motion prompt blind — it never sees the clip it caused — so every rule in
  this file got here by a human watching a render and editing prompt text by
  hand. `feedback <project> "..." --clip ID` (panel: the Feedback button on
  a rendered clip) saves the note WITH the exact motion prompt that produced
  the clip, then distils it into one short GENERAL rule appended to every
  later planning call (`_MODE_A_SYSTEM` + `lesson_prompt_block`); scope
  `style` rules join the style prompt instead.
- ...AND THE CLIP ITSELF IS A WITNESS (2026-08-02). A human note is brief
  ("looks weird"); the fault is precise. So feedback also runs a REVIEW
  (`OpenAIClient.review_clip`, on by default, `--no-watch` / the panel's
  checkbox): `sample_clip_frames` cuts the mp4 into `clip_review_frames`
  stills with ffmpeg (free, local, temp dir — never kept), and the vision
  call sees those plus both key frames, the prompt and the user's note. It
  returns what happened + named faults, and BOTH accounts go into
  `distill_lesson`, so rules describe a mechanism rather than a mood. It
  also proposes a corrected motion prompt + duration, which is RETURNED,
  never applied: the panel drops it in as an unsaved storyboard edit and
  the existing save→outdated→confirmed-render path takes over (clips are
  still never auto-re-rendered). Guards that must stay: the reviewer gets
  `_MODE_A_SYSTEM` + the active lessons so its rewrite obeys the same
  rulebook; `_coerce_review` holds the suggestion to the word cap
  (condensing like a planned prompt), rejects impossible durations, and
  falls back to the ORIGINAL prompt when the model returns nothing (so
  "apply" can never blank a transition); the reviewer is told which part of
  the prompt is the auto-prepended `global_motion_prompt` so its rewrite
  doesn't double it; and a failed/impossible review (unrendered clip, no
  ffmpeg, quota) is a reported `review_error`, never a lost note. Pinned by
  TestClipReview / TestClipReviewCoercion / TestClipFrameSampling. Rules are STUDIO-WIDE
  (`projects/_learning/lessons.json`; `_learning` is a reserved project
  name, filtered by `is_project_dir`) — a lesson learned on one order must
  not be relearned on the next. Invariants: the note is never lost (a failed
  distillation is reported and the note still saved), an EMPTY distillation
  is a valid answer (a rule that teaches nothing competes with the
  instructions it refines), with zero lessons the prompt is byte-for-byte
  what it was before, and a wrong rule is switched OFF (kept for the record)
  rather than deleted. `lessons` lists/edits them; `learning_enabled` and
  `max_lessons_in_prompt` (newest win) bound the mechanism. Pinned by
  tests/test_feedback.py + tests/test_server_learning.py.
- MONEY IS TRACKED PER PROJECT, AS AN ESTIMATE (2026-07-31). Every completed
  API call prices itself into `projects/<n>/logs/costs.json` from the
  `pricing` block in config.json — OpenAI calls report through
  `OpenAIClient.on_spend`, so the condense/restage/reword calls the runner
  never sees are counted too. NEVER present these as billed amounts: nothing
  reads a provider's billing API, so every surface (status, panel, README)
  says "estimated". Projects that predate the ledger are valued from the
  FILES ON DISK and flagged (`estimated: true`) — a finished movie reading
  "$0 spent" is worse than an approximation. Recording is best-effort by
  construction: an unwritable ledger is a warning, a dry-run books nothing,
  and a clip is booked when it is DOWNLOADED (a submitted-but-uncollected
  render is money the ledger can't see yet — that's what `pending_renders`
  is for). `costs` reports across projects. Pinned by tests/test_costs.py.
- BALDNESS GETS "CORRECTED" BY THE IMAGE MODEL unless forbidden: a real
  styling gave a cleanly bald man a horseshoe of dark hair around his ears
  (frame 920acc69 on Ari&Michal), and changed wraparound mirrored
  sunglasses into square white-framed ones. This is not only cosmetic — the
  motion prompts identify people by visible appearance ("the bald man in
  pink sunglasses"), so a styling that adds hair breaks the epithet Kling
  is shown. `style_prompt` now forbids restoring/improving hair in either
  direction and treats eyewear shape/colour as an identifying feature.

## Working rules (the user's standing instructions)

1. **Branch flow (since 2026-07-18): work lands on `dev`** (or a feature
   branch merged into `dev`); `main` is production — pushing `main`
   auto-deploys to the Mac mini via the self-hosted runner
   (`.github/workflows/deploy.yml`). Never commit straight to `main`;
   releasing is a PR `dev` → `main`. If the user asks for something odd,
   confirm the branch first.
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
images) → `storyboard` (stops for review; writes json/md/preview.html;
`tag` then states who is in each frame, which later plans are told as fact)
→ `render` (plan + confirm; `--clip ID` redoes one clip) → `audio` → `combine`
→ `publish` (web orders only: delivers the movie into the order's Cloudinary
folder as the next `final_vN`); `run` chains them (up to `combine`) with
confirmation gates. Alongside: `feedback <project> "..." --clip ID` teaches
the planner from a rendered clip, `lessons` shows/edits what it learned, and
`costs` reports estimated spend (both project-less, like `orders`).

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
- `pipeline.py serve` (`server.py`) is the admin panel + API + order intake:
  FastAPI, token auth via ADMIN_API_TOKEN in .env (Bearer header everywhere;
  `?token=` is accepted ONLY on the media files route, for `<img>`/`<video>`
  tags), serial background JobRunner (one pipeline command at a
  time, whitelisted commands/options), and the panel's static build mounted
  at `/`. Orders are fetched LIVE on request (/api/orders); the background
  watcher thread is OPT-IN (`watch_enabled`, default False — the user
  explicitly rejected background polling 2026-07-18: "a simple refresh to
  the UI should just fetch the new orders"; intake is a button click in the
  panel). Only when the watcher is enabled does `watch_auto_storyboard`
  spend OpenAI credits automatically per paid order.
- The **admin panel lives in THIS repo** (`admin_ui/`, React + Vite + Mantine; decided
  2026-07-18 — it's the pipeline's own UI, deliberately decoupled from the
  animoments storefront; the frontend repo's `admin-panel` branch is
  superseded and should not be merged). Build with `cd admin_ui && npm
  install && npm run build`; `serve` serves `admin_ui/dist` at `/` (API
  routes win). `npm run dev` proxies `/api` to 127.0.0.1:8300. The panel
  covers the full CLI surface (init/photo upload, storyboard incl. --idea
  options, per-clip render/audio/re-plan, combine toggles, run, per-clip
  feedback, and the Spending / Learning tabs) — keep that parity when adding
  CLI features. The layout is a numbered pipeline stepper
  (Storyboard → Render → Audio → Combine + a dashed "Run everything" tile),
  and EVERY mutating action goes through a confirmation modal that lists
  exactly what will change and whether money is spent (user requirement,
  2026-07-18) — put new actions behind `ask({...})`, never a bare run.
  Media (styled frames, clips, final video) is served under a stable
  filename and REPLACED in place, so every `<img>`/`<video>` src carries a
  `v=` cache-buster (`fileUrl`'s 4th argument, bumped by the panel's poll
  and once more when a job settles); without it a regenerated frame kept
  showing its previous version and looked like the button did nothing.
- THE CLOSING LETTER IS AUTHORED IN THE PANEL (2026-07-28). combine could
  always be TOLD to scroll `projects/<n>/letter.txt` (`--letter` /
  `closing_letter`), but nothing could WRITE it — behind the tunnel that
  meant a shell on the mini, so the toggle was unusable remotely and
  silently produced no letter. Now: `PUT /api/projects/<n>/letter`
  (`read_letter`/`save_letter`/`letter_state` in media/letter.py — blank
  text DELETES the file so "no letter" is one state), the text rides in the
  detail response as `letter_text`, `snapshot()["letter"]` is
  `{exists, chars}` (status + panel show it; the Combine modal warns when
  the toggle is on with `chars == 0`), and ingest SEEDS the file from the
  order's Firestore `blessing` (`_seed_letter_from_order`, best-effort,
  never overwrites an existing letter). Saving is free and local, so it is
  inline like the storyboard save, not an `ask({...})` job.
- PUBLISHING IS ADDITIVE, FOREVER (2026-07-28). `pipeline.py publish` /
  panel step 5 / `POST /api/.../actions/publish` uploads
  `output/final_video.mp4` back INTO the order's own Cloudinary folder as
  `final_v1`, `final_v2`, ... **Nothing there is ever replaced or deleted**
  — that folder holds the customer's own photos, and a delivered movie may
  already be in their hands. Four guards, all of which must stay: the next
  version is `max(...)+1` over the Cloudinary listing (by tag AND by prefix,
  unioned) MERGED with the local `projects/<n>/published.json` history, so a
  listing hiccup can't reuse a number; the upload sends `overwrite=false`
  and an "existing" response is raised, never absorbed; `published.json` is
  append-only; and the panel PINS the approved name into the job
  (`publish_as`), which refuses to upload under any other name — an approval
  is for one exact file name, shown in the modal before anything moves.
  Publish is delivery, not generation: no API credits, and it is the one
  action whose target name comes from the network, hence the read-only
  `GET /api/projects/<n>/publish/preview` that fills the modal. Uploads go
  chunked (movies exceed Cloudinary's single-request cap) and the cloud's
  `folder_mode` decides whether `asset_folder` rides along. Pinned by
  tests/test_publish.py + tests/test_server_publish.py.
- ...AND EVERY DELIVERED VERSION IS ARCHIVED LOCALLY as
  `output/published/final_vN.mp4` (`_archive_published_movie`, config
  `publish_keep_local_copy`, default on). `output/final_video.mp4` is
  rebuilt IN PLACE by the next combine, so before this the bytes a customer
  actually received survived only in Cloudinary. The copy happens AFTER the
  upload succeeds and may never sink a delivery that already happened — an
  OSError (full disk) is a warning plus an empty `local_file` in
  published.json, never an exception — and an existing archive file is left
  untouched, never overwritten. Served to the panel through the media route
  as its own kind (`published`), since that route refuses path separators.
- The **Firestore order ledger** (`clients/firebase_client.py`, REST +
  google-auth, NOT the heavy firebase-admin SDK) is the watcher's order
  source when a service-account key exists (FIREBASE_SERVICE_ACCOUNT in
  .env, or firebase-service-account.json at the repo root — gitignored,
  never commit it). The frontend writes one doc per paid order (collection
  `orders`, doc id = `AM-...`: name/phone/email/packageId/musicMood/
  blessing/folder/status "new"). The watcher checks photo completeness in
  Cloudinary (exact `photoCount` if the doc ever carries one — today's
  frontend doesn't save it — else the quiet period) and writes status back:
  new → ingesting → ingested (+ `project`). "ingesting" still counts as
  pending so a crashed ingest self-heals; statuses PAST "ingested" (a
  future "delivered") are never downgraded. No key → pure Cloudinary
  polling fallback, exactly the old behaviour.
- Newer frontend versions append the music mood to the order folder leaf
  (`..._HH-MM_warm-piano`); `_FOLDER_RE` in intake.py tolerates the suffix.
- `Pipeline.snapshot()` is the single source of truth for project status
  (cmd_status prints it, the API returns it). `projects/<name>/order.json`
  (written by ingest) ties a project to its Cloudinary order folder; the
  watcher/`orders` listing use it to know what's already handled.
- Install packages with `.venv/bin/python -m pip`, never `.venv/bin/pip`
  (the dev venv's `pip` shim points at a different interpreter).
- **Two checkouts since 2026-07-26.** Production runs from
  `~/production/ai-video-maker` on `main` (its own `.venv`, its own
  `admin_ui/dist`, and it OWNS `projects/` — the dev tree symlinks to it).
  Development happens here, in `~/Documents/code/ai-video-maker`. Editing or
  building here no longer touches production: `npm run build` in the dev
  tree is safe again. Deploys (`deploy/deploy.sh`, `REPO_DIR`) target the
  production checkout only.
- Long-term direction (agreed 2026-07-16): remaining phases — payment
  webhook from the frontend (exact photo_count completeness), delivery step
  (upload final + customer email), later move off the Mac to a VPS behind a
  tunnel. Keep review gates human; automate only the plumbing.

## Production (since 2026-07-18)

- **`deploy/PRODUCTION.md` is the runbook.** The server runs on the Mac mini
  as a launchd LaunchAgent (`deploy/com.animoments.pipeline.plist`) bound to
  **127.0.0.1 only — never 0.0.0.0, never a forwarded port**; the internet
  reaches it exclusively through a Cloudflare Tunnel, with Cloudflare Access
  (email OTP) in front of the app's own token — two auth layers. Server
  logs: `~/Library/Logs/animoments/serve.log`.
- CI (`.github/workflows/ci.yml`): offline tests + pyflakes + panel build on
  every push/PR to dev/main. CD (`deploy.yml`): push to `main` → the mini's
  self-hosted runner runs `deploy/deploy.sh` (refuses a dirty tree, ff-only
  merge, retests on the machine, rebuilds the panel, kickstarts the service,
  health-checks). **CI must stay keyless/offline — it can never be allowed
  to spend API credits.**
- Admin API security invariants (pinned by tests/test_server_auth.py):
  constant-time token compare; ADMIN_API_TOKEN ≥ 16 chars enforced at boot;
  per-address lockout (10 bad tokens / 15 min → 429; keyed on
  CF-Connecting-IP behind the tunnel); `/docs`/`/openapi.json` disabled;
  `admin_cors_origins` defaults to `[]` (the panel is same-origin); photo
  uploads capped at 40 MB/file. When adding endpoints, use the `guarded`
  (Bearer-only) dependency — `media_guarded` (also `?token=`) is reserved
  for routes that feed media tags. Don't loosen any of these.

## Gotchas / facts sessions keep rediscovering

- **NEVER `git add -A` blindly in this repo; read `git diff --cached`
  before every commit.** On 2026-07-26 a blanket add committed a `projects`
  SYMLINK (the dev tree points at the production checkout). `.gitignore`
  said `projects/`, which matches a DIRECTORY only, so the symlink was not
  ignored. The next deploy's `git checkout main` materialised it over the
  real directory and **destroyed 606 MB of styled frames, rendered clips,
  storyboards and finished movies for four projects — permanently**, since
  the machine had no backup. The pattern is now `/projects` (anchored, no
  slash), but the habit is the actual guard.
- **Git treats IGNORED paths as expendable.** `checkout`/`merge` overwrite
  and delete them with no prompt and no "would be overwritten" error — that
  protection only applies to *untracked-but-not-ignored* files. "It's in
  .gitignore" means git will destroy it without asking, not that it is safe.
- **Verify `git status` AFTER committing.** The first attempt to fix the
  above shipped a half-fix: `git rm --cached` was staged but the `.gitignore`
  edit was not, and `git commit` silently committed only the staged part, so
  the dangerous pattern reached `main` anyway.
- **`projects/` is irreplaceable and (as of 2026-07-26) UNBACKED** — no Time
  Machine destination, no snapshots. Its contents cost real API credits.
  Count/checksum before and after ANY operation that moves, relinks or
  recreates it, and refuse to do such an operation while a job is running.
- `projects/` is gitignored and holds the user's real movies (e.g. `Matan`,
  `Entebbe`, `Hila`) — treat their contents as user data: don't delete,
  regenerate, or overwrite without asking. `ai_video_maker.egg-info/` is
  generated packaging metadata; ignore it.
- TEXT THAT IS IN THE PHOTO STAYS (user call, 2026-07-26). Only text the
  video model INVENTS is unwanted; a shop sign, birthday banner or shirt
  logo is scenery and must survive the clip. So `fal_negative_prompt` names
  overlay terms only (`text overlay`, `subtitles`, `captions`, `watermark`)
  and must never carry a bare `text`/`on-screen text`, and `_MODE_A_SYSTEM`
  says "no NEW text appearing on screen" while explicitly protecting text
  already in the frames. (Consistent with `avoid_text_only_frames`, which
  bans text-ONLY frames, not text within a real scene.) Pinned by
  TestInSceneTextIsPreserved, which reads the repo's live config.json.
- Clip requests carry `fal_negative_prompt` (curated artifact preset in the
  shared config) and optional `fal_cfg_scale`; `fal_extra_arguments` is
  applied last and overrides both. All three join the render FINGERPRINT,
  but only when set — an unset knob keeps pre-existing fingerprints valid so
  already-paid pending renders stay resumable. `cfg_scale` is deliberately
  left at the provider default until a real A/B on one clip says otherwise.
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
  replaces a placeholder, the pair's rendered clip is marked outdated
  (never deleted). `snapshot()["storyboard"]["placeholder_transitions"]`
  surfaces them. A real order rendered 26/29 clips with the generic prompt
  before these guards existed.
- fal Kling durations are the string enum "5"/"10" (`fal_duration_as_string`);
  valid clip durations live in `constants.VALID_DURATIONS`.
- Clip renders use fal's QUEUE API (`FalSession.submit` + `wait_for_result`),
  never `subscribe`: subscribe holds a connection for the whole render, and a
  drop mid-wait made `with_retries` resubmit the entire job — the first one
  keeps rendering (and billing) server-side with no handle to it. The
  request_id is persisted as state `falreq:<clip>` before waiting and reused
  on the next run when the fingerprint (frames+prompt+duration+model) still
  matches, so interruptions recover the paid output instead of re-buying it.
  Keep the entry on transient failures; clear it only on moderation
  rejection, fingerprint mismatch, or successful download. NOTHING POLLS
  THESE IN THE BACKGROUND — an uncollected job is only fetched by the next
  `render` of that clip, so `Pipeline.pending_renders()` surfaces them in
  snapshot/status/panel (with `recoverable`, i.e. does the fingerprint still
  match), and saving a storyboard edit that invalidates one reports it as
  `orphaned_renders` rather than losing the money silently. `subscribe` is
  still fine for the cheap audio jobs.
- Clips are named `<startid>_to_<endid>.mp4`; bridged clips (a missing middle
  frame) get non-consecutive names like `003_to_005.mp4` and become "stray"
  once the frame is restored — `combine` ignores strays by design.
- A COMBINE FIX ISN'T VISIBLE UNTIL THE SEGMENT CACHE LETS GO. Sections in
  `output/segments/` (intro, credit stills, letter, credits+letter) were
  reused on mtime alone, and shipping code touches no project file — so
  after the emoji and no-greyed-photos fixes deployed, the user re-combined
  and got a byte-identical ending. `--force` didn't help either: it gated
  only the final-video rebuild, never `_segment_fresh`. Now each section
  writes a `<segment>.recipe` sidecar holding a hash of its settings plus
  `_SEGMENT_RENDERER_VERSION`, and reuse needs mtimes AND a matching
  recipe, with `self.force` short-circuiting the whole check. **Bump
  `_SEGMENT_RENDERER_VERSION` whenever you change how a section is drawn**
  — that is the only thing that reaches already-rendered projects. Pinned by
  TestPresentationSegments (version bump / changed setting / force).
- PHONE UPLOADS HAVE NO FILE EXTENSION, so nothing may validate on one. A
  real mp3 from iOS arrived named `Lights(chosic.com)` and the music upload
  refused it (extension-only check), while `accept="audio/*"` on the input
  greyed the same file out in the Files picker before it could even be
  chosen. Now: `looks_like_audio` (media/audio_files.py) decides from magic
  bytes → declared MIME → extension, any one of which is enough, and the
  panel's accept list is deliberately wide (includes
  `application/octet-stream` and bare extensions) because the server checks
  content anyway. The same trap is live for PHOTO uploads
  (`_PHOTO_EXTENSIONS` in server.py) — fix it the same way if it bites.
- THE CREDITS PHOTOS ARE NEVER GREYED OUT under a scrolling letter (user
  call, 2026-07-28). `_letter_overlay_cmd` used to lay a fixed `black@0.55`
  drawbox over the montage for text contrast; the photos are the product,
  so the scrim is gone (`letter_overlay_dim`, default 0) and the contrast
  comes from a stroke-outline drop shadow drawn into the transparent letter
  PNG (`_SHADOW`, only when `transparent=True` — the standalone letter is
  already on a dark background). Pinned by
  test_letter_overlay_keeps_the_photos_at_full_brightness.
- THE LETTER FONT HAS NO EMOJI, and a blessing usually ends in a heart: a
  real delivered movie showed "לאהבת חיי ט״ו באב שמח" followed by two empty
  boxes (the emoji + its variation selector, both .notdef). media/letter.py
  now splits each line into runs — text through the Hebrew font, emoji
  through a colour font (`find_emoji_font`, config
  `letter_emoji_font_path`; Apple Color Emoji on the mini, Noto Color Emoji
  on Linux/CI). Colour fonts are BITMAP STRIKES: `ImageFont.truetype` raises
  OSError('invalid pixel size') at any other size, so `load_emoji_font`
  probes known strikes (109 Noto, 137/160 Apple) and the emoji is drawn at
  that size then scaled to the text. Emoji are identified by CODE POINT
  (`_EMOJI_RE`) and dropped when no emoji font exists — never boxed — along
  with the space left dangling in front of them (it offsets the centring).
  **NEVER ask the font whether it has a glyph.** The first version did:
  Pillow exposes no glyph-index API, so it compared each character's
  rendered bitmap against the font's .notdef box. On Linux that worked; on
  the mini's Arial Hebrew it reported EVERY character as missing, so
  `_segments` returned nothing and a letter would have rendered blank. The
  deploy's own test run caught it (5 failures, service never restarted) —
  which is exactly what that test run is for, since the emoji tests are
  font-gated and only execute where a colour font exists. Apple Color Emoji
  itself is confirmed working there. Pinned by TestSegments (plain text and
  Hebrew must survive segmentation) / TestRenderWithEmoji.
- ffmpeg concat: mixed silent/sounded clips must go through the concat-filter
  path with silent padding (`_combine_clips_mixed_audio`), never the demuxer.
