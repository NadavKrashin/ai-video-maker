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
python pipeline.py publish myfilm         # 5. (web orders) deliver it into the order's Cloudinary folder

python pipeline.py status myfilm          # where am I? what's next?
python pipeline.py run myfilm             # or: everything in one go (with confirmations)

python pipeline.py feedback myfilm "the boy slid across the lawn" --clip 2_to_3
                                          # tell the planner what a clip got wrong;
                                          # it learns a rule for every future movie
python pipeline.py costs                  # what each project has cost (estimated)
```

**The storyboard is the source of truth.** `storyboard` writes it, you edit it,
`render` executes it exactly. Re-running `storyboard` never overwrites your
edits (it reuses the saved storyboard while your images are unchanged); pass
`--force` to redo styling + analysis from scratch.

Whenever a step finishes, the app prints the exact command for the next step,
and `status` will always tell you where you stand.

### Paid web orders (Cloudinary intake)

When an order comes in from the animoments web app, its photos are already in
Cloudinary (one folder per order, photos named by their position in the movie).
Instead of `init` + a manual download, pull the order straight into a project:

```bash
python pipeline.py orders                   # list waiting orders, newest first
python pipeline.py ingest matan AM-20260716-XY12   # create projects/matan/ + download
python pipeline.py storyboard matan         # ...and continue as usual
```

The order argument is the order id from the confirmation email, the full
folder name, or any unique fragment of it (e.g. the customer's name).
Photos are saved as `01.jpg, 02.jpg, ...` so the movie keeps the customer's
chosen order; re-running skips files that already exist (`--force`
re-downloads, `--dry-run` just lists). Ingesting into a project that already
holds a *different* order's images fails loudly — one project per order.

Setup: `config.json` carries the (public) `cloudinary_cloud_name` and
`cloudinary_orders_folder`; the Admin-API credentials go in `.env` as
`CLOUDINARY_API_KEY` / `CLOUDINARY_API_SECRET` (Cloudinary console →
Settings → API Keys).

### Delivering the finished movie back (`publish`)

An order's photos come *from* Cloudinary, and the finished movie goes back
*into* the very same folder, next to them:

```bash
python pipeline.py publish matan            # asks first, showing the exact file name
python pipeline.py publish matan --dry-run  # just print the name it would use
```

```
video-orders/AM-20260716-XY12_Dana-Cohen-16.07.2026_10-30/
    1.jpg, 2.jpg, ...     # the customer's photos (uploaded by the web app)
    final_v1.mp4          # ← the delivered movie
    final_v2.mp4          # ← a later cut, delivered after a fix
```

Publishing is **strictly additive — nothing in Cloudinary is ever replaced or
deleted.** Each publish takes the next free version number: the versions
already in the order folder (asked live) are merged with this project's own
delivery history (`projects/<name>/published.json`), the highest wins, and the
upload itself carries `overwrite=false` so Cloudinary refuses to replace an
asset even if that number were somehow wrong. Rebuild the movie and publish
again and the customer folder simply gains a `final_v2` beside the `final_v1`
they may already have been sent.

The confirmation always names the file first (`final_v2.mp4`, plus its full
public id) — in the terminal, and in the panel's modal. In the panel the name
shown for approval is *pinned into the job*: if someone published from
elsewhere in between, the job stops rather than uploading under a name nobody
approved.

**Every delivered version is also kept locally**, as
`projects/<name>/output/published/final_vN.mp4`. `output/final_video.mp4` is
rebuilt *in place* by the next Combine, so without that copy the bytes a
customer actually received would exist only in Cloudinary — you could not
re-send, re-check or compare an earlier cut. The copy is taken after the
upload succeeds and can never sink a delivery that already happened: if it
fails (a full disk), the publish is still recorded and the log says the
archive is missing. Each version is downloadable from the panel's Publish
step. Set `publish_keep_local_copy: false` in `config.json` to skip the
copies — a movie is 100–300 MB per version.

`status` and the panel show what has been delivered, and flag a movie that has
been re-combined since its last publish (`next_step: publish`). Only projects
created by `ingest` can be published — a hand-made project has no order folder
to publish into. The basename is `cloudinary_publish_basename` in
`config.json` (default `final`).

### The admin server (`serve`) — run the whole flow from a browser

```bash
cd admin_ui && npm install && npm run build && cd ..   # once (and after UI changes)
python pipeline.py serve            # panel + API on 127.0.0.1:8300 + order watcher
```

One process serves the whole thing — open http://127.0.0.1:8300/ and enter
the `ADMIN_API_TOKEN` from `.env`:

- **Admin panel** (`admin_ui/`, a small React + Mantine app served from its `dist/`
  build): everything the CLI can do, from a browser — order intake, creating
  a project from scratch and uploading/removing photos (`init`), storyboards
  from photos or from a typed idea (with style prompt / duration /
  no-analyze / frame-count options), regenerating a single styled image,
  reviewing and editing every transition (motion prompt, duration, sound
  prompt, global motion prompt), rendering all clips or regenerating one,
  per-clip audio redo, uploading the music track, combine/finalize with
  intro/credits/letter toggles, **writing the closing letter itself** (a
  Hebrew-safe text box in the Combine step, saved as the project's
  `letter.txt`), **publishing the finished movie into the order's Cloudinary
  folder** (step 5 — the modal names the exact file, e.g. `final_v2.mp4`,
  before anything is uploaded, and nothing there is ever replaced), and `run`
  for the whole chain, plus a **Feedback** button on every rendered clip
  (tell the planner what it got wrong — see
  [Teaching the planner](#teaching-the-planner-feedback--lessons)) and two
  studio-wide tabs: **Spending** (what each project cost and the total) and
  **Learning** (every rule the planner has learned, editable and switchable).
  Jobs stream their logs and can be cancelled from the panel.
  **Editing prompts in bulk:** saving your transition edits marks every
  already-rendered clip whose motion prompt / duration / frames changed as
  OUTDATED (a sound-prompt-only edit doesn't — that's the audio step's
  business), and then offers **“Generate everything that needs it”** — one
  background job covering the edited clips *and* the ones never rendered, so
  a round of prompt edits doesn't have to be clicked through clip by clip.
  The confirmation lists every clip by name and which of the two groups it
  is in before anything is spent; the same button lives in the Render step.
  While a storyboard job runs, the Photos panel opens itself and fills in
  live — styled frames appear one by one as they come back, instead of only
  once the whole run finishes.
- **API**: order list, per-project status, storyboard read/edit, photos, a
  music-bed upload (`POST`/`DELETE /api/projects/<name>/music` — the track is
  recognised by its CONTENT, so a file whose name has lost its `.mp3`, as
  phone uploads often have, is still accepted), the closing
  letter (`PUT /api/projects/<name>/letter` with `{"text": "…"}`; the text
  comes back in the project detail as `letter_text`, and its summary as
  `letter: {exists, chars}`), a publish preview
  (`GET /api/projects/<name>/publish/preview` — read-only; returns the next
  free version and the exact `public_id`/`filename`, plus what the order
  folder already holds), and
  clip playback, plus actions (ingest / storyboard / render / redo one clip /
  audio / combine / publish / run) that run as **background jobs** — one at a time,
  with their logs available at `/api/jobs/<id>`. `POST /api/jobs/<id>/cancel`
  cancels a job: a queued job is dropped immediately; a running one shows
  `cancelling` and stops between work items — whatever is mid-generation
  finishes and is kept (already-submitted API work bills either way), so
  re-running the same action later resumes instead of re-paying. Every route
  (except `/api/health`) requires the `ADMIN_API_TOKEN` from `.env` (16+
  chars — the server refuses to boot with a shorter one), passed as
  `Authorization: Bearer <token>`. Only the media file route also accepts
  `?token=` (for `<img>`/`<video>` tags); everywhere else the query form is
  rejected. Token comparison is constant-time, and 10 failed attempts from
  one address within 15 minutes lock it out (HTTP 429) for the rest of the
  window. API docs endpoints (`/docs`, `/openapi.json`) are disabled, and
  cross-origin access is off by default (`admin_cors_origins: []` — the
  panel is same-origin; only set origins if you host it elsewhere).
- **Orders, fetched live**: opening the panel's Orders tab (or `GET
  /api/orders`) reads the sources *at that moment* — no background process
  involved. With the **Firebase order ledger** configured (below) that
  means the Firestore `orders` collection — the authoritative "someone
  paid" record with the customer's details and `status`; without it, the
  Cloudinary folder listing. Each order not yet ingested shows **how many
  photos are actually in its Cloudinary folder right now** (flagged orange
  when the folder is empty, or short of the count the order doc expects) —
  payment confirms *before* the photos finish uploading, so a folder
  existing has never meant the order is whole, and that count is what tells
  you whether it is worth ingesting yet. Ingesting is one click ("Ingest
  photos"), which **only** creates the project and downloads the photos:
  storyboarding styles every photo and plans every pair, so it stays a
  separate deliberate click once you have looked at what arrived. The
  pipeline writes progress back into the order doc's `status`
  (`new → ingesting → ingested`).
- **Optional background watcher** (off by default): set
  `watch_enabled: true` to also have the server re-check every
  `watch_poll_seconds` and auto-ingest a new order once its upload is
  complete (exact `photoCount` when the order doc has one, else quiet for
  `watch_quiet_minutes` — payment confirms *before* photos finish
  uploading, so folder-exists ≠ order-complete). With
  `watch_auto_storyboard` (default on) it also runs `storyboard` right
  away — **this spends OpenAI styling credits automatically per paid
  order** — so the storyboard is waiting for review by the time you open
  the panel. Use it when you want orders handled while you're away from
  the panel; `--no-watch` force-disables it for one run.

For UI development, `cd admin_ui && npm run dev` serves the panel with hot
reload, proxying `/api` to a locally running `pipeline.py serve`.

### Production (the Mac mini)

**[deploy/PRODUCTION.md](deploy/PRODUCTION.md) is the runbook.** The short
version: the server runs as a launchd service bound to `127.0.0.1` only
(`deploy/com.animoments.pipeline.plist`), and the internet reaches it
exclusively through a **Cloudflare Tunnel** (see
`deploy/cloudflared-config.example.yml`) with Cloudflare Access (email +
one-time PIN) in front — two auth layers, zero open ports.

Branch flow: work lands on **`dev`**, a PR into **`main`** is the release.
CI (`.github/workflows/ci.yml`) runs the offline tests + lint + panel build
on every push/PR to either branch; pushing `main` triggers
`.github/workflows/deploy.yml`, which runs `deploy/deploy.sh` on the mini's
self-hosted runner: fast-forward, reinstall, retest, rebuild the panel,
restart the service, health-check. Never bind `0.0.0.0`; never port-forward.

### The Firebase order ledger (optional, recommended)

The web frontend writes every paid order to Firestore (collection `orders`:
customer name/phone/email, package, music mood, blessing, the Cloudinary
photo folder, and a `status` starting at `"new"`). To let the pipeline use
and update that ledger:

1. Firebase console → Project settings → Service accounts → **Generate new
   private key** (this is a secret — it's gitignored).
2. Save it as `firebase-service-account.json` at the repo root, or point
   `FIREBASE_SERVICE_ACCOUNT` in `.env` at its path.

That's all — the project id is read from the key file itself
(`firebase_project_id` / `firebase_orders_collection` /
`firebase_credentials_file` in `config.json` override when needed). The
`orders` API/panel then shows the full customer context per order, and the
watcher stops guessing completeness from folder timestamps alone.

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
| Ask the planner for a fresh motion prompt for one pair (don't like its plan) | `python pipeline.py storyboard myfilm --replan-clip 2_to_3` — its rendered clip (if any) is marked outdated, not redone | 1 small vision call |
| Change one clip's sound (edit its `sound_prompt` first) | `python pipeline.py audio myfilm --clip 2_to_3` | ~1¢ |
| Add an image between 2 and 3 | copy it in as `input_images/2a.jpg`, then:<br>`python pipeline.py storyboard myfilm`<br>`python pipeline.py render myfilm` | 1 styling + 2 clips |
| Remove image 2 | `rm projects/myfilm/input_images/2.jpg projects/myfilm/styled_images/2.png`, then:<br>`python pipeline.py storyboard myfilm`<br>`python pipeline.py render myfilm` | 1 clip |
| Swap image 2 for a different photo | overwrite `input_images/2.jpg` with the new file, then:<br>`python pipeline.py storyboard myfilm` (asks before re-styling)<br>`python pipeline.py render myfilm --clip 1_to_2 --clip 2_to_3` | 1 styling + 2 clips |
| Re-style one image (new roll of the styling dice) | `python pipeline.py storyboard myfilm --restyle-frame 2.png` — its adjacent clips are marked outdated, then:<br>`python pipeline.py render myfilm --clip 1_to_2 --clip 2_to_3` | 1 styling + 2 clips |
| Rebuild the movie after any of the above | `python pipeline.py combine myfilm --force` | free (local) |
| Redo all clips (e.g. after big storyboard edits) | `python pipeline.py render myfilm --force -y` | all clips |
| Redo styling + analysis from scratch | `python pipeline.py storyboard myfilm --force` | all stylings + 1 analysis |

Notes:

- `--clip` is repeatable (`--clip 2_to_3 --clip 3_to_4`) and always
  *regenerates* the named clips, resetting their SFX/fade state so redone
  clips get fresh audio.
- After add/remove/swap, `storyboard` re-plans **only the affected
  transitions** (a small vision call with just those frames) and keeps
  everything else verbatim. **Existing clips are never deleted or redone
  automatically**: a rendered clip whose transition changed is only marked
  OUTDATED (`status` and the admin panel flag it) — regenerate it yourself
  with `render --clip ID` when you're ready to spend the credits (the panel's
  “Generate everything that needs it” does the same for outdated + missing
  clips in one confirmed job). Old clips
  whose pair no longer exists at all become "strays" — `combine` ignores
  them and `status` lists them for deletion.
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

### How people are named (the cast)

Motion prompts never use names or relationships — the video model sees only
pixels, so "the son splashes the water" with two men in frame is a coin flip.
Every person instead gets a short **epithet** by visible appearance, pinned
once in `storyboard.characters` and reused word for word in every prompt that
mentions them (the panel's **Cast** editor shows them).

Because a movie is assembled from photos taken months or years apart, an
epithet must name what the person **carries between photos** — age band or
relative size, hair (colour, length, texture, or its absence), facial hair,
glasses, build:

| Good | Bad |
|------|-----|
| the smaller boy with curly hair | the boy in the striped shirt |
| the taller boy | the boy in the yellow shirt |
| the bald man in pink sunglasses | the man in the blue shirt |
| the teenage girl | the girl in the purple dress |

Clothing is the trap: it identifies someone perfectly in the frame the cast
was built from and matches nobody in the next one, where the video model hunts
for a striped shirt, fails, and puts the action on whoever is nearest. The
planner is told this outright, and — because prompt guidance alone did not
hold it — **code checks**: each character also reports a `durable_epithet`,
and a chosen epithet anchored to clothing is swapped for it automatically
(along with every prompt in the same plan) before anything is saved.

Two people who share every durable trait are separated by relative size
("the taller boy" / "the smaller boy"), which still works in every photo. A
passing detail can still ride along inside one clip's prompt — "the smaller
boy with curly hair, here in a striped shirt" — without changing the epithet.

Epithets must also be **distinct across the whole cast**. On a big-family
order the planner returned "woman with dark hair bun" *and* "young woman with
dark hair bun": a prompt naming one points at both, the video model acts on
whoever is nearest, and the pipeline's own swap detection reads them as the
same person and gives up. The planner is now told to separate look-alikes
cast-wide, and colliding entries are surfaced as
`snapshot()["storyboard"]["indistinct_epithets"]` plus a warning with inline
errors in the panel's Cast editor — existing casts are never rewritten
automatically (their wording is baked into planned prompts), so the fix is a
hand edit followed by re-planning the clips that mention them. A background
crowd that is really scenery (a full dinner table, distant swimmers) may be
one **collective entry** ("the large dinner group") rather than an entry per
stranger.

### Saying who is who (per-frame tagging)

The cast fixes what each person is CALLED. Which of them is standing in a
given photo is a separate judgement, and it is the one the planner gets wrong
on its own: it decides from the pixels whether the child in frame 7 is the
child from frame 6, and when it guesses wrong the video model morphs one
person into the other. Two siblings at the same age are the classic trap.

So you can state it as fact. In the panel's **Who's in each photo** section,
pick a person and click their face; a marker pins them to that spot. What the
planner is then told, per pair, is not a hint but a ruling:

- the SAME people in both frames → animate them continuously, whatever they
  are wearing and however much older they look;
- COMPLETELY DIFFERENT people → they only look alike, so stage an exit and an
  entrance and never let one turn into the other;
- a partial overlap → who stays, who leaves, who arrives.

The marker positions also give the left-to-right order the video model
actually works in, so **arrangement-swap detection uses your tags** instead of
the planner's own reading of the frames (see `--replan-clip`).

**`storyboard` now ends by taking that first pass itself** — the cast only
exists once the planner has built it, so the end of planning is the earliest
moment tags can be proposed at all. That makes the flow three parts, which is
how the panel's stepper is laid out:

1. **Storyboard** — style the photos, plan the clips, name the cast, and
   propose who is in each photo.
2. **People** — the review stop: fix any cast name that describes clothing,
   correct the tags, then **Re-plan all with these** (`storyboard
   --replan-all`) so the prompts are rewritten from what you confirmed.
3. **Render** — buy the clips, now that the prompts say who is who.

`--no-tag` skips the proposal; a standalone pass is still available:

```bash
python pipeline.py tag myfilm          # proposes who is in each untagged frame
python pipeline.py tag myfilm --retag  # redo frames you already tagged, too
```

or **“Let the AI propose…”** in the panel. It is a *draft* — one vision call
over the styled frames, correct it afterwards, especially where two people
look alike. It never touches frames you have already tagged, and a frame it
reports nobody in stays untagged (a question, not an answer).

Tagging is free, local, and feeds the NEXT plan: it never marks a rendered
clip outdated. Nothing already planned uses your tags until its pair is
planned again — so after tagging a movie, use **“Re-plan all with these
tags…”** (in the tagger, and as “Re-plan all prompts…” in the Storyboard
step), or `storyboard --replan-clip ID` for one pair. A plain `storyboard`
run will NOT do it: it reconciles rather than regenerates, so pairs that are
already planned are carried over verbatim. Clips whose plan actually changes
— in wording *or* in length — are then marked outdated for you to regenerate
when you want to spend the credits.

**Re-planning a hand-picked few.** Between "this one pair" and "the whole
movie" there is the usual case: a handful of prompts you want rewritten.
Each clip card has a checkbox, and the bar above the list re-plans exactly
what you ticked in one job (`storyboard --replan-clip A --replan-clip B`
on the CLI). It also offers one-click selection of the sets already worth
re-planning — *needs camera transition*, *behind your tags*, *ends
offscreen*, *generic prompt* — with the count next to each, since those
sets can be large. Prompts you did not select are left untouched, and
consecutive pairs share a single vision call, so a batch costs less than
clicking the same clips one at a time.

> **An existing project keeps the cast it was planned with.** Epithets are
> frozen on purpose: their wording is already baked into planned prompts, so
> rewriting one silently would split a person's identity across the movie.
> The panel flags any that name clothing so you can fix them by hand; re-plan
> a clip (`storyboard --replan-clip ID`) for the new wording to be used.

### Knowing what is out of date

Every step feeds the next, so the useful question is usually "what here no
longer matches what I changed?". `status` and the panel answer it in one
place — nothing is ever fixed silently, and nothing costing money is redone
without you asking:

| What went stale | How you see it | How to fix it |
|---|---|---|
| A **prompt** written before you tagged a photo or renamed someone in the cast | “prompt predates your tags” on the clip; a count in step 2 | Re-plan those pairs (the panel offers exactly the ones behind) |
| A **rendered clip** whose prompt, duration or frames changed since | “outdated” badge; `status` marks it | `render --clip ID`, or “Generate everything that needs it” |
| A **styled frame** whose source photo changed | listed as a changed frame | `storyboard` (asks before re-styling) |
| A **cast name** that describes clothing | flagged in the Cast editor | Rewrite it, then re-plan |
| A **prompt that was never really planned** (a quota failure left the generic fallback) | “generic prompt” badge | Run `storyboard` again |
| A **paid render** that was never collected | listed as a pending render | Collect it (free) before editing that clip |
| The **final movie**, built before a clip was re-rendered | `status` says so | `combine --force` (free, local) |

The rule behind all of them: tags, cast names and lessons are *plan-time*
inputs — they change what the planner is told next time, never a prompt that
already exists — and prompts are *render-time* inputs, so changing one marks
the clip rather than redoing it. Everything downstream is a marked state you
act on, not a surprise bill.

### Teaching the planner (feedback → lessons)

The planner writes each motion prompt **blind**: it sees two still frames and
never sees the clip that comes back. So when Kling does something silly —
sliding a person across the grass instead of walking them, morphing one face
into another — nothing in the pipeline learns from it unless you say so.

That is what `feedback` is for — and it has **two witnesses**, not one:

* **you**, who know which faults matter but are usually brief ("it looks weird");
* **the reviewer**, a vision call that actually WATCHES the clip. The clip is
  sampled into stills with ffmpeg (free, local) and shown to the model next to
  the two key frames and the prompt that produced it, so it can say precisely
  what happened — and propose a corrected motion prompt and clip length.

Both accounts go into the rule, so what gets learned describes a mechanism
rather than a mood:

```bash
python pipeline.py feedback myfilm \
  "the boy slid across the lawn without taking a step" --clip 2_to_3
#   Watching 2_to_3, the reviewer saw:
#     the boy is in a different place at the end but never takes a step;
#     the background stretches to cover the move
#     - subject slides instead of walking
#
#   It suggests rendering this pair at 10s with:
#     the small boy in the red shirt takes the last few steps to the bench
#     and sits down
#
#   -> Learned (motion): When a person stands somewhere different in the end
#      frame, write the walk that gets them there — never state the new
#      position alone.
```

**Nothing is applied or re-rendered automatically.** The suggestion is shown;
adopting it is a storyboard edit you make (which marks the clip outdated), and
re-rendering stays the separate, confirmed, paid action it always was. In the
panel that is one click — “Use this prompt for the clip” drops it into the
transition as an unsaved edit, then the usual Save → “Generate everything that
needs it” flow renders it.

- **`--good`** learns what to KEEP doing from a clip that came out well.
- **`--no-watch`** skips the review; **`--no-learn`** skips the rule.
- **No note at all** is fine when `--clip` is given: the reviewer watches it
  and reports on its own ("just tell me what's wrong with this one").
- **No `--clip`** gives feedback about the movie in general (nothing to watch).
- In the admin panel this is the **Feedback** button on every rendered clip,
  and the **Learning** tab lists every rule.

Rules are **studio-wide** — a mistake corrected on one order must not be
relearned on the next — so they live in `projects/_learning/lessons.json`
(`_learning` is a reserved name, never a project). Review and edit them with:

```bash
python pipeline.py lessons                      # what it has learned
python pipeline.py lessons --add "Keep a 5s clip to one continuous action."
python pipeline.py lessons --disable a1b2c3d4   # stop using it, keep the record
python pipeline.py lessons --forget  a1b2c3d4   # delete it
```

Details worth knowing:

- A rule the model can't generalise is **not** written: "nothing to learn
  here" is a valid outcome, because every rule competes for the planner's
  attention with the instructions it refines.
- Your note is **never lost**. If either call fails (quota, network, an
  unrendered clip, no ffmpeg), the note is still saved and you are told which
  part didn't happen — add the rule by hand if you want it.
- The reviewer is held to the planner's own rulebook: it gets the same
  instructions and the same learned lessons, and its suggested prompt goes
  through the same word-budget check, so accepting one can't smuggle in a
  prompt the planner itself would have been refused.
- Scope `motion` rules join the clip planner's system prompt; `style` rules
  join the image style prompt. Only the newest `max_lessons_in_prompt`
  (default 25) active rules ride along with a call; `learning_enabled: false`
  turns the whole mechanism off without deleting anything.
- With no lessons stored, every prompt is byte-for-byte what it was before
  this feature existed.

### What it costs (spending)

Every paid call — styling an image, a planning/vision call, a clip render, an
SFX pass — is written to `projects/<name>/logs/costs.json` as it happens, so a
project can say what it cost:

```bash
python pipeline.py costs        # every project + the studio total
python pipeline.py status myfilm    # one project's spend, with a breakdown
```

The admin panel has the same in its **Spending** tab, and each project view
shows a "Spent on this project" card that expands into the individual calls.

**These are estimates, not invoices.** Nothing here reads a provider's billing
API: each call is priced from the `pricing` block in `config.json` (`$0.35`
for a 5s clip, `$0.70` for 10s, `$0.19` per styled image, OpenAI text priced
from the token usage the API reports). Correct those numbers when a provider
changes its rates and every total follows.

Projects that predate the ledger have a full movie and an empty ledger, so
their figure is priced from the **files on disk** instead and marked as such
(`~` in the CLI, a "from files" badge in the panel). Recording can never sink
a run: an unwritable ledger is a warning, a dry-run books nothing, and a
render that was submitted and billed but never collected only appears once its
result is downloaded (`status` lists those separately as pending renders).

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
- **Node.js 18+** (only for the admin panel — `admin_ui/` builds with Vite)

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

# Cloudinary Admin API (only needed for `orders`/`ingest` — the web-order
# intake) — from the Cloudinary console, Settings -> API Keys:
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...

# Admin server (`serve`) — any long random string; the panel logs in with it:
ADMIN_API_TOKEN=...

# Firebase order ledger (optional) — path to a Firestore service-account key;
# defaults to firebase-service-account.json at the repo root when present:
FIREBASE_SERVICE_ACCOUNT=firebase-service-account.json
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

**Clip length:** you don't set it per clip. The planner rates each frame pair's
difficulty 1–5 and the code derives the duration — 4 and 5 get 10 seconds,
everything else 5 — because prompt-side instructions alone came back either
all-5s or all-10s. **Every pair that earns it gets the long clip**: there is no
quota. (There used to be a `long_clip_max_fraction` ceiling; it demoted
genuinely hard pairs to 5s, where they teleport, and because it applied per
planning call the same pair could come back a different length depending on
which others you re-planned alongside it.) Rating honestly is the planner's
job — if too much comes back long, that is a rating problem, and a 10s clip
costs roughly twice a 5s one. **Camera transitions are always 5 seconds**,
whatever their difficulty: that prompt is one continuous beat, so ten seconds
of it is a slack shot at twice the price.

**Position swaps are caught in code.** When the same people appear in both
frames but trade left–right places, pinning them where they stand makes the
video model morph each into the other. The planner reports who stands where
in each frame (left to right), and the pipeline compares the two lists
itself: a swapped pair is forced to a 10-second clip, and if its motion
prompt has nobody crossing or leaving frame, a targeted rewrite restages it
as an exit past the camera plus a walk back in. Prompt-side instructions
alone kept missing these.

**Unstageable pairs become camera transitions.** Exit-and-entrance staging
works for a couple of people and falls apart in a crowd: on a real order the
planner staged seven exits and three entrances inside one ten-second clip and
the video model rendered ghost dissolves, bodies mushing into each other, and
people vanishing. The pipeline now decides in code, per pair, whether the
staging can exist at all: more than **3 people** leaving or arriving, or a
frame holding **more than 4 people** whose roster or arrangement changes,
replaces the choreography with a deterministic **camera transition** — the
camera travels to the new setting and settles on exactly what the end frame
shows (with a drift-to-one-person variant when a group narrows to one of its
own, and a scene variant when the end frame is empty). The planner also
reports a per-frame **headcount census** (everyone visible, tagged or not),
so a crowded frame with only two people tagged still gates correctly. Small
pairs keep subject staging — this is the only other place camera language is
allowed, alongside the offscreen last resort below.

The gate acts when a pair is planned, so storyboards written before it keep
their many-mover choreography until re-planned. Those pairs are surfaced —
`snapshot()["storyboard"]["unstageable_pairs"]`, a `status` warning, and a
red **needs camera transition** badge on the clip card — wherever the saved
tags say the staging can't exist and the prompt isn't already a camera one.
Re-planning the pair (its badge's re-plan button, or `--replan-all`) yields
the camera transition.

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
- `fal_negative_prompt` — sent as the model's `negative_prompt` when
  non-empty. The shared config ships a preset targeting artifacts seen on
  real renders (face distortion, morphing, flicker); set it to `""`
  (globally or per project) to send none. It names only *generated overlay*
  text (`text overlay`, `subtitles`, `captions`, `watermark`) — never a bare
  `text`, because text that is genuinely in a photo (a shop sign, a birthday
  banner, a logo on a shirt) is part of the scene and must survive the clip.
- `fal_cfg_scale` — sent as `cfg_scale` when set (Kling: `0`–`1`, provider
  default `0.5`). Higher = stricter prompt adherence, at some cost to motion
  coherence. Leave `null` to use the provider default.
- Add extra model-specific args via `fal_extra_arguments`; they are applied
  last, so they override the fields above (handy for per-project
  experiments).
- Changing `fal_negative_prompt`, `fal_cfg_scale`, or `fal_extra_arguments`
  changes the render **fingerprint**: a pending, not-yet-collected render
  submitted under the old settings can no longer be resumed (the next render
  buys a fresh clip). Collect pending renders before flipping these knobs.

---

## How each step works

### `storyboard` (from images — the main flow)

1. Put source images in `input_images/` (`.jpg`, `.jpeg`, `.png`, `.webp`;
   ordered by natural filename order, so `img2` before `img10`). Phone
   originals are fine as-is: each image is normalised before upload (EXIF
   rotation baked in, iPhone HDR/MPO containers unwrapped to a plain image,
   oversized photos downscaled) — the file on disk is never modified.
2. Every image is styled into a consistent 1920×1080 look
   (`styled_images/001_styled.png`, …). Already-styled images are skipped on
   re-runs.
3. The styled frames are analysed by the vision model, which plans — for each
   consecutive pair — a tailored **motion prompt**, a **per-clip duration**
   (5 or 10s, leaning 5s; 10s is reserved for genuinely hard transitions),
   and a **sound prompt**, and writes
   `storyboard/storyboard.json` + `storyboard.md`. Motion prompts are
   budgeted in BEATS, not words: 5s is one continuous action, 10s at most
   two, because the video model fakes beats it can't fit (whip-pan blurs,
   swapped people). Nothing counts or trims words — the planner is told to
   say the one thing and stop, and a prompt that wanders into scenery, poses
   and atmosphere is worse than a short one, not richer. Write hand-edited
   prompts the same way.

   A prompt whose **last beat leaves someone out of the frame** ("…and walk
   out of frame to the left") is rewritten automatically.
   The clip is pinned to its end frame, which shows those people, so a prompt
   that empties the frame and stops describes a clip that cannot exist — the
   video model answers it with a cut or a teleport. The rewrite adds the walk
   back in. If it still can't — a group of nine cannot walk out and back in
   inside one clip — the pair falls back to a **camera move** to the new
   setting, settling on the group. That is the one place camera language is
   allowed, and it is deliberate: on those pairs the alternative is a cut,
   not a better staging. No plan leaves this step ending offscreen.
   Hand-edited prompts should land on the end frame too.

`--no-analyze` skips step 3's vision call and uses the single global motion
prompt with one duration for every clip. `--duration 5|10` forces one length
for all clips even with analysis on.

If a planning call fails (rate limit, out of OpenAI quota), the affected
transitions get the generic `motion_prompt` from config as a **placeholder**
so the run isn't sunk — `status` (and the admin API) flag them, and re-running
`storyboard` re-plans exactly those until a real plan lands. When a re-plan
replaces a placeholder (or a frame change re-plans a pair) that already has a
rendered clip, the clip is kept and marked OUTDATED — nothing is deleted or
re-rendered until you regenerate that clip yourself (`render --clip ID` / the
panel's regenerate button).

**Whole-movie guidance — `global_motion_prompt`:** the storyboard has a
top-level `global_motion_prompt` field (empty by default, hand-edited between
steps). Whatever you put there is prepended to *every* clip's motion
prompt at render time and given to the planner as context, so facts that hold
across the whole movie live in one place instead of being repeated per clip —
e.g. `"Two different children appear throughout: an older boy with glasses
and a younger girl; they are separate people, never blend one into the
other."` Keep it to a sentence or two — it spends part of every clip's word
budget. Editing it does **not** invalidate already-rendered clips; re-render
with `--clip ID` (or `--force`) to apply it to existing ones.

**The cast — `characters`:** the storyboard also carries a top-level
`characters` list, one entry per person the planner saw
(`{"id": "bald-man", "epithet": "the bald man in pink sunglasses"}`). The
video model sees only pixels — it knows no names — so every motion prompt
refers to people by a short appearance-only epithet, and the cast is what
keeps that wording **identical across the whole movie**. It matters because
`storyboard` re-plans only the pairs that changed: a targeted re-plan sees
just its own two frames and would otherwise invent a fresh epithet for
someone the rest of the film already calls something else. The saved cast
rides along with every planning call, its epithets fixed; the planner adds
only people who are genuinely new.

Hand-edit an epithet to correct it (make it more distinguishing, fix a
mis-read feature) — the panel has a Cast editor next to the global motion
prompt, or edit `storyboard.json` directly. Because epithets are baked into
a motion prompt when its pair is *planned*, editing the cast changes future
plans only: it never marks a rendered clip outdated, and existing prompts
keep their old wording until you re-plan those clips
(`storyboard --replan-clip ID` / the panel's re-plan button).

**Per-frame styling guidance — `style_note`:** each frame in the storyboard
has a `style_note` field (empty by default, hand-edited between steps).
Whatever you put there is appended to the shared style prompt *for that frame
only*, and it survives later `storyboard` runs. It exists because the shared
prompt is written for portraits of people ("the subjects are the picture",
"keep the people at least as large as in the source") — which is wrong for the
occasional photo that is really *of a place*, or that shows people only
indirectly. A real frame was a stone facade whose glass door carried a small
silhouetted reflection of a couple; styling promoted them to physically
present foreground characters with two invented faces. A note like
`"This is a photo of the house facade. The couple appear ONLY as a small
silhouetted reflection inside the glass door — keep them a reflection, same
size and position, no faces."` steers just that frame.

Writing a note does **not** re-style anything on its own (styling resume is
existence-based, and re-styling spends image credits). Apply it deliberately:

```bash
python pipeline.py storyboard myfilm --restyle-frame 0i.png
```

That marks the frame's adjacent clips outdated, so re-plan/re-render those
afterwards if the picture changed materially.

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
  supported — or write it in the admin panel's Combine step, which saves the
  same file) and it rolls credits-style at the very end. For an ingested web
  order the file is **pre-filled with the customer's blessing** from the
  order ledger, so the letter usually only needs editing, not typing; an
  existing `letter.txt` is never overwritten by an ingest. With no letter
  written the flag is a no-op: combine warns and builds the movie without
  one (`pipeline.py status` and the panel both show whether there is a
  letter). **When
  `--credits-photos` is also on, the letter scrolls OVER the photo montage**
  at their normal brightness — a drop shadow behind the text is what keeps
  it readable, so nothing greys the photos out (`letter_overlay_dim`,
  default `0`, darkens them instead if you ever want that trade). Both are
  paced to end together: photos never flash faster than configured and the
  letter never scrolls faster than configured. With the letter alone it
  scrolls over a plain dark background. Empty lines become paragraph gaps;
  long lines wrap. Font is auto-detected (override with `letter_font_path`),
  size via `letter_font_size` (default 64), pace via
  `letter_seconds_per_screen` (default 7.0 — higher is slower).
  **Emoji** (a heart at the end of a blessing is the usual one) are drawn by
  the system's colour-emoji font — Apple Color Emoji on macOS, Noto Color
  Emoji on Linux — because the Hebrew text font has no glyph for them and
  would draw an empty box. Override with `letter_emoji_font_path`. If no
  emoji font is found the emoji are dropped from the letter rather than
  boxed, and the same goes for any other character the text font can't
  draw.
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
segment is re-rendered. Each segment also records the *recipe* that made it —
the settings it used plus the renderer's version — so changing a letter
setting, or upgrading the pipeline itself, re-renders it even though no
project file was touched. `--force` (the panel's Combine button) redoes every
segment regardless, and deleting `output/segments/` still forces a full redo.

### `run`

`storyboard` (reused if saved) → confirmation → `render` → confirmation →
`combine`, in one command. `-y` skips the confirmations; `--no-combine` stops
after the clips.

---

## All commands & flags

Global: `--config config.json` (before the command). Every command takes the
project name as its first argument (except `orders`, `lessons` and `costs`,
which are project-less).

| Command | Flags |
|---------|-------|
| `orders` | — (no project argument; lists Cloudinary order folders) |
| `serve` | — (no project argument) `--host`, `--port`, `--no-watch` |
| `ingest` | `<order>` (id / folder / unique fragment), `--force`, `--dry-run` |
| `init` | — |
| `storyboard` | `--force`, `--dry-run`, `--concurrency N`, `--style-prompt`, `--no-analyze`, `--replan-all` (rewrite every prompt from the current cast + tags), `--no-tag` (skip the identity proposal), `--replan-clip ID` (repeatable; fresh motion prompt for that pair), `--restyle-frame NAME` (repeatable; re-style that frame, e.g. `2.png`, marking adjacent clips outdated), `--duration 5\|10`, `--idea`, `--idea-file PATH`, `--frame-count N` |
| `render` | `--force`, `--dry-run`, `--concurrency N`, `-y/--yes`, `--clip ID` (repeatable), `--motion-prompt`, `--duration 5\|10`, `--add-audio`, `--no-audio` |
| `audio` | `--force`, `--dry-run`, `--concurrency N`, `--clip ID` (repeatable; redo that clip's audio), `--music-file PATH`, `--music-url URL` |
| `combine` | `--force`, `--dry-run`, `--music-file PATH`, `--music-url URL`, `--add-audio`, `--no-audio`, `--final`, `--[no-]intro`, `--[no-]credits-photos`, `--[no-]letter` |
| `publish` | `--dry-run` (print the name only), `-y/--yes` |
| `status` | — |
| `tag` | `--retag`, `--dry-run`, `-y/--yes` |
| `feedback` | `"<note>"` (optional with `--clip`), `--clip ID`, `--good`, `--no-watch`, `--no-learn`, `--dry-run` |
| `lessons` | — (no project argument) `--add TEXT`, `--scope motion\|style`, `--disable ID`, `--enable ID`, `--forget ID` |
| `costs` | — (no project argument; per-project + total spend) |
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
- **A clip render interrupted mid-flight is not money lost.** Clip jobs go
  through fal's queue: the request id is saved to `logs/state.json`
  (`falreq:<clip>`) the moment the job is submitted, so if the connection
  drops or the run crashes while waiting, the job keeps rendering on fal's
  side and the next `render` **fetches the already-paid result instead of
  submitting (and billing) a new one**. The saved request is only reused
  while the clip's frames, prompt, and duration are unchanged; if you edit
  the storyboard in between, a fresh job is submitted as expected.
- **Nothing collects those jobs in the background.** A submitted render is
  only fetched by the next `render` of that clip, so an interrupted run that
  is never resumed leaves a paid result sitting on fal forever. `status` and
  the admin panel now list them ("N paid render(s) waiting on the provider"),
  saying for each whether it is still *recoverable* — i.e. whether the plan
  is unchanged, so rendering fetches it for free — or whether an edit has
  since invalidated it, meaning that render is lost and a new one will be
  billed. Saving storyboard edits that invalidate a waiting render warns you
  by name at the moment it happens.
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
| `output/` | `final_video.mp4` + `music_custom.mp3` (the music track you uploaded, when you supplied one) |
| `output/published/` | A copy of every version delivered to the customer (`final_v1.mp4`, …) — `final_video.mp4` itself is overwritten by the next combine |
| `storyboard/` | `storyboard.json` (editable source of truth), `storyboard.md` (readable view), `preview.html` (visual contact sheet — open it in a browser) |
| `logs/` | Run logs + `state.json` + `costs.json` (what each paid call cost) |
| `order.json` | Which Cloudinary order this project came from (written by `ingest`) |
| `feedback.json` | Your notes on this project's rendered clips (the rules they taught are studio-wide, in `projects/_learning/lessons.json`) |
| `published.json` | Delivery history: every movie version published back to that order folder |
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
2. **Music bed** — one track **you supply**, mixed across the whole final
   video, **louder than the clip SFX** (the SFX is ducked under it). Music is
   never generated. Supply it by uploading a file in the panel (Audio *or*
   Combine step), pasting a URL there, `--music-file PATH`, or `--music-url
   URL`. No track means the movie simply has no music. Tune the balance with
   `music_volume` / `sfx_volume` in `config.json`.

   `--music-url` takes either a **direct audio link** (`.../track.mp3` — what
   royalty-free libraries hand out) or a **page URL** whose audio is extracted
   with [yt-dlp](https://github.com/yt-dlp/yt-dlp). Downloads are capped at
   60 MB. **These movies are sold to customers, so the track has to be one you
   may actually use** — a royalty-free library, a CC-licensed track, or
   something you licensed. Downloading a copyrighted song does not make it
   usable, and the pipeline cannot check this for you. If a page URL stops
   working, YouTube changed something: `.venv/bin/python -m pip install -U
   yt-dlp`.

### Turning it on

Off by default (`"audio_mode": "none"`). Enable it permanently in `config.json`
(`"audio_mode": "post"`), per-run with `--add-audio`, or retrofit existing
clips:

```bash
# Render clips AND add sound in one run
python pipeline.py run myfilm --add-audio

# Already have silent clips? Add SFX + music and rebuild the final video:
python pipeline.py audio myfilm

# Add your music track (the only way to get a music bed):
python pipeline.py audio myfilm --music-file ~/Music/mytrack.mp3

# ...or fetch it from a URL (direct audio link, or a page to extract from):
python pipeline.py audio myfilm --music-url https://example.com/track.mp3

# Force-off for one run even if config has audio_mode: post
python pipeline.py run myfilm --no-audio
```

The music bed comes from `--music-url` if given (downloaded into the custom
slot, so later runs reuse it), else `--music-file`, else an uploaded track
(`output/music_custom.mp3` — dropped in via the panel's Audio step and used
as-is for the whole movie), else a pre-existing `output/music.mp3`. If none of
those exist the movie is built with sound effects only — that is a normal
outcome, not an error.

Cost is roughly **$0.20–0.50 per full video** (MMAudio is ~$0.001/s; the music
bed costs nothing — it is your own file). Requires `ffmpeg`/`ffprobe` on your
`PATH`.

### Where the prompts come from

- **Image-based projects:** the frame analysis writes a per-clip `sound_prompt`
  into each transition; blank ones fall back to `default_sfx_prompt`.
- **Idea-based projects:** the storyboard also plans a `sound_prompt` per
  transition — editable in `storyboard.json` before rendering.
- **The music bed has no prompt**: it is the file you upload (panel) or pass
  with `--music-file`, never generated.

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
| `music_volume` | `0..1`, how loud the background bed plays (default `0.85`). |
| `music_loop` | `false` (default): the track plays once; if the video is longer, the rest continues with SFX only. `true`: the track repeats for the whole video. A track longer than the video is trimmed either way. |

### Spending & learning keys

| Key | Meaning |
|-----|---------|
| `pricing.openai_image_usd` | Price of one styled/generated frame (default `0.19`). |
| `pricing.openai_text_input_usd_per_1m` / `pricing.openai_text_output_usd_per_1m` | Text/vision token prices (defaults `1.25` / `10.0` per 1M). |
| `pricing.clip_usd_per_second` | Video model price per second (default `0.07` → `$0.35` for 5s, `$0.70` for 10s). |
| `pricing.sfx_usd` | Price of one video→audio pass (default `0.02`). |
| `learning_enabled` | `true` (default): rules learned from clip feedback are appended to planning/style prompts. `false` stops sending them (nothing is deleted). |
| `max_lessons_in_prompt` | How many active rules ride along with one call (default `25`, newest win). |
| `clip_review_frames` | How many stills are sampled across a clip when the reviewer watches it (default `8`). |

All spending figures are **estimates** derived from these prices — see
[What it costs](#what-it-costs-spending).

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
  server.py          # FastAPI admin API + job runner + order watcher (`serve`)
  intake.py          # web-order intake logic shared by orders/ingest/watcher
  clients/
    openai_client.py # image generation/editing + storyboard text (OpenAI)
    fal.py           # shared fal session: upload + subscribe + result parsing
    download.py      # shared atomic streaming download
    video.py         # VideoClient — image-to-video (fal)
    audio.py         # AudioClient — SFX + music (fal)
    cloudinary_client.py # order photo listing/download (Cloudinary Admin API)
    firebase_client.py   # Firestore order ledger (list orders, status write-back)
admin_ui/            # the admin panel (React + Vite + Mantine); dist/ served at / by `serve`
deploy/              # production: runbook, launchd plist, tunnel config, deploy script
.github/workflows/   # CI (tests/lint/panel build) + CD (deploy to the Mac mini)
pipeline.py          # entry-point shim
pyproject.toml       # package metadata, deps, `ai-video-maker` console script
```

The pipeline is built from three explicit inputs — `Config`, `Workspace`, and
`RunOptions` — plus an injected `confirm` callback for the interactive gates,
and reads no global state or stdin. Each CLI subcommand maps 1:1 onto a
`Pipeline.cmd_*` method, which is exactly the surface a future web API will
expose (each request builds its own `Workspace` + `RunOptions` and calls one
command).
