# Production on the Mac mini

How the pipeline runs in production, and the one-time setup steps. The moving
parts:

```
customer → animoments frontend → Cloudinary (photos) + Firestore (order doc)
                                            │
internet → Cloudflare Tunnel (TLS + Access) → 127.0.0.1:8300 pipeline.py serve
                                            │            (launchd, KeepAlive)
GitHub: dev → PR → main ──push──> self-hosted runner ──> deploy/deploy.sh
```

Design choices, deliberately:

- **The server only ever binds `127.0.0.1`.** No port forwarding, no open
  router ports, no `0.0.0.0`. The only way in from the internet is the
  Cloudflare tunnel, which terminates TLS at Cloudflare's edge and makes an
  *outbound* connection from the mini.
- **Two auth layers.** Cloudflare Access (your email + one-time PIN) in front
  of the hostname, then the app's own `ADMIN_API_TOKEN`. A leaked token alone
  gets nobody in; a Cloudflare session alone doesn't either.
- **Deploys only ever fast-forward `main`** and refuse a dirty tree, rerun
  the (offline) test suite on the mini itself, and health-check before
  declaring success.

## One-time setup

### 1. Secrets hygiene

```bash
# a strong admin token (the server refuses tokens under 16 chars):
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # → .env ADMIN_API_TOKEN

chmod 600 .env firebase-service-account.json
```

`.env` and `firebase-service-account.json` are gitignored — keep them that
way; they never travel through GitHub. Back both up somewhere private
(password manager attachment works well).

### 2. Cloudflare Tunnel (`cloudflared` is already installed via homebrew)

The panel lives at **`studio.animoment.co.il`** behind a named tunnel
(**`animoments-studio`**). The mini makes an *outbound* connection to
Cloudflare's edge; no inbound port is ever opened. (`animoment.co.il` itself is
a Cloudflare zone now — its DNS was migrated off SitesDepot; the apex/`www`
still point at Vercel as `DNS only`/grey-cloud records, only `studio` is
proxied.)

```bash
cloudflared tunnel login                  # opens the browser, pick your domain
cloudflared tunnel create animoments-studio
cloudflared tunnel route dns animoments-studio studio.animoment.co.il
cp deploy/cloudflared-config.example.yml ~/.cloudflared/config.yml  # edit UUID + hostname
cloudflared tunnel run animoments-studio       # test once in the foreground (Ctrl+C when done)
```

**Then install it as a daemon — but do NOT use `sudo cloudflared service
install`.** On macOS that runs as root, whose home is `/var/root`, so it never
finds `~/.cloudflared/config.yml` and writes a broken plist whose
`ProgramArguments` is just `cloudflared` (no `tunnel run`). The daemon starts,
does nothing, and the site returns **HTTP 530** ("tunnel not connected").
Instead, stage the config where a root daemon can see it (`/etc/cloudflared`)
and install the known-good plist committed at
`deploy/com.cloudflare.cloudflared.plist`:

```bash
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/config.yml /etc/cloudflared/config.yml
sudo cp ~/.cloudflared/<TUNNEL-UUID>.json /etc/cloudflared/
sudo sed -i '' 's#/Users/atlas/\.cloudflared#/etc/cloudflared#' /etc/cloudflared/config.yml
sudo cp deploy/com.cloudflare.cloudflared.plist /Library/LaunchDaemons/
sudo chown root:wheel /Library/LaunchDaemons/com.cloudflare.cloudflared.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.cloudflare.cloudflared.plist
cloudflared tunnel info animoments-studio      # should list active connections
curl -s -o /dev/null -w "%{http_code}\n" https://studio.animoment.co.il  # 200
```

Then in the Cloudflare dashboard → **Zero Trust → Access → Applications**:
add `studio.animoment.co.il`, policy **Allow → emails →** your email, login
method **One-Time PIN**. Session length 24h is a good default. Scope the app to
exactly that hostname — never the bare domain or a `*.animoment.co.il`
wildcard, or you'd gate the customer frontend too.

### 3. The server as a launchd service

```bash
mkdir -p ~/Library/Logs/animoments
cp deploy/com.animoments.pipeline.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.animoments.pipeline.plist
curl -s http://127.0.0.1:8300/api/health       # {"ok":true}
```

It's a Launch**Agent** (runs as you, starts at login), so the mini must log
your user in at boot: **System Settings → Users & Groups → automatically log
in**. Also keep the machine awake:

```bash
sudo pmset -a sleep 0 displaysleep 10 autorestart 1   # autorestart = boot after power loss
```

### 3b. Two checkouts: development and production

Production runs from **`~/production/ai-video-maker`** (branch `main`), a
clone used by nothing else. The mini is *developed* in
`~/Documents/code/ai-video-maker`, and the two must stay separate — they were
the same directory until 2026-07-26, which caused three distinct failures in
one day:

- `deploy.sh` refuses a dirty tree, so one uncommitted file in the dev tree
  broke every deploy — including deploys triggered from another machine.
- A deploy force-checks-out `main`, switching branches under whoever was
  working on the mini.
- `admin_ui/dist/` is served from disk on every request, so rebuilding the
  panel while developing changed what customers saw *instantly*, while the
  Python kept running the old code until a restart. That mismatch produced a
  batch-render button the backend did not understand, a false "paid renders
  waiting" alarm, and a 405 on a brand-new endpoint.

The production checkout needs the gitignored files that never travel through
git — copy them once when setting it up:

```bash
cp -p .env firebase-service-account.json intro.mp4 ~/production/ai-video-maker/
```

`projects/` (the real customer movies) lives in the **production** checkout;
the dev tree has a symlink to it, so both see the same data and there is only
ever one copy. Each checkout has its own `.venv`.

### 4. GitHub: branches, protection, runner

Branch flow: day-to-day work happens on **`dev`** (or feature branches merged
into it); a PR from `dev` → **`main`** is the release; pushing `main` deploys.

Branch protection (repo → Settings → Branches → Add rule for `main`):
- Require a pull request before merging
- Require status checks: `Tests + lint (Python 3.11)`, `Tests + lint
  (Python 3.12)`, `Admin panel build`
- (For a solo repo, skip required reviews — the checks are the gate.)

Self-hosted runner (repo → Settings → Actions → Runners → New self-hosted
runner → macOS/arm64, follow the shown commands, then):

```bash
./config.sh --url https://github.com/NadavKrashin/ai-video-maker \
            --token <shown-token> --labels mac-mini --unattended
./svc.sh install && ./svc.sh start              # runner as a service too
```

**Installed 2026-07-26, re-registered 2026-11-11.** It lives in
`~/actions-runner`, registered as `Atlass-Mac-mini` with labels
`self-hosted, macOS, ARM64, mac-mini` (the workflow's `runs-on` needs the
first, second and last — the runner's NAME is irrelevant to routing). It runs
as a **user LaunchAgent**,
`actions.runner.NadavKrashin-ai-video-maker.Atlass-Mac-mini` — that is
deliberate and must stay that way: `deploy/deploy.sh` calls
`launchctl kickstart gui/$(id -u)/com.animoments.pipeline`, which only works
from inside the logged-in user session. A system-level daemon cannot restart
the server.

```bash
cd ~/actions-runner && ./svc.sh status    # or stop / start
gh api repos/NadavKrashin/ai-video-maker/actions/runners \
  -q '.runners[] | "\(.name) \(.status)"'
```

**If a deploy never seems to happen, check the runner first.** With no runner
online a push to `main` does not fail — the job sits *queued* indefinitely and
the release looks like it simply did nothing. That is exactly what happened on
2026-07-26: `deploy.yml` had been in the repo for over a week with no runner
ever registered, so every "push main to deploy" was a no-op and the mini was
only ever updated by hand.

**The runner lives in `~/actions-runner` — never inside the app checkout**
(owner's call, 2026-11-11). It was once reinstalled at
`~/production/ai-video-maker/deploy/actions-runner`. When the owner removed
that folder, its LaunchAgent kept running, re-spawning a `Runner.Listener`
that no longer existed (`ENOENT`) every 5 seconds. GitHub showed the runner
offline and a release sat queued. Keep the runner outside both checkouts.
To recover from a deleted install:

```bash
launchctl bootout gui/$(id -u)/actions.runner.NadavKrashin-ai-video-maker.<name>
# then move that LaunchAgent's plist out of ~/Library/LaunchAgents
cd ~/actions-runner && ./config.sh remove --token <remove-token>   # if configured
./config.sh --unattended --url https://github.com/NadavKrashin/ai-video-maker \
            --token <registration-token> --name Atlass-Mac-mini \
            --labels mac-mini --replace
./svc.sh install && ./svc.sh start
```

Both tokens come from `gh api -X POST
repos/NadavKrashin/ai-video-maker/actions/runners/{remove,registration}-token
-q .token`. `--replace` takes over the existing registration of that name,
so no offline duplicate is left behind. A queued deploy then runs on its own.

Settings → Actions → General: set **"Allow select actions"** and disable
workflow runs for fork PRs (defaults are fine for a private repo, but check —
a self-hosted runner must never run untrusted PR code).

### 5. macOS hardening

- System Settings → General → Sharing: everything **off** (no Screen Sharing,
  Remote Login/SSH off unless you actively use it — prefer Tailscale-only SSH
  if you need remote shell access).
- Firewall **on** (the server is loopback-only; nothing needs an inbound rule).
- FileVault on, automatic macOS security updates on.

## Day-to-day

| What | How |
|---|---|
| Ship to production | merge PR `dev` → `main` (deploy runs itself) |
| Deploy manually | `bash deploy/deploy.sh` on the mini |
| Restart the server | `launchctl kickstart -k gui/$(id -u)/com.animoments.pipeline` |
| Restart the tunnel | `sudo launchctl kickstart -k system/com.cloudflare.cloudflared` |
| Server logs | `tail -f ~/Library/Logs/animoments/serve.log` |
| Tunnel logs | `tail -f /Library/Logs/com.cloudflare.cloudflared.err.log` |
| Rotate the token | edit `.env`, restart the server, re-enter it in the panel |
| Roll back | `git revert` the bad commit on `main`, push (deploys the revert) |

## Restarts and running jobs

A restart kills whatever job is running. Two things soften that:

- **`deploy.sh` refuses to restart a busy server.** It asks
  `/api/health` first (`"busy": true` while any job is queued or running)
  and exits before pulling or rebuilding anything. Wait for the job, or run
  `FORCE_DEPLOY=1 bash deploy/deploy.sh` when you have looked and mean it. A
  deploy triggered by a push to `main` fails the same way and can be re-run
  from the Actions tab once the job has finished.
- **The job list survives.** It is saved to `projects/_learning/jobs.json`
  (newest 200 jobs, last 300 log lines each). After a restart, a job that
  was queued or running shows as **interrupted**, with its log. Nothing is
  re-run automatically — re-running spends credits, and a restart is not a
  decision to spend them. Run the same action again from the panel: finished
  files are skipped, and a clip that was already submitted to the video
  provider is fetched, not bought twice.

## What is deliberately NOT automated

`projects/` holds the customers' real movies and is not in git. Rendering and
styling always stay behind explicit human action or the opt-in watcher —
deploys never touch projects, never re-render, never spend credits.

### Backups — `deploy/backup.sh` (schedule it; it is NOT on by default)

`projects/` is the ONLY irreplaceable data on this box — everything else is
in git or reinstallable, and its contents cost real API credits to produce.
On 2026-07-26, 606 MB of styled frames, rendered clips, storyboards and
finished movies across four projects were destroyed by a single
`git checkout`, and with no backup the loss was permanent.

`deploy/backup.sh` writes a dated, complete snapshot of `projects/` to
`$BACKUP_DEST/projects-YYYYmmdd-HHMMSS/`. Unchanged files are hard links to
the previous snapshot, so 30 snapshots cost roughly one copy plus what
changed. It never writes to `projects/`, refuses a destination that does not
exist (an unplugged disk must not become a "backup" onto the internal one),
refuses an empty source, and prunes only its own complete snapshots beyond
`BACKUP_KEEP` (default 30).

**The script existing is not a backup.** Until the schedule below is
installed with a real destination and its log shows a snapshot, there is
still none — `tmutil destinationinfo` reported "No destinations configured"
on both machines as of 2026-11-11.

```bash
# once, by hand, to see it work:
BACKUP_DEST=/Volumes/<disk>/animoments-backups bash deploy/backup.sh

# nightly at 04:00: edit BACKUP_DEST in the plist first, then
mkdir -p ~/Library/Logs/animoments
cp deploy/com.animoments.backup.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.animoments.backup.plist
tail ~/Library/Logs/animoments/backup.log
```

The destination must be a DIFFERENT disk (external drive, another machine's
mounted share). A folder on the same disk survives a bad `git checkout` but
not the disk. For off-site cover, point a cloud sync client at the backup
folder — not at `projects/` itself: a sync client once bumped every mtime in
there.

**Restoring.** Snapshots are plain folders. Copy the project back out —
never rsync a snapshot over the live `projects/`, and never while a job is
running:

```bash
ls /Volumes/<disk>/animoments-backups/            # pick a snapshot
cp -R /Volumes/<disk>/animoments-backups/projects-<stamp>/<project> \
      ~/production/ai-video-maker/projects/<project>
```

If the project folder still exists in a damaged state, move it aside first
(`mv <project> <project>.damaged`) rather than copying on top of it.
Pinned by tests/test_backup_script.py, which runs the real script.
