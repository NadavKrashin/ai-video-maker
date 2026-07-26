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

**Installed 2026-07-26.** It lives in `~/actions-runner`, registered as
`mac-mini` with labels `self-hosted, macOS, ARM64, mac-mini` (the workflow's
`runs-on` needs the first, second and last). It runs as a **user LaunchAgent**,
`actions.runner.NadavKrashin-ai-video-maker.mac-mini` — that is deliberate and
must stay that way: `deploy/deploy.sh` calls
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

## What is deliberately NOT automated

`projects/` holds the customers' real movies and is not in git. Back it up on
its own schedule (Time Machine covers it if enabled). Rendering/styling always
stays behind explicit human action or the opt-in watcher — deploys never touch
projects, never re-render, never spend credits.
