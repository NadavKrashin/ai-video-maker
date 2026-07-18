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

```bash
cloudflared tunnel login                       # opens the browser, pick your domain
cloudflared tunnel create animoments-admin
cloudflared tunnel route dns animoments-admin admin.<your-domain>
cp deploy/cloudflared-config.example.yml ~/.cloudflared/config.yml  # edit UUID + hostname
cloudflared tunnel run animoments-admin        # test once in the foreground
sudo cloudflared service install               # then install as a service
```

Then in the Cloudflare dashboard → **Zero Trust → Access → Applications**:
add `admin.<your-domain>`, policy **Allow → emails →** your email, login
method **One-Time PIN**. Session length 24h is a good default.

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
| Server logs | `tail -f ~/Library/Logs/animoments/serve.log` |
| Rotate the token | edit `.env`, restart the server, re-enter it in the panel |
| Roll back | `git revert` the bad commit on `main`, push (deploys the revert) |

## What is deliberately NOT automated

`projects/` holds the customers' real movies and is not in git. Back it up on
its own schedule (Time Machine covers it if enabled). Rendering/styling always
stays behind explicit human action or the opt-in watcher — deploys never touch
projects, never re-render, never spend credits.
