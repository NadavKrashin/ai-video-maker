#!/usr/bin/env bash
# Production deploy for the Mac mini: fast-forward the live checkout to
# origin/main, reinstall, rebuild the panel, restart the launchd service,
# and verify health. Run by .github/workflows/deploy.yml (self-hosted
# runner) on every push to main — or by hand: bash deploy/deploy.sh
set -euo pipefail

REPO="${REPO_DIR:-$HOME/Documents/Code/Personal/ai-video-maker}"
SERVICE="com.animoments.pipeline"
HEALTH_URL="http://127.0.0.1:8300/api/health"

cd "$REPO"

# Never clobber local work: production deploys only ever fast-forward a
# clean tree. A dirty tree means someone is mid-edit on the machine.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "deploy: working tree at $REPO is dirty — commit/stash first." >&2
  exit 1
fi

echo "deploy: updating $REPO to origin/main"
git fetch origin main
git checkout -q main
git merge --ff-only origin/main

echo "deploy: installing python package"
.venv/bin/python -m pip install --quiet -e ".[dev]"

echo "deploy: running the test suite"
.venv/bin/python -m pytest tests/ -q

echo "deploy: building the admin panel"
(cd admin_ui && npm ci --silent && npm run build --silent)

echo "deploy: restarting $SERVICE"
launchctl kickstart -k "gui/$(id -u)/$SERVICE"

echo "deploy: waiting for health"
for i in {1..15}; do
  if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "deploy: healthy — $(git log --oneline -1)"
    exit 0
  fi
  sleep 2
done
echo "deploy: server did not come back healthy at $HEALTH_URL" >&2
exit 1
