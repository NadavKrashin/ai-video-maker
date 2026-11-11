#!/usr/bin/env bash
# Snapshot backup of projects/ — the only irreplaceable data on the machine.
#
# Everything else here is in git or reinstallable; projects/ holds styled
# frames, rendered clips and delivered movies that cost real API credits and
# cannot be regenerated identically. It has been destroyed once already (606
# MB, 2026-07-26, by a single `git checkout`) with nothing to restore from.
#
#   BACKUP_DEST=/Volumes/Backup/animoments bash deploy/backup.sh
#
# Each run writes a complete, browsable copy to
#   $BACKUP_DEST/projects-YYYYmmdd-HHMMSS/
# Unchanged files are HARD LINKS to the previous snapshot, so thirty
# snapshots of a 1 GB folder cost about 1 GB plus whatever changed — and
# every snapshot is still a full tree you can copy a project straight out of.
#
# Safety rules, each one here because the alternative loses data quietly:
#   * BACKUP_DEST must ALREADY EXIST. An unmounted external disk leaves
#     /Volumes/Backup absent; creating it would "back up" onto the internal
#     disk — the same disk as the original — and report success.
#   * An EMPTY source is refused. It means projects/ is missing, unlinked or
#     already wiped, and a run of empty snapshots would push the last good
#     ones out through pruning.
#   * A snapshot is written as *.partial and renamed only after its file
#     count matches the source, so an interrupted run never looks complete.
#   * Pruning removes only directories this script named, only complete
#     ones, and only beyond BACKUP_KEEP. It never touches the source.
#   * Nothing here ever writes to the source. Restoring is a manual copy —
#     see deploy/PRODUCTION.md.
set -euo pipefail

SRC="${PROJECTS_DIR:-$HOME/production/ai-video-maker/projects}"
DEST="${BACKUP_DEST:-}"
KEEP="${BACKUP_KEEP:-30}"
STAMP="${BACKUP_STAMP:-$(date +%Y%m%d-%H%M%S)}"

fail() { echo "backup: $*" >&2; exit 1; }

[[ -n "$DEST" ]] || fail "BACKUP_DEST is not set — where should the copy go?"
[[ -d "$DEST" ]] || fail "destination $DEST does not exist (disk not mounted?) — refusing to create it."
[[ -d "$SRC" ]] || fail "source $SRC is not a directory."
[[ "$KEEP" =~ ^[0-9]+$ && "$KEEP" -ge 1 ]] || fail "BACKUP_KEEP must be a whole number >= 1 (got '$KEEP')."
command -v rsync >/dev/null || fail "rsync is not installed."

# Resolve symlinks: the dev tree's projects/ can be a link to production's.
SRC="$(cd "$SRC" && pwd -P)"
DEST="$(cd "$DEST" && pwd -P)"
case "$DEST/" in
  "$SRC/"*) fail "destination $DEST is inside the source $SRC." ;;
esac

count_files() { find "$1" -type f ! -name .DS_Store | wc -l | tr -d ' '; }

src_count="$(count_files "$SRC")"
[[ "$src_count" -gt 0 ]] || fail "source $SRC contains no files — refusing to snapshot an empty projects folder."

final="$DEST/projects-$STAMP"
partial="$final.partial"
[[ ! -e "$final" && ! -e "$partial" ]] || fail "snapshot $final already exists."

# Newest COMPLETE snapshot, to hard-link unchanged files against.
previous="$(find "$DEST" -maxdepth 1 -type d -name 'projects-[0-9]*' ! -name '*.partial' | sort | tail -n 1)"
link_args=()
[[ -n "$previous" ]] && link_args=(--link-dest="$previous")

# The ${x[@]+...} form is for macOS's bash 3.2, where expanding an EMPTY
# array under `set -u` is an "unbound variable" error.
sync_once() {
  rsync -a --exclude .DS_Store ${link_args[@]+"${link_args[@]}"} "$SRC/" "$partial/"
}

sync_once
dst_count="$(count_files "$partial")"
if [[ "$dst_count" -ne "$src_count" ]]; then
  # A job was probably writing while we copied. One catch-up pass, then
  # judge against a fresh count.
  sync_once
  src_count="$(count_files "$SRC")"
  dst_count="$(count_files "$partial")"
fi
[[ "$dst_count" -ge "$src_count" ]] || fail "snapshot has $dst_count files, source has $src_count — left as $partial for inspection."

mv "$partial" "$final"

# Prune: complete snapshots only, oldest first, never below KEEP.
pruned=0
while IFS= read -r old; do
  [[ -n "$old" ]] || continue
  rm -rf "$old"
  pruned=$((pruned + 1))
done < <(find "$DEST" -maxdepth 1 -type d -name 'projects-[0-9]*' ! -name '*.partial' | sort | awk -v keep="$KEEP" '{ a[NR] = $0 } END { for (i = 1; i <= NR - keep; i++) print a[i] }')

echo "backup: $final — $dst_count files$( [[ -n "$previous" ]] && echo ", linked against $(basename "$previous")" )$( [[ "$pruned" -gt 0 ]] && echo ", pruned $pruned old" )"
