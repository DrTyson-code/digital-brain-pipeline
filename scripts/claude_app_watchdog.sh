#!/bin/bash
# Watchdog for the Claude.app log-update wedge seen on 2026-04-25.
# Run every 15 minutes via launchd or cron to detect a stalled Claude.app.
set -u

log_dir="$HOME/Library/Logs/Claude"
threshold=3600
now=$(date +%s)
newest_file=""
newest_mtime=0

while IFS= read -r file; do
  mtime=$(stat -f '%m' "$file" 2>/dev/null) || continue
  if [ "$mtime" -gt "$newest_mtime" ]; then
    newest_mtime=$mtime
    newest_file=$file
  fi
done < <(find "$log_dir" -type f -name '*.log' 2>/dev/null)

if [ -z "$newest_file" ]; then
  age=$((threshold + 1))
  echo "WEDGE: claude.app main.log not updated in $age seconds" >&2
  exit 2
fi

age=$((now - newest_mtime))
if [ "$age" -gt "$threshold" ]; then
  echo "WEDGE: claude.app main.log not updated in $age seconds" >&2
  exit 2
fi
echo "OK: claude.app main.log updated $age seconds ago"
