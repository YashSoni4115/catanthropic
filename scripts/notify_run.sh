#!/usr/bin/env bash
# Run a long job detached (survives the Copilot terminal closing), keep the Mac
# awake, log output, and push a phone notification via a Telegram bot on start
# and finish.
#
# One-time setup (on your phone / via BotFather):
#   1. In Telegram, message @BotFather -> /newbot -> follow prompts -> copy the
#      bot token it gives you (looks like 123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ).
#   2. Message your new bot anything (e.g. "hi") so it has an open chat.
#   3. Run:
#        curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" | python3 -m json.tool
#      and find "chat":{"id": ...} in the output. That's your chat id.
#   4. Put both in your shell profile (~/.zshrc) so every terminal has them:
#        export TG_BOT_TOKEN="<token>"
#        export TG_CHAT_ID="<chat id>"
#      or export them just for this session before running this script.
#
# Usage:
#   scripts/notify_run.sh <run-name> -- <command...>
#
# Example:
#   scripts/notify_run.sh B3_shape -- python -m catan_rl.rl.train --league B3 --workers 8
#
# This launches the command with `caffeinate -dims nohup`, logs to
# runs/<run-name>/stdout.log, and returns immediately. A detached watcher process
# sends Telegram notifications for start and completion (with exit code + log tail).

set -euo pipefail

if [[ -z "${TG_BOT_TOKEN:-}" || -z "${TG_CHAT_ID:-}" ]]; then
  echo "Missing TG_BOT_TOKEN and/or TG_CHAT_ID env vars. See the setup comment at the top of this script." >&2
  exit 1
fi

if [[ "${2:-}" != "--" ]]; then
  echo "Usage: $0 <run-name> -- <command...>" >&2
  exit 1
fi

NAME="$1"
shift 2
CMD=("$@")

RUN_DIR="runs/${NAME}"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/stdout.log"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

notify() {
  # $1 = message text (Telegram pushes natively, no priority levels needed)
  curl -s \
    --data-urlencode "chat_id=${TG_CHAT_ID}" \
    --data-urlencode "text=$1" \
    "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" >/dev/null || true
}

notify "🚀 Catanthropic: ${NAME} started
$(printf '%s' "${CMD[*]}")
@ ${TS}"

(
  caffeinate -dims nohup "${CMD[@]}" >"$LOG" 2>&1
  CODE=$?
  TAIL="$(tail -n 15 "$LOG" 2>/dev/null || true)"
  if [[ $CODE -eq 0 ]]; then
    notify "✅ Catanthropic: ${NAME} finished OK
exit=0
${TAIL}"
  else
    notify "🚨 Catanthropic: ${NAME} FAILED (exit=$CODE)
${TAIL}"
  fi
) &
disown

echo "Launched '${NAME}' (PID $!) -> ${LOG}"
echo "Notifications -> Telegram chat ${TG_CHAT_ID}"

