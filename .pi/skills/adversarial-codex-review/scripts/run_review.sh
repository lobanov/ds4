#!/usr/bin/env bash
# Dispatch codex to run an adversarial review non-interactively.
#
# The prompt may be supplied as a file or via stdin. Model/effort/sandbox are
# flags, so order doesn't matter — easy to invoke from bash:
#
#   ./run_review.sh prompt.md                         # file, all defaults
#   ./run_review.sh -m gpt-5.5 -e xhigh prompt.md     # file + flags
#   cat prompt.md | ./run_review.sh                   # stdin, all defaults
#   echo "review X" | ./run_review.sh -e high -s read-only
#   ./run_review.sh -e high <<'EOF'                   # heredoc
#   review this work...
#   EOF
#   ./run_review.sh -                                 # explicit stdin marker
#
# Options:
#   -m, --model MODEL       codex model id   (default: gpt-5.5)
#   -e, --effort EFFORT     reasoning effort (default: xhigh)
#   -s, --sandbox MODE      read-only | workspace-write | danger-full-access
#                                              (default: read-only)
#   -o, --log PATH          write codex log here (default: mktemp)
#   -f, --final PATH        write codex's final message here
#                                              (default: <log>.final.md)
#   -h, --help              show this help
#
# Env:
#   CODEX_EXTRA_ARGS  extra flags appended to the codex invocation.
#
# Prints the codex log path AND final-message path to stdout; codex's exit code
# is reported on stderr.
# The caller (pi) must INDEPENDENTLY VERIFY any decisive claim before accepting.
set -euo pipefail

usage() {
  sed -n '2,/^set -euo pipefail$/p' "$0" | sed 's/^# \?//' | sed 's/^#//'
}

MODEL="gpt-5.5"
EFFORT="xhigh"
SANDBOX="read-only"
LOG_PATH=""
FINAL_PATH=""
PROMPT_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    -m|--model)    MODEL="${2:?--model needs a value}"; shift 2 ;;
    --model=*)     MODEL="${1#*=}"; shift ;;
    -e|--effort)   EFFORT="${2:?--effort needs a value}"; shift 2 ;;
    --effort=*)    EFFORT="${1#*=}"; shift ;;
    -s|--sandbox)  SANDBOX="${2:?--sandbox needs a value}"; shift 2 ;;
    --sandbox=*)   SANDBOX="${1#*=}"; shift ;;
    -o|--log)      LOG_PATH="${2:?--log needs a value}"; shift 2 ;;
    --log=*)       LOG_PATH="${1#*=}"; shift ;;
    -f|--final)    FINAL_PATH="${2:?--final needs a value}"; shift 2 ;;
    --final=*)     FINAL_PATH="${1#*=}"; shift ;;
    -h|--help)     usage; exit 0 ;;
    --)            shift; [ $# -gt 0 ] && PROMPT_FILE="${PROMPT_FILE:-$1}"; break ;;
    -*)            echo "error: unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)             if [ -z "$PROMPT_FILE" ]; then PROMPT_FILE="$1"; else
                     echo "error: unexpected argument: $1" >&2; exit 1; fi; shift ;;
  esac
done

# Resolve the prompt: positional file, "-", or stdin.
STDIN_PROMPT=""
if [ -z "$PROMPT_FILE" ] || [ "$PROMPT_FILE" = "-" ]; then
  if [ -t 0 ]; then
    echo "error: no prompt provided. Pass a prompt file, or pipe the prompt via stdin." >&2
    usage >&2
    exit 1
  fi
  STDIN_PROMPT="$(mktemp -t adv_codex_prompt_XXXXXX.md)"
  cat > "$STDIN_PROMPT"
  PROMPT_FILE="$STDIN_PROMPT"
fi
trap '[ -n "$STDIN_PROMPT" ] && rm -f "$STDIN_PROMPT"' EXIT

# Validate.
if ! command -v codex >/dev/null 2>&1; then
  echo "error: codex CLI not found on PATH (install codex-cli and authenticate)" >&2
  exit 127
fi
[ -r "$PROMPT_FILE" ] || { echo "error: prompt file not readable: $PROMPT_FILE" >&2; exit 1; }
case "$SANDBOX" in
  read-only)          CODEX_ARGS=(-s read-only) ;;
  workspace-write)    CODEX_ARGS=(-s workspace-write) ;;
  danger-full-access) CODEX_ARGS=(--dangerously-bypass-approvals-and-sandbox) ;;
  *) echo "error: unknown sandbox '$SANDBOX' (use read-only|workspace-write|danger-full-access)" >&2; exit 1 ;;
esac

[ -z "$LOG_PATH" ] && LOG_PATH="$(mktemp -t adversarial_codex_XXXXXX.log)"
[ -z "$FINAL_PATH" ] && FINAL_PATH="${LOG_PATH}.final.md"

{
  echo "=== adversarial codex review ==="
  echo "model=$MODEL effort=$EFFORT sandbox=$SANDBOX prompt=$PROMPT_FILE final=$FINAL_PATH"
  echo "cwd=$(pwd) started=$(date -u +%FT%TZ)"
  echo "=== codex output follows ==="
} >&2

# approval_policy=never keeps workspace-write non-interactive.
# shellcheck disable=SC2086
codex exec -m "$MODEL" \
  -c model_reasoning_effort="$EFFORT" \
  -c approval_policy="never" \
  --output-last-message "$FINAL_PATH" \
  "${CODEX_ARGS[@]}" \
  ${CODEX_EXTRA_ARGS:-} \
  - < "$PROMPT_FILE" > "$LOG_PATH" 2>&1 || true
CODEX_EXIT=$?

echo "=== codex finished: exit=$CODEX_EXIT ; log=$LOG_PATH ; final=$FINAL_PATH ===" >&2
echo "$LOG_PATH"
echo "$FINAL_PATH"
