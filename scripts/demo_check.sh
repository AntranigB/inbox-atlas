#!/usr/bin/env bash
# Demo preflight: hits every endpoint the DEMO.md walkthrough uses and prints PASS/FAIL per step.
#
#   scripts/demo_check.sh                          # http://127.0.0.1:8765, ATLAS_TOKEN from env or .env
#   scripts/demo_check.sh http://127.0.0.1:8790    # another instance
#   SIDECAR=http://127.0.0.1:8766 MCP=1 scripts/demo_check.sh
#
# Run it 10 minutes before judging. It also warms Grok's expansion cache for the demo queries, so
# the live searches come back fast and with the same facets this check saw.
# Exit code is the number of FAILs (WARN does not count).

set -u
BASE="${1:-${ATLAS_URL:-http://127.0.0.1:8765}}"
BASE="${BASE%/}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${ATLAS_TOKEN:-}" ] && [ -f "$ROOT/.env" ]; then
  ATLAS_TOKEN="$(grep -E '^ATLAS_TOKEN=' "$ROOT/.env" | tail -1 | cut -d= -f2- | tr -d '"'"'")"
fi
TOKEN="${ATLAS_TOKEN:-}"
# the checks below use Python 3.12 f-strings (same quotes nested), so prefer the project venv
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
SIDECAR="${SIDECAR:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAILS=0

g() { printf '\033[32mPASS\033[0m'; }
r() { printf '\033[31mFAIL\033[0m'; }
y() { printf '\033[33mWARN\033[0m'; }

# call METHOD PATH [JSON]: writes the body to $TMP/out, prints "<http code> <seconds>"
call() {
  local m="$1" p="$2" d="${3:-}"
  local args=(-s -o "$TMP/out" -w '%{http_code} %{time_total}' --max-time 90 -X "$m" -H "Authorization: Bearer $TOKEN")
  [ -n "$d" ] && args+=(-H 'content-type: application/json' -d "$d")
  curl "${args[@]}" "$BASE$p" 2>/dev/null || echo "000 0"
}

# check NAME PYTHON_EXPR_ON_d [DETAIL_EXPR]: d is the parsed JSON body; expr True means PASS
report() {
  local name="$1" meta="$2" ok="$3" detail="$4"
  local code="${meta%% *}" secs="${meta##* }"
  local status
  case "$ok" in
    pass) status="$(g)" ;;
    warn) status="$(y)" ;;
    *) status="$(r)"; FAILS=$((FAILS + 1)) ;;
  esac
  printf '%s  %-38s %5.1fs  %s\n' "$status" "$name" "$secs" "$detail"
  [ "$code" = "200" ] || [ "$ok" != "fail" ] || printf '      http %s from %s\n' "$code" "$BASE"
}

judge() {
  "$PY" - "$TMP/out" "$1" <<'PY'
import json, sys
path, expr = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception:
    print("fail\tno JSON back"); sys.exit()
env = {"d": d, "any": any, "all": all, "len": len, "str": str, "round": round}
try:
    ok, detail = eval(expr, env)
except Exception as e:  # noqa: BLE001
    ok, detail = False, f"{type(e).__name__}: {e}"
print(("pass" if ok is True else "warn" if ok == "warn" else "fail") + "\t" + str(detail)[:150])
PY
}

step() {  # step NAME METHOD PATH JSON EXPR
  local name="$1" meta out
  meta="$(call "$2" "$3" "${4:-}")"
  out="$(judge "$5")"
  report "$name" "$meta" "${out%%$'\t'*}" "${out#*$'\t'}"
}

echo "Inbox Atlas demo preflight against $BASE"
echo

# 0. server and token
step "health + token" GET /api/health "" \
  '("n_emails" in d and d["n_emails"] > 0, f"{d.get("n_emails")} items, encoder {d.get("encoder")}" if "n_emails" in d else "token rejected: export ATLAS_TOKEN")'
meta="$(curl -s -o "$TMP/page" -w '%{http_code} %{time_total}' --max-time 10 "$BASE/" || echo '000 0')"
if grep -q 'Inbox Atlas' "$TMP/page" 2>/dev/null; then report "web page served" "$meta" pass "$BASE/"
else report "web page served" "$meta" fail "no page at $BASE/"; fi
step "map" GET /api/map "" '(len(d.get("points") or []) > 0, f"{len(d.get("points") or [])} points, {len(d.get("clusters") or [])} clusters")'
step "encoders (Base/Tuned toggle)" GET /api/encoders "" \
  '(True if len(d.get("available") or []) >= 2 else "warn", ", ".join(d.get("available") or []) + ("" if len(d.get("available") or []) >= 2 else ": toggle hidden, build models/atlas-embed index"))'

# 1. search by meaning
step "1a keyword: coding competition" POST /api/search '{"query":"coding competition","mode":"keyword","k":5}' \
  '(len(d["hits"]) > 0, "top: " + (d["hits"][0].get("subject") or "") if d["hits"] else "no hits")'
step "1b region: coding competition" POST /api/search '{"query":"coding competition","mode":"region","k":12}' \
  '((d["region"]["related"] and not any(h.get("member") and "assessment" in (h.get("subject") or "").lower() for h in d["hits"])) or False, f"RELATED {d["region"]["related"]}, {d["region"]["count"]} in region; job test in region: {any(h.get("member") and "assessment" in (h.get("subject") or "").lower() for h in d["hits"])}")'
FIRST_ID="$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["hits"][0]["id"] if d.get("hits") else "")' "$TMP/out" 2>/dev/null)"
step "1c facet chips" POST /api/search '{"query":"coding competition","mode":"region","k":3}' \
  '(len(d["facets"]["positive"]) >= 3, f"{len(d["facets"]["positive"])} facets, {len(d["facets"]["negative"])} negatives: " + ", ".join(d["facets"]["negative"]))'
step "1d region: yacht maintenance = NO" POST /api/search '{"query":"yacht maintenance","mode":"region","k":3}' \
  '(d["region"]["related"] is False, f"RELATED {d["region"]["related"]}, max z {d["region"]["max_z"]}")'
step "1e tuned encoder region" POST /api/search '{"query":"coding competition","mode":"region","k":5,"encoder":"atlas-embed"}' \
  '(d["region"]["related"] or "warn", f"encoder {d.get("encoder")}, {d["region"]["count"]} in region")'
if [ -n "$FIRST_ID" ]; then
  step "1f email expand" GET "/api/email/$FIRST_ID" "" '(len(d.get("body") or "") > 0, (d.get("subject") or "")[:60])'
fi
step "1g token counter (agent pack)" POST /api/context '{"question":"coding competition"}' \
  '(d.get("answerable") is True and d.get("tokens", 0) > 0, f"{d.get("tokens")} tokens, answerable {d.get("answerable")}")'
for q in "money I owe someone" "travel plans"; do
  step "1h demo chip: $q" POST /api/search '{"query":"'"$q"'","mode":"region","k":5}' \
    '(d["region"]["related"] or "warn", f"RELATED {d["region"]["related"]}, {d["region"]["count"]} in region")'
done

# 2. text it (same agent the iMessage sidecar calls)
step "2a ask: what do I have tomorrow?" POST /api/ask '{"text":"what do I have tomorrow?","channel":"imessage"}' \
  '(bool(d.get("reply")), d.get("reply","").replace("\n"," "))'
step "2b ask: anyone ask me for money?" POST /api/ask '{"text":"did anyone ask me for money?","channel":"imessage"}' \
  '(("$" in d.get("reply","")) or "warn", d.get("reply","").replace("\n"," "))'
step "2c ask: coding competitions" POST /api/ask '{"text":"any emails about coding competitions?","channel":"imessage"}' \
  '(True if "assessment" not in d.get("reply","").lower() else "warn", d.get("reply","").replace("\n"," "))'
if [ -n "$SIDECAR" ]; then
  meta="$(curl -s -o "$TMP/out" -w '%{http_code} %{time_total}' --max-time 5 "$SIDECAR/health" || echo '000 0')"
  out="$(judge '(d.get("ok") is True, f"{d.get("mode")} mode, connected {d.get("connected")}")')"
  report "2d iMessage sidecar" "$meta" "${out%%$'\t'*}" "${out#*$'\t'}"
fi

# 3. voice
if command -v say >/dev/null && command -v afconvert >/dev/null; then
  say -o "$TMP/clip.aiff" "find me emails about, um, no wait, the coding competition" 2>/dev/null
  afconvert -f WAVE -d LEI16@16000 "$TMP/clip.aiff" "$TMP/clip.wav" 2>/dev/null
  meta="$(curl -s -o "$TMP/out" -w '%{http_code} %{time_total}' --max-time 30 -H "Authorization: Bearer $TOKEN" \
    -F audio=@"$TMP/clip.wav" -F style=query "$BASE/api/dictate" || echo '000 0')"
  out="$(judge '(bool(d.get("text")) and "wait" not in d.get("text","").lower(), repr(d.get("text")))')"
  report "3 dictate (say + afconvert)" "$meta" "${out%%$'\t'*}" "${out#*$'\t'}"
else
  printf '%s  %-38s        %s\n' "$(y)" "3 dictate" "skipped (needs macOS say + afconvert)"
fi

# 4. agent context
step "4a folders: arm servo pulse widths" POST /api/context/folders '{"question":"what pulse widths did I use for the arm servos?"}' \
  '(d.get("related") is True, f"{d.get("tokens")} tokens, top folder {(d.get("folders") or [{}])[0].get("path")}")'
step "4b folders: yacht charter = stop" POST /api/context/folders '{"question":"did I book a yacht charter?"}' \
  '(d.get("related") is False and d.get("tokens", 999) < 60, f"{d.get("tokens")} tokens, related {d.get("related")}")'
if [ "${MCP:-0}" = "1" ]; then
  if (cd "$ROOT" && uv run python scripts/mcp_check.py >"$TMP/mcp" 2>&1); then
    printf '%s  %-38s        %s\n' "$(g)" "4c MCP stdio server" "$(grep -c 'tokens' "$TMP/mcp") tool calls answered"
  else
    printf '%s  %-38s        %s\n' "$(r)" "4c MCP stdio server" "see: uv run python scripts/mcp_check.py"
    FAILS=$((FAILS + 1))
  fi
fi

echo
if [ "$FAILS" = 0 ]; then echo "all demo steps pass"; else echo "$FAILS step(s) failed"; fi
exit "$FAILS"
