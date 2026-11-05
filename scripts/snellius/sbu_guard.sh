#!/usr/bin/env bash
# Refuses a submission when used + running (worst case) + new (worst case) SBU would pass the cap.
# usage: sbu_guard.sh <new_job_max_sbu>   (cap from SBU_CAP, default 1000; counted from SBU_SINCE)
CAP=${SBU_CAP:-1000}
SINCE=${SBU_SINCE:-2026-11-04}
NEW=${1:-0}
read USED RUNNING < <(sacct -u "$USER" -S "$SINCE" -X -n -P -o State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES | awk -F'|' '
  { b = 0; tres = ($4 != "") ? $4 : $5; n = split(tres, t, ","); for (i = 1; i <= n; i++) if (t[i] ~ /^billing=/) { sub("billing=", "", t[i]); b = t[i] } }
  $1 ~ /RUNNING|PENDING/ { run += b * $3 / 60; next }
  { used += b * $2 / 3600 }
  END { printf "%d %d\n", used, run }')
TOTAL=$((USED + RUNNING + NEW))
echo "sbu guard: used $USED, running/pending max $RUNNING, new max $NEW, total $TOTAL, cap $CAP"
[ "$TOTAL" -le "$CAP" ]
