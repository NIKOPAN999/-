#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
P=/root/autodl-tmp/mosei-q1/mosei_q1
PY=/root/autodl-tmp/envs/q1/bin/python
# Wait for the registered extraction process; never stop or modify other tasks.
for attempt in $(seq 1 240); do
    if ! kill -0 22911 2>/dev/null; then break; fi
    sleep 30
done
if kill -0 22911 2>/dev/null; then echo 'Extraction still running after 2h; finalization deferred'; exit 2; fi
$PY "$P/scripts/team_report.py"
$PY "$P/scripts/audit_features.py"
$PY "$P/scripts/package_research_outputs.py"
