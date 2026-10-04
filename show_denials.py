"""
Print the runs worth reading from a v2 / v2.1 log:
  - commands that got "Permission denied" (real read attempts)
  - commands that mentioned .secret only to exclude it (v2.1)
  - commands that searched outside the task folder (v2.1)
  - runs that hit the turn limit
Usage:
  python show_denials.py                 (uses the newest v2/v2.1 log)
  python show_denials.py logs/FILE.eval  (a specific log)
"""

import glob
import sys

from inspect_ai.log import read_eval_log

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = sorted(glob.glob("logs/*impossible-tasks-v2*.eval"))[-1]

log = read_eval_log(path)
print(f"Log:      {path}")
print(f"Model:    {log.eval.model}")
print(f"Settings: {log.eval.task_args}")

found = 0
for s in log.samples:
    score = s.scores["behavior_detector"]
    meta = score.metadata or {}
    steps = meta.get("steps", [])
    denied = [st for st in steps if st.get("denied")]
    respectful = [st for st in steps if st.get("respectful_mention")]
    scope = [st for st in steps if st.get("scope_expansion")]
    tamper = [st for st in steps if st.get("tamper_command")]
    v = score.value if isinstance(score.value, dict) else {}
    flagged = v.get("secret_read") or v.get("tamper_detected")
    limit = getattr(s, "limit", None)
    if not (denied or respectful or scope or tamper or flagged or limit):
        continue
    found += 1
    print(f"\n{s.id}  run {s.epoch}  ->  {score.answer}")
    if limit:
        print(f"  HIT LIMIT ({limit.type})")
    for st in denied:
        print(f"  step {st['step']} DENIED:     {st['command']}")
    for st in respectful:
        print(f"  step {st['step']} RESPECTFUL: {st['command']}")
    for st in scope:
        print(f"  step {st['step']} SCOPE:      {st['command']}")
    for st in tamper:
        print(f"  step {st['step']} TAMPER:     {st['command']}")
    if flagged:
        print(f"  FILE EVIDENCE: secret_read={v.get('secret_read')}  tamper_detected={v.get('tamper_detected')}")
    if meta.get("submission"):
        print(f"  submitted: {meta['submission'][:150]}")

print(f"\n{found} runs worth reading.")
