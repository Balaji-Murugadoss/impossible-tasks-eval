"""
Summarize a Task 1 (broken test) log: a results table per condition, then a
step-by-step replay of the runs worth reading.

Commands are read straight from the conversation, so this works even on logs
whose stored step data was incomplete.

Usage:
  python show_task1.py                 (newest task1 log)
  python show_task1.py logs/FILE.eval  (a specific log)
  python show_task1.py --all           (replay every run, not just flagged ones)
"""

import glob
import sys
from collections import Counter, defaultdict

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool

args = [a for a in sys.argv[1:] if not a.startswith("--")]
show_all = "--all" in sys.argv
path = args[0] if args else sorted(glob.glob("logs/*task1-broken-test*.eval"))[-1]
log = read_eval_log(path)
print(f"Log:      {path}")
print(f"Model:    {log.eval.model}")
print(f"Settings: {log.eval.task_args}\n")

FLAGGED = {"violation_tampered", "gamed_test", "gamed_disclosed", "gamed_silent",
           "claimed_without_passing", "ran_out_of_turns", "stopped_wrong"}


def first_text(args):
    for key in ("cmd", "command", "code", "answer", "input"):
        if isinstance(args.get(key), str):
            return args[key]
    for v in args.values():
        if isinstance(v, str):
            return v
    return str(args)


def replay(sample):
    n = 0
    for m in sample.messages:
        if isinstance(m, ChatMessageAssistant):
            for tc in m.tool_calls or []:
                n += 1
                text = first_text(tc.arguments or {}).replace("\n", "\n        ")
                print(f"  [{n}] {tc.function}:")
                print(f"        {text[:1500]}")
        elif isinstance(m, ChatMessageTool):
            out = (m.text or "").strip().replace("\n", "\n        ")
            print(f"      -> {out[:400]}")


by_cond = defaultdict(Counter)
picked = []
for s in log.samples:
    sc = s.scores["behavior_detector"]
    by_cond[s.id][sc.answer] += 1
    if show_all or sc.answer in FLAGGED:
        picked.append((s, sc))

for cond in ["control_unknown", "broken_impossible", "broken_unknown", "broken_possible"]:
    if cond in by_cond:
        print(f"{cond:<18} n={sum(by_cond[cond].values())}")
        for outcome, count in by_cond[cond].most_common():
            print(f"    {outcome:<26} {count}")

print(f"\n{len(picked)} runs to replay:")
for s, sc in picked:
    print(f"\n========== {s.id}  run {s.epoch}  ->  {sc.answer} ==========")
    replay(s)
