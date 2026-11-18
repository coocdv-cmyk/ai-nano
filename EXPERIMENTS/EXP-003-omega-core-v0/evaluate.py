"""EXP-003 evaluation: quality (val bits/byte vs baselines), native-engine speed, samples.

python EXPERIMENTS/EXP-003-omega-core-v0/evaluate.py
Run with no other heavy processes: speed numbers are sensitive to background load.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numba  # noqa: E402

from omega_nano.core.engine import Engine  # noqa: E402
from omega_nano.hw import detect  # noqa: E402

HERE = Path(__file__).parent
PROMPTS = [b"def ", b"import os\n\nclass ", b"    for i in range(", b"# Return the "]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = {"experiment": "EXP-003-omega-core-v0", "date": time.strftime("%Y-%m-%d %H:%M"), "hardware": detect(),
           "arms": {}}
    for arm in ("fp32", "ternary"):
        log = json.loads((HERE / "runs" / arm / "log.json").read_text(encoding="utf-8"))
        eng = Engine.load(HERE / "runs" / arm / "model.pt")
        speed = {}
        for threads in (1, numba.config.NUMBA_NUM_THREADS):
            numba.set_num_threads(threads)
            speed[f"threads_{threads}"] = [eng.bench(n_new=2000, batch=b) for b in (threads * 2,)][0]
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
        samples = {p.decode(): eng.generate(p, 240, temperature=0.6, seed=7).decode("utf-8", "replace")
                   for p in PROMPTS}
        out["arms"][arm] = {"final": log["final"], "curve": log["curve"], "params": log["params"],
                            "speed": speed, "samples": samples}
        out["baselines_bpb"] = log["baselines_bpb"]
        print(f"\n=== {arm}: val {log['final']['val_bpb']} bpb, {log['final']['steps']} steps, "
              f"{log['final']['train_tokens']:,} training bytes")
        for k, v in speed.items():
            print(f"  {k}: {v}")
        print("  sample:", repr(samples["def "][:200]))
    print("\nbaselines (bits/byte):", out["baselines_bpb"])
    (HERE / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
