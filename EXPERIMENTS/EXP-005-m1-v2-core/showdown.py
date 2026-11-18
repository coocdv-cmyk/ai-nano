"""EXP-005 Part B (M1 Exit Gate Showdown): 2.0M OmegaCore v2 vs 3.34M OmegaCore v0 & zlib-9."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from omega_nano.core.data import baselines, load  # noqa: E402
from omega_nano.core.model import CoreConfig  # noqa: E402
from omega_nano.hw import detect  # noqa: E402
from run_exp import DATA, train_arm  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    train, val = load(DATA)
    base_bpb = json.loads((DATA / "baselines.json").read_text(encoding="utf-8")) if (DATA / "baselines.json").exists() \
        else baselines(train, val)

    arms = [
        ("v0_3.34M_5min", CoreConfig(d=256, layers=4, heads=4, hidden=704, ctx=256, arch="v0")),
        ("v2_2.00M_5min", CoreConfig(
            d=192, layers=2, heads=4, hidden=384, ctx=256, arch="v2",
            n_loops=2, mtp_k=1, ponder_weight=0.001, patch_size=4,
            hash_buckets=4096, fast_dim=16, use_fast_weights=True,
            use_copy_gate=True, copy_window=256,
        )),
    ]

    out = {
        "experiment": "EXP-005-showdown",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "hardware": detect(),
        "baselines_bpb": base_bpb,
        "arms": [],
    }
    for name, cfg in arms:
        print(f"\n=== Training {name} ({a.seconds}s) ===", flush=True)
        res, model = train_arm(name, cfg, train, val, seconds=a.seconds, threads=a.threads)
        sample = model.generate(b"def ", 160, temperature=0.6, seed=7).decode("utf-8", "replace")
        res["sample_def"] = sample
        out["arms"].append(res)
        print(
            f"{name}: params={res['params']['total']:,} steps={res['steps']} "
            f"val_bpb={res['final_val_bpb']} vs zlib9={base_bpb['zlib-9']}",
            flush=True,
        )
        print("  sample:", repr(sample[:140]), flush=True)

    (HERE / "showdown_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved showdown_results.json", flush=True)


if __name__ == "__main__":
    main()
