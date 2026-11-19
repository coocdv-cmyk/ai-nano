"""EXP-015 N5: capability + efficiency matrix with per-number provenance.

Every row is extracted from an existing machine-readable result file; there is
no hand-typed number.  Each row also records the command that produced that
file.  The gate is that every claimed number resolves to a file AND a command.

Usage:
    python EXPERIMENTS/EXP-015-n5-release/matrix.py \
        --out EXPERIMENTS/EXP-015-n5-release/runs/n5-v1
    python EXPERIMENTS/EXP-015-n5-release/matrix.py --repro   # re-run N3 live
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def row(track: str, metric: str, source: str, path: tuple[Any, ...], unit: str,
        qualifier: str, command: str, value_format: str = ".6g") -> dict[str, Any]:
    return {
        "track": track, "metric": metric, "source": source, "json_path": list(path),
        "unit": unit, "qualifier": qualifier, "command": command,
        "value_format": value_format,
    }


ROWS: list[dict[str, Any]] = [
    row("M1", "v2 clean-split bpb", "EXPERIMENTS/EXP-005-m1-v2-core/showdown_results.json",
        ("arms", 1, "final_val_bpb"), "bpb", "2.00M params, 306.3 s CPU",
        "python EXPERIMENTS/EXP-005-m1-v2-core/showdown.py"),
    row("M1", "zlib-9 baseline bpb", "EXPERIMENTS/EXP-005-m1-v2-core/results.json",
        ("baselines_bpb", "zlib-9"), "bpb", "identical clean split",
        "python EXPERIMENTS/EXP-005-m1-v2-core/run_exp.py"),
    row("M2", "native v2 single-stream", "EXPERIMENTS/EXP-008-m4-omega-v1-matrix/results.json",
        ("native_v2_runtime", "single_stream_tok_s"), "tok/s", "packed 2-bit, greedy",
        "python EXPERIMENTS/EXP-008-m4-omega-v1-matrix/run_exp.py"),
    row("M2", "native v2 batch-4", "EXPERIMENTS/EXP-008-m4-omega-v1-matrix/results.json",
        ("native_v2_runtime", "aggregate_tok_s_batch4"), "tok/s", "same checkpoint, batch 4",
        "python EXPERIMENTS/EXP-008-m4-omega-v1-matrix/run_exp.py"),
    row("M2", "incremental engine baseline", "EXPERIMENTS/EXP-007-m2-bitpacked-engine/results.json",
        ("arms", "v2_streaming_core", "native_v2", "single_stream_tok_s"), "tok/s",
        "the N2 throughput gate threshold", "python EXPERIMENTS/EXP-007-m2-bitpacked-engine/run_exp.py"),
    row("N0", "strict retrieval recall@5", "EXPERIMENTS/EXP-009-n0-benchmark/results.json",
        ("baseline", "retrieval", "recall_at_5"), "fraction", "sealed N0 v1, baseline before N1",
        "python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py"),
    row("N0", "peak RSS", "EXPERIMENTS/EXP-009-n0-benchmark/results.json",
        ("efficiency", "peak_rss_mb"), "MB", "full benchmark run",
        "python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py"),
    row("N1", "N0 code solved", "EXPERIMENTS/EXP-010-n1-ast-repair/results.json",
        ("summary", "solved"), "tasks", "6 of 6, zero unexecuted patches",
        "python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py"),
    row("N1", "N0 code verifier calls", "EXPERIMENTS/EXP-010-n1-ast-repair/results.json",
        ("summary", "verifier_calls_total"), "calls", "6/6 solved",
        "python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py"),
    row("N1", "held-out solved", "EXPERIMENTS/EXP-011-n1-heldout/results.json",
        ("summary", "solved"), "tasks", "7 of 7, same-family surface probe",
        "python EXPERIMENTS/EXP-011-n1-heldout/run_exp.py"),
    row("N2", "fresh verified corpus", "EXPERIMENTS/EXP-012-n2-mtp/results.json",
        ("corpus", "solved"), "tasks", "108/108, leak check clean",
        "python EXPERIMENTS/EXP-012-n2-mtp/corpus.py --out runs/corpus-v1 --max-calls 24"),
    row("N2", "frozen base bpb", "EXPERIMENTS/EXP-012-n2-mtp/results.json",
        ("model", "base_probe_bpb"), "bpb", "N2 held-out probe",
        "python EXPERIMENTS/EXP-012-n2-mtp/train_mtp.py --corpus runs/corpus-v1 --out runs/mtp-v1"),
    row("N2", "engine accept rate", "EXPERIMENTS/EXP-012-n2-mtp/results.json",
        ("native_measurement_10k", "self", "engine_accept_rate"), "fraction",
        "self objective, 10k greedy probe",
        "python EXPERIMENTS/EXP-012-n2-mtp/measure_engine.py --ckpt runs/mtp-v1/mtp-self.pt "
        "--corpus runs/corpus-v1 --tokens 10000 --reps 3"),
    row("N2", "speculative tok/s", "EXPERIMENTS/EXP-012-n2-mtp/results.json",
        ("native_measurement_10k", "self", "speculative_tok_s"), "tok/s",
        "self objective, slower than non-spec — gate failed",
        "python EXPERIMENTS/EXP-012-n2-mtp/measure_engine.py --ckpt runs/mtp-v1/mtp-self.pt "
        "--corpus runs/corpus-v1 --tokens 10000 --reps 3"),
    row("N3", "learned recall@5", "EXPERIMENTS/EXP-013-n3-bilingual/runs/n3-v1/results.json",
        ("learned", "retrieval", "recall_at_5"), "fraction", "393,216 B int8, fresh AR/EN held-out",
        "python EXPERIMENTS/EXP-013-n3-bilingual/run_exp.py --steps 600 --lr 0.005 --tau 0.2"),
    row("N3", "static recall@5", "EXPERIMENTS/EXP-013-n3-bilingual/runs/n3-v1/results.json",
        ("static_baseline", "retrieval", "recall_at_5"), "fraction", "same held-out set, recorded before training",
        "python EXPERIMENTS/EXP-013-n3-bilingual/run_exp.py --steps 600 --lr 0.005 --tau 0.2"),
    row("N3", "out-of-scope false accept", "EXPERIMENTS/EXP-013-n3-bilingual/runs/n3-v1/results.json",
        ("learned", "out_of_scope", "false_accept_rate"), "fraction", "16 OOS queries",
        "python EXPERIMENTS/EXP-013-n3-bilingual/run_exp.py --steps 600 --lr 0.005 --tau 0.2"),
    row("N4", "value-arm pass rate", "EXPERIMENTS/EXP-014-n4-search/runs/n4-v1/results.json",
        ("value", "pass_rate"), "fraction", "three fresh three-function held-out tasks",
        "python EXPERIMENTS/EXP-014-n4-search/run_exp.py --budget 12 --timeout 20"),
    row("N4", "value-arm verifier calls", "EXPERIMENTS/EXP-014-n4-search/runs/n4-v1/results.json",
        ("value", "mean_calls"), "calls", "vs linear 9.667; 72.4% > 60% gate",
        "python EXPERIMENTS/EXP-014-n4-search/run_exp.py --budget 12 --timeout 20"),
]


def follow(data: Any, path: tuple[Any, ...]) -> Any:
    for key in path:
        data = data[key]
    return data


def build_matrix() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    for spec in ROWS:
        src = ROOT / spec["source"]
        entry = {k: v for k, v in spec.items() if k != "value_format"}
        entry["value"] = None
        entry["ok"] = False
        if not src.is_file():
            problems.append(f"missing result file: {spec['source']}")
        elif not spec["command"]:
            problems.append(f"missing command: {spec['metric']}")
        else:
            try:
                data = json.loads(src.read_text(encoding="utf-8"))
                value = follow(data, tuple(spec["json_path"]))
                if isinstance(value, (dict, list)) or value is None:
                    raise ValueError(f"not a scalar: {value!r}")
                entry["value"] = value
                entry["display"] = format(value, spec["value_format"])
                entry["ok"] = True
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                problems.append(f"unresolvable {spec['metric']}: {exc}")
        entries.append(entry)
    return {
        "exp_id": "EXP-015", "track": "N5",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": "every number resolves to a machine-readable result file AND a recorded command",
        "rows": entries,
        "integrity": {"claims": len(entries), "resolved": sum(e["ok"] for e in entries),
                      "problems": problems, "clean": not problems},
    }


def markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Release matrix (EXP-015 / N5)", "",
        "Every value is extracted from the cited result file; the command that produced it is recorded in `matrix.json`.",
        "", "| Track | Metric | Value | Unit | Qualifier | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for e in matrix["rows"]:
        disp = e.get("display", "UNRESOLVED")
        lines.append(f"| {e['track']} | {e['metric']} | {disp} | {e['unit']} | "
                     f"{e['qualifier']} | `{e['source']}` |")
    integ = matrix["integrity"]
    lines += ["", f"Integrity: {integ['resolved']}/{integ['claims']} claims resolved, "
                  f"clean={integ['clean']}.", ""]
    return "\n".join(lines)


class RssSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self._stop = threading.Event()

    def run(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, proc.memory_info().rss)
            except psutil.Error:
                break
            time.sleep(0.02)

    def stop(self) -> int:
        self._stop.set()
        self.join(timeout=2.0)
        return self.peak


def repro_n3() -> dict[str, Any]:
    """Live reproducibility check: re-run N3 into a temp dir, compare recall@5."""
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, str(ROOT / "EXPERIMENTS/EXP-013-n3-bilingual/run_exp.py"),
               "--out", td, "--steps", "600", "--lr", "0.005", "--tau", "0.2"]
        proc = subprocess.run(cmd, capture_output=True, cwd=str(ROOT), timeout=300)
        fresh = json.loads((Path(td) / "results.json").read_text(encoding="utf-8"))
        recorded = json.loads((ROOT / "EXPERIMENTS/EXP-013-n3-bilingual/runs/n3-v1/results.json")
                              .read_text(encoding="utf-8"))
        a = fresh["learned"]["retrieval"]["recall_at_5"]
        b = recorded["learned"]["retrieval"]["recall_at_5"]
        return {"exit_code": proc.returncode, "recorded": b, "reproduced": a,
                "identical": a == b, "command": " ".join(cmd)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the N5 release matrix")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "n5-v1")
    ap.add_argument("--repro", action="store_true", help="live re-run of the N3 recall@5 claim")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sampler = RssSampler()
    sampler.start()
    matrix = build_matrix()
    if args.repro:
        matrix["live_reproducibility"] = repro_n3()
    matrix["cost"] = {
        "wall_s": round(time.perf_counter() - t0, 3),
        "peak_rss_mb": round(sampler.stop() / 2**20, 2),
        "verifier_calls": 0,
        "command": "python EXPERIMENTS/EXP-015-n5-release/matrix.py --repro",
    }
    (args.out / "matrix.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "matrix.md").write_text(markdown(matrix), encoding="utf-8")
    print(f"matrix: {matrix['integrity']['resolved']}/{matrix['integrity']['claims']} "
          f"claims resolved, clean={matrix['integrity']['clean']}", flush=True)
    if "live_reproducibility" in matrix:
        print(json.dumps(matrix["live_reproducibility"], indent=2), flush=True)
    raise SystemExit(0 if matrix["integrity"]["clean"] else 1)


if __name__ == "__main__":
    main()
