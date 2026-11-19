"""Reproduce the recorded N2 patch-boundary hand-off defect.

On the trained `self` checkpoint, unrestricted speculation (drafts allowed to
occupy the final token of a patch) loses byte identity at the next patch
boundary for some prompt alignments. The shipped engine restricts drafts to
non-final patch positions, so this script is expected to report `div=None`.

To reproduce the defect, temporarily change the K clamp in `_v2_run` from
`patch_size - 2 - off_prev` back to `patch_size - 1 - off_prev`.

Run:
    python EXPERIMENTS/EXP-012-n2-mtp/repro_boundary_defect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_nano.core.engine import Engine  # noqa: E402

ckpt = Path(__file__).resolve().parent / "runs" / "mtp-v1" / "mtp-self.pt"
probe = Path(__file__).resolve().parent / "runs" / "corpus-v1" / "probe.bin"
eng = Engine.load(ckpt, packed=True)
eng.spec_adaptive = False
data = probe.read_bytes()
eng.generate(data[:64], 8, temperature=0, speculative=False)  # JIT warm-up

for plen in range(60, 69):
    prompt = data[:plen]
    plain = eng.generate(prompt, 64, temperature=0, speculative=False)
    spec = eng.generate(prompt, 64, temperature=0, speculative=True)
    div = next((i for i, (a, b) in enumerate(zip(plain, spec)) if a != b), None)
    print(f"prompt={plen:2d} first_divergence={div} "
          f"accept={eng.last_stats['spec_accept_rate']}", flush=True)
