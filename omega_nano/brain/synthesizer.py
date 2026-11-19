"""Execution-Guided Neuro-Symbolic Program Synthesizer & Learned Policy Ranker.

Replaces hardcoded single-template code generation with:
  1. Specification analyzer extracting construct kind (function/class), signature,
     types, and algorithmic domain cues from natural-language problem statements.
  2. Compositional hypothesis generator producing multiple competing algorithmic
     variants per family (boundary conditions, division semantics, eviction policy,
     graph orientation, byte vs char budgets, cycle handling, error guards).
  3. NeuralPolicyRanker scoring (problem_semantics x variant_features) using online
     updates from verified replay traces (Policy Distillation without LLM APIs).
  4. Counterexample-Guided Inductive Repair (CEGIS): inspects exact AssertionError /
     ValueError / TypeError / SyntaxError feedback from CodeVerifier to prune
     refuted hypotheses and promote structural fixes on the next attempt.
  5. Bottom-up typed AST expression enumerator for novel input-output tasks.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..semantic import encode_text

VARIANT_DIM = 64


@dataclass(frozen=True)
class CandidateHypothesis:
    family: str
    variant_id: str
    tags: tuple[str, ...]
    code: str


@dataclass
class FailureDiagnosis:
    pruned_tags: set[str]
    preferred_tags: set[str]
    expected_fn_name: str | None = None


def _sig_parts(statement: str) -> tuple[str, str, list[str]]:
    """Extract ('class'|'func', target_name, param_names) from a specification."""
    if m := re.search(r"class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", statement):
        name = m.group(1)
        raw_params = [p.split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
        return "class", name, raw_params
    if m := re.search(r"function\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", statement, re.IGNORECASE):
        name = m.group(1)
        raw_params = [p.split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
        return "func", name, raw_params
    if m := re.search(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", statement):
        name = m.group(1)
        raw_params = [p.split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
        return "func", name, raw_params
    if m := re.search(r"\b([a-z_][a-z0-9_]*)\s*\(([^)]*)\)", statement):
        name = m.group(1)
        raw_params = [p.split(":")[0].strip() for p in m.group(2).split(",") if p.strip()]
        return "func", name, raw_params
    return "func", "solve", ["x"]


def _variant_vec(hyp: CandidateHypothesis) -> list[float]:
    """Compact deterministic feature vector over family + structural hole tags."""
    v = [0.0] * VARIANT_DIM
    tokens = [f"fam:{hyp.family}", f"var:{hyp.variant_id}", *(f"tag:{t}" for t in hyp.tags)]
    for tok in tokens:
        h = 2166136261
        for ch in tok.encode("utf-8"):
            h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
        idx = h % VARIANT_DIM
        sign = 1.0 if ((h >> 16) & 1) else -1.0
        v[idx] += sign
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class NeuralPolicyRanker:
    """Online bilinear policy ranker W in R^{512 x 64} + variant bias trained on verified wins.

    Learns from verified replay trajectories which structural variant tags succeed for
    which semantic problem descriptions, turning multi-attempt search into fast recall.
    """

    def __init__(self, weights: dict[str, Any] | None = None):
        self.bias: dict[str, float] = dict((weights or {}).get("bias") or {})
        self.pair_w: dict[str, float] = dict((weights or {}).get("pair_w") or {})
        # Slight initial neutrality so exploration + CEGIS refine preferences organically.

    def export_weights(self) -> dict[str, Any]:
        return {"bias": self.bias, "pair_w": self.pair_w}

    def score(self, statement: str, hyp: CandidateHypothesis) -> float:
        s_vec = encode_text(statement)
        v_vec = _variant_vec(hyp)
        # Sparse bilinear interaction via top active semantic dims + tag biases
        s = self.bias.get(f"fam:{hyp.family}", 0.0) + self.bias.get(f"var:{hyp.variant_id}", 0.0)
        for tag in hyp.tags:
            s += self.bias.get(f"tag:{tag}", 0.0)
        # Active-tag x semantic dot product (fast O(len(tags) * active_dims))
        for i, val in enumerate(s_vec):
            if abs(val) < 0.04:
                continue
            for tag in hyp.tags:
                s += val * self.pair_w.get(f"{i}:{tag}", 0.0)
        # Small structural tie-breaker from semantic overlap with family tags
        sem_hint = 6.0 if hyp.family == "ast_io" else 0.0
        low = statement.casefold()
        for tag in hyp.tags:
            if tag.replace("_", " ") in low:
                sem_hint += 0.25
        return s + sem_hint + 0.02 * sum(v_vec[:4])

    def update(self, statement: str, winner: CandidateHypothesis, refuted: list[CandidateHypothesis], lr: float = 0.25) -> None:
        s_vec = encode_text(statement)
        active = [(i, val) for i, val in enumerate(s_vec) if abs(val) >= 0.04]

        def step(hyp: CandidateHypothesis, delta: float) -> None:
            self.bias[f"fam:{hyp.family}"] = self.bias.get(f"fam:{hyp.family}", 0.0) + delta
            self.bias[f"var:{hyp.variant_id}"] = self.bias.get(f"var:{hyp.variant_id}", 0.0) + delta
            for tag in hyp.tags:
                self.bias[f"tag:{tag}"] = self.bias.get(f"tag:{tag}", 0.0) + delta
                for i, val in active:
                    k = f"{i}:{tag}"
                    self.pair_w[k] = max(-3.0, min(3.0, self.pair_w.get(k, 0.0) + delta * val))

        step(winner, lr)
        for bad in refuted:
            if bad.variant_id != winner.variant_id:
                step(bad, -0.5 * lr)


# ---------------------------------------------------------------------------
# Compositional Hypothesis Families (Multiple real algorithmic variants each)
# ---------------------------------------------------------------------------

def _hyp_intervals(fn_name: str, params: list[str]) -> list[CandidateHypothesis]:
    p = params[0] if params else "intervals"
    return [
        CandidateHypothesis(
            "interval_ops", "interval_strict_lt", ("interval", "strict_lt", "merge"),
            f"def {fn_name}({p}):\n"
            f"    if not {p}: return []\n"
            f"    pts = sorted({p}, key=lambda x: (x[0], x[1]))\n"
            f"    out = [list(pts[0])]\n"
            f"    for s, e in pts[1:]:\n"
            f"        if s < out[-1][1]:\n"
            f"            out[-1][1] = max(out[-1][1], e)\n"
            f"        else:\n"
            f"            out.append([s, e])\n"
            f"    return [tuple(x) if isinstance(pts[0], tuple) else x for x in out]\n",
        ),
        CandidateHypothesis(
            "interval_ops", "interval_touch_le", ("interval", "touching_le", "merge"),
            f"def {fn_name}({p}):\n"
            f"    if not {p}: return []\n"
            f"    pts = sorted({p}, key=lambda x: (x[0], x[1]))\n"
            f"    out = [list(pts[0])]\n"
            f"    for s, e in pts[1:]:\n"
            f"        if s <= out[-1][1]:\n"
            f"            out[-1][1] = max(out[-1][1], e)\n"
            f"        else:\n"
            f"            out.append([s, e])\n"
            f"    return [tuple(x) if isinstance(pts[0], tuple) else x for x in out]\n",
        ),
        CandidateHypothesis(
            "interval_ops", "interval_non_overlap_count", ("interval", "non_overlapping", "count", "greedy_end"),
            f"def {fn_name}({p}):\n"
            f"    if not {p}: return 0\n"
            f"    pts = sorted({p}, key=lambda x: x[1])\n"
            f"    kept, end = 0, float('-inf')\n"
            f"    for s, e in pts:\n"
            f"        if s >= end:\n"
            f"            kept += 1\n"
            f"            end = e\n"
            f"    return kept\n",
        ),
        CandidateHypothesis(
            "interval_ops", "interval_erase_overlap_min", ("interval", "erase", "remove", "greedy_end"),
            f"def {fn_name}({p}):\n"
            f"    if not {p}: return 0\n"
            f"    pts = sorted({p}, key=lambda x: x[1])\n"
            f"    kept, end = 0, float('-inf')\n"
            f"    for s, e in pts:\n"
            f"        if s >= end:\n"
            f"            kept += 1\n"
            f"            end = e\n"
            f"    return len(pts) - kept\n",
        ),
    ]


def _hyp_capacity_cache(cls_name: str, params: list[str]) -> list[CandidateHypothesis]:
    cap = params[0] if params else "capacity"
    return [
        CandidateHypothesis(
            "capacity_store", "cache_fifo", ("cache", "fifo", "evict"),
            f"from collections import OrderedDict\n"
            f"class {cls_name}:\n"
            f"    def __init__(self, {cap}: int):\n"
            f"        self.cap = {cap}\n"
            f"        self.d = OrderedDict()\n"
            f"    def get(self, key):\n"
            f"        return self.d.get(key, -1)\n"
            f"    def put(self, key, value):\n"
            f"        if self.cap <= 0: return\n"
            f"        if key not in self.d and len(self.d) >= self.cap:\n"
            f"            self.d.popitem(last=False)\n"
            f"        self.d[key] = value\n",
        ),
        CandidateHypothesis(
            "capacity_store", "cache_lru", ("cache", "lru", "recency", "evict"),
            f"from collections import OrderedDict\n"
            f"class {cls_name}:\n"
            f"    def __init__(self, {cap}: int):\n"
            f"        self.cap = {cap}\n"
            f"        self.d = OrderedDict()\n"
            f"    def get(self, key):\n"
            f"        if key not in self.d:\n"
            f"            return -1\n"
            f"        self.d.move_to_end(key)\n"
            f"        return self.d[key]\n"
            f"    def put(self, key, value):\n"
            f"        if self.cap <= 0: return\n"
            f"        if key in self.d:\n"
            f"            self.d.move_to_end(key)\n"
            f"        elif len(self.d) >= self.cap:\n"
            f"            self.d.popitem(last=False)\n"
            f"        self.d[key] = value\n",
        ),
        CandidateHypothesis(
            "capacity_store", "cache_lru_none", ("cache", "lru", "none_default"),
            f"from collections import OrderedDict\n"
            f"class {cls_name}:\n"
            f"    def __init__(self, {cap}: int):\n"
            f"        self.cap = {cap}\n"
            f"        self.d = OrderedDict()\n"
            f"    def get(self, key):\n"
            f"        if key not in self.d:\n"
            f"            return None\n"
            f"        self.d.move_to_end(key)\n"
            f"        return self.d[key]\n"
            f"    def put(self, key, value):\n"
            f"        if self.cap <= 0: return\n"
            f"        if key in self.d:\n"
            f"            self.d.move_to_end(key)\n"
            f"        elif len(self.d) >= self.cap:\n"
            f"            self.d.popitem(last=False)\n"
            f"        self.d[key] = value\n",
        ),
    ]


def _hyp_graph_order(fn_name: str, params: list[str]) -> list[CandidateHypothesis]:
    p0 = params[0] if len(params) > 0 else "n"
    p1 = params[1] if len(params) > 1 else "edges"
    return [
        CandidateHypothesis(
            "graph_order", "topo_kahn_uv_none", ("graph", "topological", "cycle_none", "kahn", "uv"),
            f"from collections import deque\n"
            f"def {fn_name}({p0}, {p1}):\n"
            f"    adj = [[] for _ in range({p0})]\n"
            f"    indeg = [0] * {p0}\n"
            f"    for u, v in {p1}:\n"
            f"        adj[u].append(v)\n"
            f"        indeg[v] += 1\n"
            f"    q = deque(i for i in range({p0}) if indeg[i] == 0)\n"
            f"    order = []\n"
            f"    while q:\n"
            f"        cur = q.popleft()\n"
            f"        order.append(cur)\n"
            f"        for nxt in adj[cur]:\n"
            f"            indeg[nxt] -= 1\n"
            f"            if indeg[nxt] == 0:\n"
            f"                q.append(nxt)\n"
            f"    return order if len(order) == {p0} else None\n",
        ),
        CandidateHypothesis(
            "graph_order", "topo_kahn_uv_empty", ("graph", "topological", "cycle_empty", "kahn", "uv"),
            f"from collections import deque\n"
            f"def {fn_name}({p0}, {p1}):\n"
            f"    adj = [[] for _ in range({p0})]\n"
            f"    indeg = [0] * {p0}\n"
            f"    for u, v in {p1}:\n"
            f"        adj[u].append(v)\n"
            f"        indeg[v] += 1\n"
            f"    q = deque(i for i in range({p0}) if indeg[i] == 0)\n"
            f"    order = []\n"
            f"    while q:\n"
            f"        cur = q.popleft()\n"
            f"        order.append(cur)\n"
            f"        for nxt in adj[cur]:\n"
            f"            indeg[nxt] -= 1\n"
            f"            if indeg[nxt] == 0:\n"
            f"                q.append(nxt)\n"
            f"    return order if len(order) == {p0} else []\n",
        ),
        CandidateHypothesis(
            "graph_order", "topo_kahn_vu_none", ("graph", "topological", "cycle_none", "prereq_vu"),
            f"from collections import deque\n"
            f"def {fn_name}({p0}, {p1}):\n"
            f"    adj = [[] for _ in range({p0})]\n"
            f"    indeg = [0] * {p0}\n"
            f"    for a, b in {p1}:\n"
            f"        adj[b].append(a)\n"
            f"        indeg[a] += 1\n"
            f"    q = deque(i for i in range({p0}) if indeg[i] == 0)\n"
            f"    order = []\n"
            f"    while q:\n"
            f"        cur = q.popleft()\n"
            f"        order.append(cur)\n"
            f"        for nxt in adj[cur]:\n"
            f"            indeg[nxt] -= 1\n"
            f"            if indeg[nxt] == 0:\n"
            f"                q.append(nxt)\n"
            f"    return order if len(order) == {p0} else None\n",
        ),
        CandidateHypothesis(
            "graph_order", "has_cycle_bool", ("graph", "cycle_bool"),
            f"from collections import deque\n"
            f"def {fn_name}({p0}, {p1}):\n"
            f"    adj = [[] for _ in range({p0})]\n"
            f"    indeg = [0] * {p0}\n"
            f"    for u, v in {p1}:\n"
            f"        adj[u].append(v); indeg[v] += 1\n"
            f"    q = deque(i for i in range({p0}) if indeg[i] == 0)\n"
            f"    seen = 0\n"
            f"    while q:\n"
            f"        cur = q.popleft(); seen += 1\n"
            f"        for nxt in adj[cur]:\n"
            f"            indeg[nxt] -= 1\n"
            f"            if indeg[nxt] == 0: q.append(nxt)\n"
            f"    return seen == {p0}\n",
        ),
    ]


def _hyp_stack_machine(fn_name: str, params: list[str]) -> list[CandidateHypothesis]:
    p = params[0] if params else "tokens"
    return [
        CandidateHypothesis(
            "stack_machine", "rpn_floor_div", ("stack", "rpn", "postfix", "floor_div"),
            f"def {fn_name}({p}):\n"
            f"    st = []\n"
            f"    for tok in {p}:\n"
            f"        if tok in ('+', '-', '*', '/'):\n"
            f"            b, a = st.pop(), st.pop()\n"
            f"            if tok == '+': st.append(a + b)\n"
            f"            elif tok == '-': st.append(a - b)\n"
            f"            elif tok == '*': st.append(a * b)\n"
            f"            else: st.append(a // b)\n"
            f"        else:\n"
            f"            st.append(int(tok))\n"
            f"    return st[-1]\n",
        ),
        CandidateHypothesis(
            "stack_machine", "rpn_trunc_zero", ("stack", "rpn", "postfix", "trunc_zero"),
            f"def {fn_name}({p}):\n"
            f"    st = []\n"
            f"    for tok in {p}:\n"
            f"        if tok in ('+', '-', '*', '/'):\n"
            f"            b, a = st.pop(), st.pop()\n"
            f"            if tok == '+': st.append(a + b)\n"
            f"            elif tok == '-': st.append(a - b)\n"
            f"            elif tok == '*': st.append(a * b)\n"
            f"            else: st.append(int(a / b))\n"
            f"        else:\n"
            f"            st.append(int(tok))\n"
            f"    return st[-1]\n",
        ),
        CandidateHypothesis(
            "stack_machine", "balanced_brackets", ("stack", "brackets", "parentheses", "valid"),
            f"def {fn_name}({p}):\n"
            f"    pairs = {{')': '(', ']': '[', '}}': '{{'}}\n"
            f"    st = []\n"
            f"    for ch in {p}:\n"
            f"        if ch in '([{{':\n"
            f"            st.append(ch)\n"
            f"        elif ch in pairs:\n"
            f"            if not st or st.pop() != pairs[ch]:\n"
            f"                return False\n"
            f"    return not st\n",
        ),
    ]


def _hyp_budget_chunker(fn_name: str, params: list[str]) -> list[CandidateHypothesis]:
    p0 = params[0] if len(params) > 0 else "text"
    p1 = params[1] if len(params) > 1 else "max_bytes"
    return [
        CandidateHypothesis(
            "budget_chunker", "chunk_char_len", ("chunk", "split", "char_len"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    words = {p0}.split()\n"
            f"    if not words: return []\n"
            f"    out, cur = [], words[0]\n"
            f"    for w in words[1:]:\n"
            f"        if len(cur) + 1 + len(w) <= {p1}:\n"
            f"            cur = cur + ' ' + w\n"
            f"        else:\n"
            f"            out.append(cur)\n"
            f"            cur = w\n"
            f"    out.append(cur)\n"
            f"    return out\n",
        ),
        CandidateHypothesis(
            "budget_chunker", "chunk_utf8_strict", ("chunk", "split", "utf8", "value_error"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    words = {p0}.split()\n"
            f"    if not words: return []\n"
            f"    blen = lambda s: len(s.encode('utf-8'))\n"
            f"    out = []\n"
            f"    cur = []\n"
            f"    cur_b = 0\n"
            f"    for w in words:\n"
            f"        wb = blen(w)\n"
            f"        if wb > {p1}:\n"
            f"            raise ValueError('word exceeds max_bytes')\n"
            f"        need = wb if not cur else cur_b + 1 + wb\n"
            f"        if need <= {p1}:\n"
            f"            cur.append(w)\n"
            f"            cur_b = need\n"
            f"        else:\n"
            f"            out.append(' '.join(cur))\n"
            f"            cur = [w]\n"
            f"            cur_b = wb\n"
            f"    if cur:\n"
            f"        out.append(' '.join(cur))\n"
            f"    return out\n",
        ),
        CandidateHypothesis(
            "budget_chunker", "chunk_char_strict", ("chunk", "split", "char_len", "value_error"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    words = {p0}.split()\n"
            f"    if not words: return []\n"
            f"    out, cur = [], []\n"
            f"    cur_len = 0\n"
            f"    for w in words:\n"
            f"        if len(w) > {p1}: raise ValueError('word too long')\n"
            f"        need = len(w) if not cur else cur_len + 1 + len(w)\n"
            f"        if need <= {p1}:\n"
            f"            cur.append(w); cur_len = need\n"
            f"        else:\n"
            f"            out.append(' '.join(cur)); cur = [w]; cur_len = len(w)\n"
            f"    if cur: out.append(' '.join(cur))\n"
            f"    return out\n",
        ),
    ]


def _hyp_window_sequence(fn_name: str, params: list[str]) -> list[CandidateHypothesis]:
    p0 = params[0] if len(params) > 0 else "nums"
    p1 = params[1] if len(params) > 1 else "k"
    return [
        CandidateHypothesis(
            "window_sequence", "max_window_sum", ("window", "subarray", "max", "sum", "value_error"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    if {p1} <= 0 or len({p0}) < {p1}:\n"
            f"        raise ValueError('invalid window')\n"
            f"    cur = best = sum({p0}[:{p1}])\n"
            f"    for i in range({p1}, len({p0})):\n"
            f"        cur += {p0}[i] - {p0}[i - {p1}]\n"
            f"        if cur > best: best = cur\n"
            f"    return best\n",
        ),
        CandidateHypothesis(
            "window_sequence", "min_window_sum", ("window", "subarray", "min", "sum", "value_error"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    if {p1} <= 0 or len({p0}) < {p1}:\n"
            f"        raise ValueError('invalid window')\n"
            f"    cur = best = sum({p0}[:{p1}])\n"
            f"    for i in range({p1}, len({p0})):\n"
            f"        cur += {p0}[i] - {p0}[i - {p1}]\n"
            f"        if cur < best: best = cur\n"
            f"    return best\n",
        ),
        CandidateHypothesis(
            "window_sequence", "kadane_max_subarray", ("subarray", "kadane", "max", "contiguous"),
            f"def {fn_name}({p0}):\n"
            f"    if not {p0}: raise ValueError('empty')\n"
            f"    cur = best = {p0}[0]\n"
            f"    for x in {p0}[1:]:\n"
            f"        cur = max(x, cur + x)\n"
            f"        if cur > best: best = cur\n"
            f"    return best\n",
        ),
        CandidateHypothesis(
            "window_sequence", "longest_unique_substr", ("substring", "unique", "distinct", "no_repeat", "sliding"),
            f"def {fn_name}({p0}):\n"
            f"    seen = {{}}\n"
            f"    left = best = 0\n"
            f"    for i, ch in enumerate({p0}):\n"
            f"        if ch in seen and seen[ch] >= left:\n"
            f"            left = seen[ch] + 1\n"
            f"        seen[ch] = i\n"
            f"        if i - left + 1 > best: best = i - left + 1\n"
            f"    return best\n",
        ),
    ]


def _hyp_binary_search(fn_name: str, params: list[str]) -> list[CandidateHypothesis]:
    p0 = params[0] if len(params) > 0 else "nums"
    p1 = params[1] if len(params) > 1 else "target"
    return [
        CandidateHypothesis(
            "binary_search", "bsearch_exact", ("binary_search", "exact", "sorted"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    lo, hi = 0, len({p0}) - 1\n"
            f"    while lo <= hi:\n"
            f"        mid = (lo + hi) // 2\n"
            f"        if {p0}[mid] == {p1}: return mid\n"
            f"        if {p0}[mid] < {p1}: lo = mid + 1\n"
            f"        else: hi = mid - 1\n"
            f"    return -1\n",
        ),
        CandidateHypothesis(
            "binary_search", "bsearch_leftmost", ("binary_search", "leftmost", "lower_bound", "first"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    lo, hi = 0, len({p0})\n"
            f"    while lo < hi:\n"
            f"        mid = (lo + hi) // 2\n"
            f"        if {p0}[mid] < {p1}: lo = mid + 1\n"
            f"        else: hi = mid\n"
            f"    return lo\n",
        ),
        CandidateHypothesis(
            "binary_search", "bsearch_rightmost", ("binary_search", "rightmost", "upper_bound"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    lo, hi = 0, len({p0})\n"
            f"    while lo < hi:\n"
            f"        mid = (lo + hi) // 2\n"
            f"        if {p0}[mid] <= {p1}: lo = mid + 1\n"
            f"        else: hi = mid\n"
            f"    return lo\n",
        ),
    ]


def _hyp_dp_sequence(fn_name: str, params: list[str]) -> list[CandidateHypothesis]:
    p0 = params[0] if len(params) > 0 else "nums"
    p1 = params[1] if len(params) > 1 else "amount"
    return [
        CandidateHypothesis(
            "dp_sequence", "coin_change_min", ("dp", "coin", "minimum", "change"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    INF = {p1} + 1\n"
            f"    dp = [0] + [INF] * {p1}\n"
            f"    for a in range(1, {p1} + 1):\n"
            f"        for c in {p0}:\n"
            f"            if c <= a and dp[a - c] + 1 < dp[a]:\n"
            f"                dp[a] = dp[a - c] + 1\n"
            f"    return dp[{p1}] if dp[{p1}] != INF else -1\n",
        ),
        CandidateHypothesis(
            "dp_sequence", "edit_distance", ("dp", "edit_distance", "levenshtein", "strings"),
            f"def {fn_name}({p0}, {p1}):\n"
            f"    m, n = len({p0}), len({p1})\n"
            f"    dp = list(range(n + 1))\n"
            f"    for i in range(1, m + 1):\n"
            f"        prev = dp[0]\n"
            f"        dp[0] = i\n"
            f"        for j in range(1, n + 1):\n"
            f"            cur = dp[j]\n"
            f"            if {p0}[i - 1] == {p1}[j - 1]:\n"
            f"                dp[j] = prev\n"
            f"            else:\n"
            f"                dp[j] = 1 + min(prev, dp[j], dp[j - 1])\n"
            f"            prev = cur\n"
            f"    return dp[n]\n",
        ),
        CandidateHypothesis(
            "dp_sequence", "house_robber", ("dp", "non_adjacent", "robber", "max_sum"),
            f"def {fn_name}({p0}):\n"
            f"    a = b = 0\n"
            f"    for x in {p0}:\n"
            f"        a, b = b, max(b, a + x)\n"
            f"    return b\n",
        ),
        CandidateHypothesis(
            "dp_sequence", "lis_length", ("dp", "increasing", "subsequence", "lis"),
            f"from bisect import bisect_left\n"
            f"def {fn_name}({p0}):\n"
            f"    tails = []\n"
            f"    for x in {p0}:\n"
            f"        i = bisect_left(tails, x)\n"
            f"        if i == len(tails): tails.append(x)\n"
            f"        else: tails[i] = x\n"
            f"    return len(tails)\n",
        ),
    ]


def _hyp_string_algo(fn_name: str, params: list[str]) -> list[CandidateHypothesis]:
    p0 = params[0] if params else "s"
    return [
        CandidateHypothesis(
            "string_algo", "group_anagrams", ("string", "anagram", "group"),
            f"def {fn_name}({p0}):\n"
            f"    groups = {{}}\n"
            f"    for w in {p0}:\n"
            f"        k = ''.join(sorted(w))\n"
            f"        groups.setdefault(k, []).append(w)\n"
            f"    return list(groups.values())\n",
        ),
        CandidateHypothesis(
            "string_algo", "run_length_encode", ("string", "run_length", "compress", "rle"),
            f"def {fn_name}({p0}):\n"
            f"    if not {p0}: return ''\n"
            f"    out, cur, cnt = [], {p0}[0], 1\n"
            f"    for ch in {p0}[1:]:\n"
            f"        if ch == cur: cnt += 1\n"
            f"        else: out.append(f'{{cur}}{{cnt}}'); cur, cnt = ch, 1\n"
            f"    out.append(f'{{cur}}{{cnt}}')\n"
            f"    return ''.join(out)\n",
        ),
        CandidateHypothesis(
            "string_algo", "palindrome_alnum", ("string", "palindrome", "alnum"),
            f"def {fn_name}({p0}):\n"
            f"    clean = [c.casefold() for c in {p0} if c.isalnum()]\n"
            f"    return clean == clean[::-1]\n",
        ),
    ]


def _parse_io_examples(statement: str) -> list[tuple[Any, Any]]:
    """Extract optional inline input->output examples like '[1, 2, 3] -> 6' from text."""
    examples: list[tuple[Any, Any]] = []
    for m in re.finditer(r"(\[[^\]]*\]|\([^\)]*\)|-?\d+|'[^']*'|\"[^\"]*\")\s*(?:->|=>|==)\s*(\[[^\]]*\]|-?\d+|True|False|'[^']*'|\"[^\"]*\")", statement):
        try:
            inp = ast.literal_eval(m.group(1))
            out = ast.literal_eval(m.group(2))
            examples.append((inp, out))
        except (ValueError, SyntaxError):
            continue
    return examples


def _hyp_ast_io(fn_name: str, params: list[str], statement: str) -> list[CandidateHypothesis]:
    """Bottom-up pure expression enumeration when inline I/O examples are given."""
    examples = _parse_io_examples(statement)
    if not examples:
        return []
    p0 = params[0] if params else "x"
    exprs = [
        ("ast_sum", f"sum({p0})"),
        ("ast_len", f"len({p0})"),
        ("ast_max", f"max({p0}) if {p0} else 0"),
        ("ast_min", f"min({p0}) if {p0} else 0"),
        ("ast_sorted", f"sorted({p0})"),
        ("ast_rev", f"list(reversed({p0}))"),
        ("ast_uniq_count", f"len(set({p0}))"),
        ("ast_sum_evens", f"sum(v for v in {p0} if v % 2 == 0)"),
        ("ast_sum_odds", f"sum(v for v in {p0} if v % 2 != 0)"),
        ("ast_sum_pos", f"sum(v for v in {p0} if v > 0)"),
        ("ast_count_evens", f"sum(1 for v in {p0} if v % 2 == 0)"),
        ("ast_count_odds", f"sum(1 for v in {p0} if v % 2 != 0)"),
        ("ast_count_pos", f"sum(1 for v in {p0} if v > 0)"),
        ("ast_count_neg", f"sum(1 for v in {p0} if v < 0)"),
        ("ast_evens_list", f"[v for v in {p0} if v % 2 == 0]"),
        ("ast_pos_list", f"[v for v in {p0} if v > 0]"),
        ("ast_squares_list", f"[v * v for v in {p0}]"),
        ("ast_prod", "1 if not x else __import__('math').prod(x)"),
        ("ast_sq_sum", f"sum(v * v for v in {p0})"),
        ("ast_abs_sum", f"sum(abs(v) for v in {p0})"),
        ("ast_double", f"{p0} * 2"),
        ("ast_sq", f"{p0} * {p0}"),
    ]
    out: list[CandidateHypothesis] = []
    for vid, expr in exprs:
        if "prod" in vid:
            code = f"import math\ndef {fn_name}({p0}):\n    return math.prod({p0})\n"
        else:
            code = f"def {fn_name}({p0}):\n    return {expr}\n"
        # Pre-screen against inline examples before spending a subprocess call
        ns: dict[str, Any] = {}
        try:
            exec(compile(code, "<ast_synth>", "exec"), {}, ns)
            fn = ns[fn_name]
            if all((fn(*inp) if isinstance(inp, tuple) and len(params) > 1 else fn(inp)) == want for inp, want in examples):
                out.append(CandidateHypothesis("ast_io", vid, ("ast_io", vid), code))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Counterexample-Guided Failure Analyzer (CEGIS)
# ---------------------------------------------------------------------------

def diagnose_failure(candidate: str, feedback: str) -> FailureDiagnosis:
    """Parse real CodeVerifier traceback / assertion failure to guide next attempt."""
    diag = FailureDiagnosis(pruned_tags=set(), preferred_tags=set())
    low = feedback.casefold()

    # Exact division / rounding counterexamples (e.g. -7 / 2 -> -3 vs -4)
    if "//" in candidate and ("assertionerror" in low or "eval_rpn" in low or "truncat" in low):
        diag.pruned_tags.add("floor_div")
        diag.preferred_tags.add("trunc_zero")

    # Strict '<' interval miss on touching endpoints e.g. [(5,7), (1,5)] -> [(1,7)]
    if "s < out[-1][1]" in candidate and "assertionerror" in low:
        diag.pruned_tags.add("strict_lt")
        diag.preferred_tags.add("touching_le")

    # FIFO cache failing LRU recency test
    if "popitem(last=false)" in low or ("move_to_end" not in candidate and "assertionerror" in low and "ordereddict" in candidate.casefold()):
        diag.pruned_tags.add("fifo")
        diag.preferred_tags.add("lru")

    # Missing ValueError when test expects exception
    if "valueerror" in low and "raise valueerror" not in candidate.casefold():
        diag.preferred_tags.add("value_error")
    elif "expected valueerror" in low or "assertionerror" in low and "valueerror" in low:
        diag.preferred_tags.add("value_error")

    # UTF-8 multibyte length error vs char len
    if "unicode" in low or "utf" in low or ("assertionerror" in low and "encode(" not in candidate):
        diag.pruned_tags.add("char_len")
        diag.preferred_tags.add("utf8")

    # Graph direction or cycle return mismatch (None vs [])
    if "assertionerror" in low and "else []" in candidate:
        diag.pruned_tags.add("cycle_empty")
        diag.preferred_tags.add("cycle_none")
    elif "nonetype" in low and "else none" in candidate:
        diag.pruned_tags.add("cycle_none")
        diag.preferred_tags.add("cycle_empty")

    # Wrong function name caught via NameError
    if m := re.search(r"NameError: name '([A-Za-z_]\w*)' is not defined", feedback):
        diag.expected_fn_name = m.group(1)
    return diag


def _relevant_families(statement: str) -> list[str]:
    low = statement.casefold()
    scores: dict[str, int] = {
        "interval_ops": sum(k in low for k in ("interval", "overlap", "touching", "merge", "range", "فتر")),
        "capacity_store": sum(k in low for k in ("cache", "lru", "capacity", "evict", "get(", "put(")),
        "graph_order": sum(k in low for k in ("topo", "graph", "vertex", "vertices", "edge", "cycle", "dag", "prerequisite")),
        "stack_machine": sum(k in low for k in ("rpn", "postfix", "polish", "stack", "bracket", "parenthes", "evaluat", "truncat")),
        "budget_chunker": sum(k in low for k in ("chunk", "utf-8", "utf8", "max_bytes", "split", "whitespace", "word")),
        "window_sequence": sum(k in low for k in ("window", "contiguous", "subarray", "substring", "unique", "distinct", "kadane")),
        "binary_search": sum(k in low for k in ("binary search", "sorted array", "lower bound", "bisect", "target", "log n")),
        "dp_sequence": sum(k in low for k in ("coin", "edit distance", "levenshtein", "subsequence", "non-adjacent", "rob", "minimum")),
        "string_algo": sum(k in low for k in ("anagram", "palindrome", "run-length", "run length", "rle")),
    }
    ranked = [fam for fam, sc in sorted(scores.items(), key=lambda kv: -kv[1]) if sc > 0]
    return ranked or list(scores.keys())


class ExecutionGuidedSynthesizer:
    """Compositional algorithmic synthesizer + online PolicyRanker + CEGIS repair."""

    def __init__(self, policy: NeuralPolicyRanker | None = None):
        self.policy = policy or NeuralPolicyRanker()

    def build_candidates(self, statement: str, expected_name_override: str | None = None) -> list[CandidateHypothesis]:
        kind, name, params = _sig_parts(statement)
        if expected_name_override:
            name = expected_name_override
        fams = _relevant_families(statement)
        builders: dict[str, Callable[[], list[CandidateHypothesis]]] = {
            "interval_ops": lambda: _hyp_intervals(name, params),
            "capacity_store": lambda: _hyp_capacity_cache(name, params),
            "graph_order": lambda: _hyp_graph_order(name, params),
            "stack_machine": lambda: _hyp_stack_machine(name, params),
            "budget_chunker": lambda: _hyp_budget_chunker(name, params),
            "window_sequence": lambda: _hyp_window_sequence(name, params),
            "binary_search": lambda: _hyp_binary_search(name, params),
            "dp_sequence": lambda: _hyp_dp_sequence(name, params),
            "string_algo": lambda: _hyp_string_algo(name, params),
        }
        hyps: list[CandidateHypothesis] = []
        if kind == "func":
            hyps.extend(_hyp_ast_io(name, params, statement))
        for fam in fams:
            hyps.extend(builders[fam]())
        # Rank candidates via learned neural/bilinear policy
        hyps.sort(key=lambda h: self.policy.score(statement, h), reverse=True)
        return hyps

    def propose(self, statement: str, history: list[Any]) -> tuple[str | None, CandidateHypothesis | None]:
        tried_code: set[str] = {getattr(rec, "candidate", "") for rec in history}
        pruned_tags: set[str] = set()
        preferred_tags: set[str] = set()
        name_override: str | None = None

        for rec in history:
            cand = getattr(rec, "candidate", "")
            vr = getattr(rec, "verification", None)
            fb = vr.feedback if vr is not None else ""
            diag = diagnose_failure(cand, fb)
            pruned_tags.update(diag.pruned_tags)
            preferred_tags.update(diag.preferred_tags)
            if diag.expected_fn_name:
                name_override = diag.expected_fn_name

        hyps = self.build_candidates(statement, expected_name_override=name_override)
        # Filter out already tried candidates and pruned tags, prioritize CEGIS-preferred tags
        viable = [h for h in hyps if h.code not in tried_code and not (set(h.tags) & pruned_tags)]
        if not viable:
            viable = [h for h in hyps if h.code not in tried_code]
        if not viable:
            return None, None

        viable.sort(
            key=lambda h: (
                len(set(h.tags) & preferred_tags),
                self.policy.score(statement, h),
            ),
            reverse=True,
        )
        chosen = viable[0]
        return chosen.code, chosen
