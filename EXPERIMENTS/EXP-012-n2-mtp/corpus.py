"""Fresh verified corpus builder for EXP-012 (N2).

The N2 corpus is deliberately **not** N0 and **not** EXP-011.  Every task in
this file was authored for N2 from the same nine frozen AST operators, with
fresh names, modules, data and test phrasing.  A task is admitted to the
corpus only after ``RepairEngine`` has produced a patch and the repaired
workspace has passed its own pytest suite.  Unsolved tasks are recorded and
dropped; they are never replaced by an oracle by hand.

Two disjoint splits are emitted:

* ``train`` — the frames the MTP heads are trained on;
* ``probe`` — a held-out same-family surface split used only for acceptance
  and greedy-exactness measurement.

The split is by variant index, fixed before training.  No probe frame is ever
written into ``train.bin``.

Run:
    python EXPERIMENTS/EXP-012-n2-mtp/corpus.py --out runs/corpus-v1
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omega_nano.brain.repair import RepairEngine  # noqa: E402

FAMILIES = (
    "mutable_default",
    "exception_scope",
    "cache_key",
    "closure_binding",
    "boundary_guard",
    "off_by_one",
    "integer_distribution",
    "mapping_fallback",
    "comparison_flip",
)

_TRAIN_THEMES = (
    ("ledger", "record_entry", "journal", "open", "close", "entry"),
    ("roster", "append_member", "members", "alice", "bob", "member"),
    ("inbox", "queue_message", "pending", "hello", "goodbye", "message"),
    ("cart", "add_line", "lines", "apple", "pear", "line"),
    ("garden", "plant_seed", "bed", "tulip", "rose", "seed"),
    ("library", "checkout_book", "loans", "dune", "emma", "book"),
    ("kitchen", "add_order", "tickets", "soup", "bread", "order"),
    ("studio", "schedule_take", "takes", "intro", "outro", "take"),
)

_PROBE_THEMES = (
    ("harbor", "dock_vessel", "berths", "tug", "ferry", "vessel"),
    ("orchard", "pack_crate", "crates", "plum", "fig", "crate"),
    ("atelier", "pin_swatch", "swatches", "linen", "wool", "swatch"),
    ("aviary", "tag_bird", "tags", "finch", "robin", "bird"),
)

_GEOMETRY_TRAIN = (
    ("geometry", "rectangle_area", "width", "height", 3, 4, 5),
    ("shapes", "tile_area", "length", "breadth", 2, 6, 7),
    ("surfaces", "panel_area", "across", "down", 4, 3, 8),
    ("frames", "window_area", "span", "rise", 5, 2, 6),
    ("plots", "field_area", "east", "north", 6, 3, 4),
    ("boards", "sheet_area", "rows", "cols", 7, 2, 5),
    ("panels", "face_area", "wide", "tall", 8, 4, 3),
    ("stages", "deck_area", "front", "back", 9, 2, 6),
)

_GEOMETRY_PROBE = (
    ("terrace", "patio_area", "near", "far", 10, 3, 7),
    ("hangar", "bay_area", "width_x", "depth_y", 11, 4, 5),
    ("pavilion", "floor_area", "side_a", "side_b", 12, 2, 6),
    ("gallery", "wall_area", "high", "low", 13, 5, 3),
)

_CLOSURE_TRAIN = (
    ("handlers", "make_handlers", "prefix", "names", "x", ("a", "b", "c")),
    ("routes", "make_routes", "scope", "parts", "api", ("v1", "v2", "v3")),
    ("formatters", "make_formatters", "tag", "labels", "fmt", ("red", "green", "blue")),
    ("validators", "make_validators", "realm", "rules", "chk", ("min", "max", "step")),
    ("renderers", "make_renderers", "skin", "views", "ui", ("list", "grid", "card")),
    ("filters", "make_filters", "channel", "stages", "flt", ("raw", "clean", "final")),
    ("dispatchers", "make_dispatchers", "route", "methods", "dsp", ("get", "post", "put")),
    ("wrappers", "make_wrappers", "kind", "layers", "wrp", ("base", "mid", "top")),
)

_CLOSURE_PROBE = (
    ("decorators", "make_decorators", "flavor", "steps", "dec", ("one", "two", "three")),
    ("adapters", "make_adapters", "format", "sources", "adp", ("csv", "json", "xml")),
    ("listeners", "make_listeners", "topic", "channels", "lst", ("a", "b", "c")),
    ("middleware", "make_middleware", "layer", "phases", "mid", ("pre", "core", "post")),
)

_BOUNDARY_TRAIN = (
    ("racks", "peek", "items", "index", (10, 20, 30)),
    ("stack", "top_at", "values", "slot", (4, 8, 16)),
    ("buffer", "read_at", "cells", "offset", (5, 6, 7)),
    ("window", "show_at", "rows", "position", (2, 4, 6)),
    ("track", "clip_at", "frames", "cursor", (11, 12, 13)),
    ("pages", "line_at", "lines", "number", (21, 22, 23)),
    ("slots", "entry_at", "slots", "which", (31, 32, 33)),
    ("grid", "cell_at", "grid", "spot", (41, 42, 43)),
)

_BOUNDARY_PROBE = (
    ("tray", "sample_at", "samples", "where", (51, 52, 53)),
    ("bins", "bucket_at", "bins", "pick", (61, 62, 63)),
    ("lanes", "runner_at", "lanes", "rank", (71, 72, 73)),
    ("shelves", "book_at", "shelves", "place", (81, 82, 83)),
)

_OFF_BY_ONE_TRAIN = (
    ("pagination", "window", "pages", 2),
    ("slicing", "take_prefix", "steps", 3),
    ("ranges", "head_slice", "blocks", 4),
    ("chunks", "lead_chunk", "groups", 5),
    ("segments", "front_segment", "units", 2),
    ("sections", "opening_section", "parts", 3),
    ("batches", "first_batch", "rounds", 4),
    ("strides", "start_stride", "moves", 5),
)

_OFF_BY_ONE_PROBE = (
    ("trims", "leading_trim", "cuts", 2),
    ("fences", "front_fence", "posts", 3),
    ("bands", "opening_band", "rings", 4),
    ("zones", "first_zone", "cells", 5),
)

_INTEGER_TRAIN = (
    ("budget", "split_budget", "total", "workers", 100, 3, 11, 4),
    ("portions", "share_portions", "amount", "people", 101, 3, 13, 4),
    ("rewards", "distribute_rewards", "points", "players", 102, 3, 17, 4),
    ("tasks", "assign_tasks", "jobs", "agents", 103, 3, 19, 4),
    ("coins", "divide_coins", "coins", "purses", 104, 3, 23, 4),
    ("seats", "allocate_seats", "seats", "tables", 105, 3, 29, 4),
    ("pages", "spread_pages", "pages", "binders", 106, 3, 31, 4),
    ("tokens", "split_tokens", "tokens", "buckets", 107, 3, 37, 4),
)

_INTEGER_PROBE = (
    ("funds", "split_funds", "funds", "accounts", 108, 3, 41, 4),
    ("credits", "share_credits", "credits", "teams", 109, 3, 43, 4),
    ("marbles", "divide_marbles", "marbles", "jars", 110, 3, 47, 4),
    ("cards", "deal_cards", "cards", "hands", 111, 3, 53, 4),
)

_MAPPING_TRAIN = (
    ("settings", "setting", "DEFAULTS", "theme", "light", "font"),
    ("colors", "color_for", "PALETTE", "accent", "blue", "border"),
    ("labels", "label_for", "LABELS", "title", "draft", "footer"),
    ("modes", "mode_for", "MODES", "fast", "turbo", "safe"),
    ("levels", "level_for", "LEVELS", "low", "quiet", "high"),
    ("formats", "format_for", "FORMATS", "text", "plain", "html"),
    ("routes", "route_for", "ROUTES", "home", "/", "/missing"),
    ("units", "unit_for", "UNITS", "meter", "m", "inch"),
)

_MAPPING_PROBE = (
    ("themes", "theme_for", "THEMES", "dark", "night", "sepia"),
    ("sizes", "size_for", "SIZES", "small", "tiny", "huge"),
    ("speeds", "speed_for", "SPEEDS", "slow", "crawl", "warp"),
    ("grades", "grade_for", "GRADES", "pass", "ok", "top"),
)

_COMPARISON_TRAIN = (
    ("access", "can_access", "level", "required", 2),
    ("gates", "may_enter", "rank", "minimum", 3),
    ("doors", "opens_at", "badge", "threshold", 4),
    ("locks", "unlocks_with", "key", "limit", 5),
    ("permits", "permits_level", "grade", "floor", 6),
    ("tickets", "admits_ticket", "tier", "cutoff", 7),
    ("passes", "passes_bar", "score", "bar", 8),
    ("lanes", "clears_lane", "speed", "cap", 9),
)

_COMPARISON_PROBE = (
    ("clearance", "clears_clearance", "clearance", "gate", 10),
    ("badges", "badge_opens", "badge", "lock", 11),
    ("ranks", "rank_meets", "rank", "need", 12),
    ("quotas", "quota_meets", "quota", "target", 13),
)


def _theme_blocks() -> dict[str, tuple[tuple[Any, ...], ...]]:
    return {
        "mutable_default": _TRAIN_THEMES + _PROBE_THEMES,
        "exception_scope": _TRAIN_THEMES + _PROBE_THEMES,
        "cache_key": _GEOMETRY_TRAIN + _GEOMETRY_PROBE,
        "closure_binding": _CLOSURE_TRAIN + _CLOSURE_PROBE,
        "boundary_guard": _BOUNDARY_TRAIN + _BOUNDARY_PROBE,
        "off_by_one": _OFF_BY_ONE_TRAIN + _OFF_BY_ONE_PROBE,
        "integer_distribution": _INTEGER_TRAIN + _INTEGER_PROBE,
        "mapping_fallback": _MAPPING_TRAIN + _MAPPING_PROBE,
        "comparison_flip": _COMPARISON_TRAIN + _COMPARISON_PROBE,
    }


def _split_count(family: str) -> tuple[int, int]:
    blocks = _theme_blocks()[family]
    train = len(blocks) - 4
    return train, 4


def make_task(family: str, variant: int) -> dict[str, Any]:
    """Return a fresh buggy workspace + spec for ``family`` and ``variant``."""
    blocks = _theme_blocks()[family]
    train_n, _ = _split_count(family)
    split = "train" if variant < train_n else "probe"
    v = blocks[variant]
    task_id = f"n2-{family.replace('_', '-')}-{variant:02d}"

    if family in {"mutable_default"}:
        module, func, bucket, a, b, noun = v
        files = {
            f"{module}.py": Template(
                '"""Helpers for the $module service."""\n\n'
                "def $func(value, $bucket=[]):\n"
                "    $bucket.append(value)\n"
                "    return $bucket\n\n"
                "def count_$noun(items):\n"
                "    return len(items)\n"
            ).substitute(module=module, func=func, bucket=bucket, noun=noun),
            "tests/test_service.py": Template(
                "from $module import $func\n\n\n"
                "def test_each_call_gets_its_own_$bucket():\n"
                "    first = $func('$a')\n"
                "    second = $func('$b')\n"
                "    assert first == ['$a']\n"
                "    assert second == ['$b']\n"
                "    assert first is not second\n"
            ).substitute(module=module, func=func, bucket=bucket, a=a, b=b),
        }
        spec = (
            f"The {module} service appends a {noun} to a collection. Each call must "
            f"start from an empty collection; the current default is shared between calls."
        )

    elif family == "exception_scope":
        module, func, bucket, a, b, noun = v
        files = {
            "errors.py": "class PolicyError(Exception):\n    pass\n",
            f"{module}.py": Template(
                "from errors import PolicyError\n\n\n"
                "def $func(handler, value):\n"
                "    try:\n"
                "        return handler(value)\n"
                "    except Exception:\n"
                "        return -1\n"
            ).substitute(func=func),
            "tests/test_service.py": Template(
                "import pytest\n"
                "from errors import PolicyError\n"
                "from $module import $func\n\n\n"
                "def test_policy_error_falls_back():\n"
                "    def handler(value):\n"
                "        raise PolicyError('blocked')\n"
                "    assert $func(handler, 1) == -1\n\n\n"
                "def test_type_error_propagates():\n"
                "    def handler(value):\n"
                "        raise TypeError('bad')\n"
                "    with pytest.raises(TypeError):\n"
                "        $func(handler, 1)\n"
            ).substitute(module=module, func=func),
        }
        spec = (
            f"The {module} helper may convert a policy failure into -1, but a programming "
            f"error such as TypeError must never be swallowed."
        )

    elif family == "cache_key":
        module, func, k1, k2, a, b, c = v
        files = {
            f"{module}.py": Template(
                "_CACHE = {}\n\n\n"
                "def $func($k1, $k2):\n"
                "    key = $k1\n"
                "    if key not in _CACHE:\n"
                "        _CACHE[key] = $k1 * $k2\n"
                "    return _CACHE[key]\n"
            ).substitute(func=func, k1=k1, k2=k2),
            "tests/test_service.py": Template(
                "from $module import $func\n\n\n"
                "def test_result_depends_on_both_inputs():\n"
                "    assert $func($a, $b) == $a * $b\n"
                "    assert $func($a, $c) == $a * $c\n"
            ).substitute(module=module, func=func, a=a, b=b, c=c),
        }
        spec = (
            f"The {module} cache key must include both {k1} and {k2}; omitting one "
            f"returns a stale result when only that input changes."
        )

    elif family == "closure_binding":
        module, func, fixed, arg, a, names = v
        joined = ", ".join(repr(n) for n in names)
        expected = ", ".join(f"'{a}:{n}:1'" for n in names)
        files = {
            f"{module}.py": Template(
                "def $func($fixed, $arg):\n"
                "    handlers = []\n"
                "    for name in $arg:\n"
                "        handlers.append(lambda value: f'{$fixed}:{name}:{value}')\n"
                "    return handlers\n"
            ).substitute(func=func, fixed=fixed, arg=arg),
            "tests/test_service.py": Template(
                "from $module import $func\n\n\n"
                "def test_each_closure_binds_its_own_name():\n"
                "    handlers = $func('$a', [$names])\n"
                "    assert [h(1) for h in handlers] == [$expected]\n"
            ).substitute(module=module, func=func, a=a, names=joined, expected=expected),
        }
        spec = (
            f"The {module} factory must bind the loop variable into each closure; "
            f"today every closure sees the final value."
        )

    elif family == "boundary_guard":
        module, func, arg, idx, values = v
        vals = ", ".join(str(x) for x in values)
        files = {
            f"{module}.py": Template(
                "def $func($arg, $idx):\n"
                "    if $idx < len($arg):\n"
                "        return $arg[$idx]\n"
                "    return None\n"
            ).substitute(func=func, arg=arg, idx=idx),
            "tests/test_service.py": Template(
                "import pytest\n"
                "from $module import $func\n\n\n"
                "def test_valid_positions():\n"
                "    assert $func([$vals], 0) == $values\n"
                "    assert $func([$vals], 2) == $values3\n\n\n"
                "def test_negative_rejected():\n"
                "    with pytest.raises(IndexError):\n"
                "        $func([$vals], -1)\n\n\n"
                "def test_overflow_rejected():\n"
                "    with pytest.raises(IndexError):\n"
                "        $func([$vals], 3)\n"
            ).substitute(module=module, func=func, vals=vals, values=values[0], values3=values[2]),
        }
        spec = (
            f"The {module} accessor accepts only positions 0..2 and must raise IndexError "
            f"for negative or overflowing positions instead of returning a silent None."
        )

    elif family == "off_by_one":
        module, func, arg, factor = v
        files = {
            f"{module}.py": Template(
                "def $func(data, $arg):\n"
                "    size = $arg * $factor\n"
                "    return data[:size]\n"
            ).substitute(func=func, arg=arg, factor=factor),
            "tests/test_service.py": Template(
                "from $module import $func\n\n\n"
                "def test_window_includes_the_extra_slot():\n"
                "    assert $func('abcdef', 1) == 'abcd'\n"
                "    assert $func('abcdef', 2) == 'abcdef'\n"
            ).substitute(module=module, func=func),
        }
        spec = (
            f"The {module} window must cover (n + 1) * {factor} bytes; the current "
            f"multiplication omits the extra slot at every size."
        )

    elif family == "integer_distribution":
        module, func, total, parts, t1, p1, t2, p2 = v
        q1, r1 = divmod(t1, p1)
        q2, r2 = divmod(t2, p2)
        first = ", ".join(str(q1 + 1 if i < r1 else q1) for i in range(p1))
        second = ", ".join(str(q2 + 1 if i < r2 else q2) for i in range(p2))
        files = {
            f"{module}.py": Template(
                "def $func($total, $parts):\n"
                "    share = $total // $parts\n"
                "    return [share] * $parts\n"
            ).substitute(func=func, total=total, parts=parts),
            "tests/test_service.py": Template(
                "from $module import $func\n\n\n"
                "def test_remainder_is_distributed():\n"
                "    assert $func($t1, $p1) == [$first]\n"
                "    assert $func($t2, $p2) == [$second]\n"
            ).substitute(module=module, func=func, t1=t1, p1=p1, first=first, t2=t2, p2=p2, second=second),
        }
        spec = (
            f"The {module} splitter must distribute the integer remainder one unit at a "
            f"time; the current answer drops it, so the parts do not sum to the whole."
        )

    elif family == "mapping_fallback":
        module, func, table, key, default, missing = v
        files = {
            f"{module}.py": Template(
                "$table = {'$key': '$default'}\n\n\n"
                "def $func(name):\n"
                "    return $table[name]\n"
            ).substitute(table=table, key=key, default=default, func=func),
            "tests/test_service.py": Template(
                "from $module import $func\n\n\n"
                "def test_known_key():\n"
                "    assert $func('$key') == '$default'\n\n\n"
                "def test_missing_key_falls_back():\n"
                "    assert $func('$missing') == '$default'\n"
            ).substitute(module=module, func=func, key=key, default=default, missing=missing),
        }
        spec = (
            f"The {module} lookup must fall back to the declared default for an unknown "
            f"key instead of raising KeyError."
        )

    elif family == "comparison_flip":
        module, func, left, right, bound = v
        files = {
            f"{module}.py": Template(
                "def $func($left, $right):\n"
                "    if $left <= $right:\n"
                "        return True\n"
                "    return False\n"
            ).substitute(func=func, left=left, right=right),
            "tests/test_service.py": Template(
                "from $module import $func\n\n\n"
                "def test_equal_is_denied():\n"
                "    assert $func($bound, $bound) is False\n\n\n"
                "def test_lower_is_allowed():\n"
                "    assert $func($lo, $bound) is True\n\n\n"
                "def test_higher_is_denied():\n"
                "    assert $func($hi, $bound) is False\n"
            ).substitute(module=module, func=func, bound=bound, lo=bound - 1, hi=bound + 1),
        }
        spec = (
            f"The {module} gate must allow only values strictly below {right}; equality "
            f"is not enough and the current comparison admits it."
        )

    else:  # pragma: no cover - guarded by FAMILIES
        raise ValueError(family)

    return {
        "id": task_id,
        "family": family,
        "variant": variant,
        "split": split,
        "files": files,
        "spec": spec,
    }


def write_workspace(root: Path, task: dict[str, Any]) -> None:
    for rel, text in task["files"].items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    (root / "task.json").write_text(json.dumps({
        "id": task["id"], "family": task["family"], "variant": task["variant"],
        "split": task["split"], "spec": task["spec"],
    }, indent=2) + "\n", encoding="utf-8")


def _repo_files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.glob("*.py")):
        out[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return out


def _test_files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted((root / "tests").rglob("*.py")):
        out[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return out


def _trace_from_report(report: Any) -> str:
    lines = [
        f"pytest: {report.initial_failed} failed before repair; "
        f"{report.verifier_calls} verifier call(s); {report.final_failed} failed after repair"
    ]
    for attempt in report.attempts[:3]:
        lines.append(f"candidate: {attempt.get('operator', '?')} -> {attempt.get('summary', '')}")
    for diff in report.diffs[:2]:
        for line in diff.splitlines():
            if line.startswith(("+++", "---", "@@", "+", "-")) and not line.startswith(("+++", "---")):
                lines.append(line)
    return "\n".join(lines)


def build_frame(task: dict[str, Any], repo: dict[str, str], tests: dict[str, str], trace: str) -> bytes:
    test_text = "\n".join(
        f"# {rel}\n{text.rstrip()}\n" for rel, text in tests.items()
    )
    sol_text = "\n".join(
        f"# {rel}\n{text.rstrip()}\n" for rel, text in repo.items()
    )
    return (
        f"<SPEC>\n{task['spec'].strip()}\n<TEST>\n{test_text.strip()}\n"
        f"<FAIL>\n{trace.strip()}\n<SOL>\n{sol_text.strip()}\n</SOL>\n"
    ).encode("utf-8")


def repair_task(task: dict[str, Any], max_calls: int = 24) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_workspace(root, task)
        report = RepairEngine(max_verifier_calls=max_calls).repair(root)
        repo = _repo_files(root)
        tests = _test_files(root)
        trace = _trace_from_report(report)
        if report.solved and report.accepted_without_execution == 0:
            frame = build_frame(task, repo, tests, trace)
            return {
                "id": task["id"], "family": task["family"], "variant": task["variant"],
                "split": task["split"], "solved": True,
                "verifier_calls": report.verifier_calls,
                "initial_failed": report.initial_failed,
                "edits": report.edits,
                "frame": frame,
                "elapsed_s": round(time.perf_counter() - started, 3),
            }
        return {
            "id": task["id"], "family": task["family"], "variant": task["variant"],
            "split": task["split"], "solved": False,
            "verifier_calls": report.verifier_calls,
            "initial_failed": report.initial_failed,
            "final_failed": report.final_failed,
            "attempts": report.attempts[:4],
            "elapsed_s": round(time.perf_counter() - started, 3),
        }


def _sealed_shingles() -> set[bytes]:
    """64-byte shingles from every sealed N0/EXP-011 source file."""
    sealed: set[bytes] = set()
    roots = (
        REPO_ROOT / "EXPERIMENTS" / "EXP-009-n0-benchmark" / "task_suite",
        REPO_ROOT / "EXPERIMENTS" / "EXP-011-n1-heldout" / "task_suite",
    )
    width = 64
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            for i in range(0, max(0, len(data) - width + 1)):
                sealed.add(data[i:i + width])
    return sealed


def leak_check(blobs: dict[str, bytes]) -> dict[str, Any]:
    sealed = _sealed_shingles()
    hits: list[dict[str, Any]] = []
    width = 64
    for name, data in blobs.items():
        for i in range(0, max(0, len(data) - width + 1)):
            if data[i:i + width] in sealed:
                hits.append({"stream": name, "offset": i, "sample": data[i:i + width].decode("utf-8", "replace")})
                break
    return {"sealed_shingles": len(sealed), "hits": hits, "clean": not hits}


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    """Read the append-only per-task log, tolerating a torn final line."""
    done: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "id" in rec:
            done[str(rec["id"])] = rec
    return done


def build(
    out_dir: Path,
    max_calls: int = 24,
    *,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build the corpus incrementally.

    Every task is persisted the moment it finishes:

    * ``frames/<task-id>.bin`` — the verified frame, or absent when unsolved;
    * ``records.jsonl`` — one JSON record per task, append-only.

    Re-running with the same ``--out`` resumes: finished tasks are skipped and
    only missing/failed ones are retried.  ``force=True`` wipes both first.
    The final ``train.bin``/``probe.bin`` are assembled deterministically from
    the cached frames in family/variant order, so resuming cannot change the
    byte layout of a completed corpus.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    records_path = out_dir / "records.jsonl"
    if force:
        shutil.rmtree(frames_dir, ignore_errors=True)
        records_path.unlink(missing_ok=True)
        for stale in ("train.bin", "probe.bin", "manifest.json"):
            (out_dir / stale).unlink(missing_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    done = _load_records(records_path)
    plan: list[dict[str, Any]] = []
    for family in FAMILIES:
        train_n, probe_n = _split_count(family)
        for variant in range(train_n + probe_n):
            plan.append(make_task(family, variant))
    total = len(plan)
    started = time.perf_counter()
    resumed = sum(1 for t in plan if t["id"] in done)

    if verbose:
        print(f"corpus build: {total} tasks, {resumed} already recorded, "
              f"max_calls={max_calls}, out={out_dir}", flush=True)

    with records_path.open("a", encoding="utf-8") as log:
        for idx, task in enumerate(plan, start=1):
            tid = task["id"]
            frame_path = frames_dir / f"{tid}.bin"
            prev = done.get(tid)
            if prev is not None and (not prev.get("solved") or frame_path.is_file()):
                if verbose:
                    print(f"[{idx:3d}/{total}] {tid:36s} cached "
                          f"(solved={prev.get('solved')}, calls={prev.get('verifier_calls')})",
                          flush=True)
                continue

            t0 = time.perf_counter()
            rec = repair_task(task, max_calls=max_calls)
            frame = rec.pop("frame", None)
            if rec.get("solved") and frame is not None:
                frame_path.write_bytes(frame)
            else:
                frame_path.unlink(missing_ok=True)
            rec["cached_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            done[tid] = rec
            if verbose:
                elapsed = time.perf_counter() - started
                rate = idx / max(1e-9, elapsed)
                eta = (total - idx) / max(1e-9, rate)
                extra = "" if rec.get("solved") else f" final_failed={rec.get('final_failed')}"
                print(f"[{idx:3d}/{total}] {tid:36s} "
                      f"solved={str(bool(rec.get('solved'))):5s} "
                      f"calls={rec.get('verifier_calls'):2d} "
                      f"{time.perf_counter() - t0:6.2f}s{extra}  "
                      f"(elapsed {elapsed:6.1f}s, eta {eta:6.0f}s)",
                      flush=True)

    # Deterministic assembly from cached frames (never from `done` insertion order).
    streams: dict[str, list[bytes]] = {"train": [], "probe": []}
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in plan:
        rec = done.get(task["id"])
        if rec is None:
            missing.append(task["id"])
            continue
        records.append(rec)
        if rec.get("solved"):
            frame_path = frames_dir / f"{task['id']}.bin"
            if frame_path.is_file():
                streams[str(rec["split"])].append(frame_path.read_bytes())
            else:
                missing.append(task["id"])
    if missing:
        raise RuntimeError(
            f"{len(missing)} task(s) missing a frame after the build: {missing[:8]}"
        )

    blobs = {f"{split}.bin": b"\n".join(frames) for split, frames in streams.items()}
    for name, data in blobs.items():
        (out_dir / name).write_bytes(data)
    leak = leak_check(blobs)
    solved = sum(1 for r in records if r["solved"])
    by_family: dict[str, dict[str, int]] = {}
    for fam in FAMILIES:
        rows = [r for r in records if r["family"] == fam]
        by_family[fam] = {
            "total": len(rows),
            "solved": sum(1 for r in rows if r["solved"]),
            "train": sum(1 for r in rows if r["solved"] and r["split"] == "train"),
            "probe": sum(1 for r in rows if r["solved"] and r["split"] == "probe"),
        }
    manifest = {
        "exp_id": "EXP-012",
        "title": "N2 fresh verified corpus",
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "operators": list(FAMILIES),
        "split_rule": "variant index < train_count is train; the last four variants per family are probe",
        "max_verifier_calls": max_calls,
        "tasks": records,
        "by_family": by_family,
        "solved": solved,
        "total": len(records),
        "streams": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                   "frames": sum(1 for _ in data.split(b"<SPEC>")) - 1}
            for name, data in blobs.items()
        },
        "sealed_leak_check": leak,
        "wall_s": round(time.perf_counter() - started, 3),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the EXP-012 N2 verified corpus")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "corpus-v1")
    ap.add_argument("--max-calls", type=int, default=24)
    ap.add_argument("--force", action="store_true", help="discard cached records/frames and rebuild")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    result = build(args.out, max_calls=args.max_calls, force=args.force, verbose=not args.quiet)
    print(json.dumps({
        "solved": result["solved"], "total": result["total"],
        "streams": result["streams"], "by_family": result["by_family"],
        "sealed_leak_check": result["sealed_leak_check"], "wall_s": result["wall_s"],
    }, indent=2))
