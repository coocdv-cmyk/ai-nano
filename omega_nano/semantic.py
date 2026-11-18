"""Bilingual Semantic Concept Encoder, Behavioral Guard (EXP-L L3), and Drift Monitor (EXP-L L1).

Upgrades MemoryFabric beyond raw surface hashing without external downloads:
  * Code identifier splitting (camelCase / snake_case / dot.path)
  * Lightweight Arabic & English morphological stemming
  * Bilingual (Arabic <-> English <-> Code) canonical concept anchors
  * EXP-L L1 mapping-consistency + CUSUM semantic drift monitor
  * EXP-L L3 behavioral retrieval verification gate (prevents silent substitution)
"""

from __future__ import annotations

import math
import re
import zlib
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence

DIM = 512

_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_AR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_AR_CHAR_RE = re.compile(r"[\u0600-\u06FF]")

_RAW_STOP = (
    "the a an of to in on for and or is are was were be been it this that with as at by from what who "
    "do you know about tell me my your our where when which how why whom whose many much does did "
    "kind sort type there their they them can could would should will has have had into out "
    "في من على الى عن ما ماذا هل هو هي ان او كان تعرف لي اين متى كيف كم لماذا اي "
    "هذا هذه ذلك التي الذي فيه فيها عليه منها عند يحتاج تحتاج محتاج محتاجه"
).split()


def normalize_text(text: str) -> str:
    text = _AR_DIACRITICS.sub("", text).translate(_AR_MAP).casefold()
    return " ".join(_WORD_RE.findall(text))


STOP_WORDS = {normalize_text(w) for w in _RAW_STOP}

# Curated bilingual + CS canonical concept groups (all normalized automatically).
_CONCEPT_CLUSTERS: dict[str, tuple[str, ...]] = {
    "vehicle": ("car", "cars", "vehicle", "vehicles", "automobile", "auto", "truck", "سياره", "سيارات", "مركبه", "مركبات", "عربيه", "شاحنه"),
    "engine": ("engine", "motor", "engines", "محرك", "محركات", "موتور"),
    "repair": ("repair", "fix", "fixed", "fixing", "maintenance", "troubleshoot", "debug", "patch", "صيانه", "تصليح", "اصلاح", "تعديل"),
    "cache_ram": ("cache", "cached", "caching", "redis", "memcached", "ram", "ephemeral", "temporary", "in-memory", "كاش", "مؤقت", "مؤقته", "رام", "ذاكره"),
    "memory_store": ("memory", "store", "stored", "stores", "storage", "persist", "persistent", "remember", "recall", "backup", "snapshot", "ذاكره", "تخزين", "مخزن", "مخزنه", "محفوظ", "محفوظه", "حفظ", "استرجاع", "احتياطي", "احتياطيه"),
    "database": ("database", "db", "data", "sql", "sqlite", "postgres", "mysql", "table", "query", "قاعده", "قواعد", "بيانات", "داتا", "معلومات", "جدول"),
    "secret_auth": ("password", "secret", "secrets", "token", "tokens", "key", "keys", "credential", "credentials", "auth", "login", "كلمه", "سر", "سري", "سريه", "مرور", "رمز", "مفتاح", "دخول", "مصادقه"),
    "file_fs": ("file", "files", "document", "documents", "doc", "path", "folder", "directory", "disk", "env", "ملف", "ملفات", "مستند", "مجلد", "مسار", "قرص", "بيئه"),
    "server_net": ("server", "servers", "host", "port", "ports", "endpoint", "api", "url", "http", "network", "cluster", "خادم", "سيرفر", "منفذ", "بورت", "شبكه", "عنقود"),
    "code_prog": ("code", "function", "functions", "method", "class", "script", "program", "programming", "python", "algorithm", "داله", "دوال", "كود", "برمجه", "برنامج", "بايثون", "خوارزميه"),
    "sort_order": ("sort", "sorted", "sorting", "order", "ordered", "arrange", "rank", "ترتيب", "فرز", "مرتب", "تصاعدي"),
    "search_find": ("search", "find", "lookup", "retrieve", "retrieval", "match", "filter", "بحث", "ايجاد", "استعلام", "تطابق"),
    "graph_net": ("graph", "node", "nodes", "edge", "edges", "vertex", "vertices", "tree", "cycle", "dag", "topo", "topological", "رسم", "عقده", "حافه", "شجره", "دوره"),
    "interval_span": ("interval", "intervals", "range", "ranges", "overlap", "overlapping", "merge", "merging", "span", "فتره", "فترات", "نطاق", "تداخل", "دمج"),
    "stack_queue": ("stack", "queue", "lifo", "fifo", "deque", "push", "pop", "rpn", "postfix", "bracket", "parentheses", "مكدس", "طابور", "اقواس"),
    "math_calc": ("math", "calculate", "compute", "equation", "solve", "formula", "number", "sum", "total", "prime", "حساب", "احسب", "معادله", "حل", "رياضيات", "رقم", "مجموع", "اولي"),
    "error_fail": ("error", "errors", "bug", "bugs", "fail", "failure", "breakdown", "exception", "crash", "traceback", "خطا", "اخطاء", "عطل", "فشل", "انهيار", "مشكله"),
    "fast_perf": ("fast", "faster", "speed", "latency", "throughput", "performance", "quick", "efficient", "efficiency", "سريع", "اسرع", "سرعه", "اداء", "كفاءه"),
    "compress": ("compress", "compressed", "compression", "zip", "zlib", "lzma", "compact", "tiny", "ضغط", "مضغوط", "صغير", "مصغر"),
    "org_project": ("company", "organization", "org", "enterprise", "work", "office", "project", "system", "شركه", "مؤسسه", "عمل", "شغل", "مكتب", "مشروع", "نظام"),
    "ai_model": ("model", "neural", "core", "brain", "ai", "intelligence", "agent", "weights", "train", "نموذج", "موديل", "نواه", "عقل", "ذكاء", "وكيل", "اوزان", "تدريب"),
    "test_verify": ("test", "tests", "pytest", "verify", "verification", "check", "assert", "proof", "prove", "اختبار", "اختبارات", "تحقق", "برهان", "اثبات"),
    "text_lang": ("text", "string", "word", "words", "utf8", "chunk", "split", "arabic", "english", "language", "نص", "كلمه", "كلمات", "تقطيع", "عربي", "عربيه", "انجليزي", "لغه"),
    "hot_temp": ("boil", "boils", "boiling", "heat", "hot", "temperature", "steam", "water", "غليان", "يغلي", "حراره", "ساخن", "ماء"),
    "location": ("where", "location", "city", "capital", "headquartered", "place", "اين", "موقع", "مدينه", "عاصمه", "مقر", "مكان"),
}

_TERM_TO_CONCEPTS: dict[str, list[str]] = {}
for _cname, _words in _CONCEPT_CLUSTERS.items():
    for _w in _words:
        _nw = normalize_text(_w)
        if _nw:
            _TERM_TO_CONCEPTS.setdefault(_nw, []).append(_cname)


def _arabic_stem(tok: str) -> str:
    s = tok
    if len(s) >= 5 and s[:2] in ("ال", "ول", "بل", "كل", "فل"):
        s = s[2:]
    elif len(s) >= 4 and s[0] in ("و", "ف", "ب", "ك", "ل") and not s.startswith("لا"):
        s = s[1:]
    if len(s) >= 4 and s.startswith("ال"):
        s = s[2:]
    for suf in ("ات", "ون", "ين", "يه", "ها", "هم", "هن", "كم", "نا"):
        if len(s) >= 5 and s.endswith(suf):
            s = s[: -len(suf)]
            break
    if len(s) >= 4 and s.endswith(("ه", "ي")):
        s = s[:-1]
    return s


def _english_stem(tok: str) -> str:
    s = tok
    for suf in ("ations", "ation", "ments", "ment", "ness", "ing", "ies", "ied", "ers", "es", "ed", "ly", "er", "s"):
        if len(s) - len(suf) >= 3 and s.endswith(suf):
            if suf == "ies":
                return s[:-3] + "y"
            return s[: -len(suf)]
    return s


def _split_identifiers(raw_text: str) -> list[str]:
    """Extract sub-tokens from camelCase and snake_case identifiers while preserving order."""
    expanded = _CAMEL_RE.sub(" ", raw_text).replace("_", " ").replace(".", " ").replace("/", " ")
    norm = normalize_text(expanded)
    return [t for t in norm.split() if t]


def _feat(name: str) -> tuple[int, float]:
    x = zlib.crc32(name.encode("utf-8"))
    return x % DIM, (1.0 if (x >> 31) & 1 else -1.0)


def encode_text(text: str, dim: int = DIM) -> list[float]:
    """Deterministic hybrid surface + morphological + cross-lingual concept embedding."""
    norm = normalize_text(text)
    raw_toks = norm.split()
    content = [t for t in raw_toks if t not in STOP_WORDS] or raw_toks
    sub_toks = [t for t in _split_identifiers(text) if t not in STOP_WORDS]

    feats: dict[str, float] = {}

    def add(f: str, w: float) -> None:
        feats[f] = feats.get(f, 0.0) + w

    for t in content:
        add("w:" + t, 1.0)
        if _AR_CHAR_RE.search(t):
            st = _arabic_stem(t)
            if st != t:
                add("st:" + st, 0.75)
            for cname in _TERM_TO_CONCEPTS.get(t, ()) or _TERM_TO_CONCEPTS.get(st, ()):
                add("c_anchor:" + cname, 1.65)
        else:
            st = _english_stem(t)
            if st != t:
                add("st:" + st, 0.65)
            for cname in _TERM_TO_CONCEPTS.get(t, ()) or _TERM_TO_CONCEPTS.get(st, ()):
                add("c_anchor:" + cname, 1.65)

        padded = f"#{t}#"
        for i in range(len(padded) - 2):
            add("c:" + padded[i : i + 3], 0.25)

    for st in sub_toks:
        if st not in content:
            add("w:" + st, 0.75)
            est = _arabic_stem(st) if _AR_CHAR_RE.search(st) else _english_stem(st)
            if est != st:
                add("st:" + est, 0.55)
            for cname in _TERM_TO_CONCEPTS.get(st, ()) or _TERM_TO_CONCEPTS.get(est, ()):
                add("c_anchor:" + cname, 0.85)

    for a, b in zip(content, content[1:]):
        add(f"b:{a}_{b}", 0.7)

    vec = [0.0] * dim
    for f, w in feats.items():
        i, sign = _feat(f)
        if dim != DIM:
            i = zlib.crc32(f.encode("utf-8")) % dim
        vec[i] += sign * (1.0 + math.log(w)) if w >= 1.0 else sign * w
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]


# ---------------------------------------------------------------------------
# EXP-L L1: Mapping-Consistency Semantic Drift Monitor
# ---------------------------------------------------------------------------

@dataclass
class DriftStatus:
    drifted: bool
    recent_pass_rate: float
    baseline_pass_rate: float
    cusum_score: float
    reason: str


class SemanticDriftMonitor:
    """Tracks skill/verifier mapping consistency over a sliding window (EXP-L L1 insight).

    Feature marginals can stay identical while underlying task semantics shift; watching
    verification consistency + CUSUM detects silent semantic drift with low false alarms.
    """

    def __init__(self, window: int = 12, baseline_rate: float = 0.9, threshold: float = 1.6):
        self.window = max(3, window)
        self.baseline = baseline_rate
        self.threshold = threshold
        self._outcomes: deque[float] = deque(maxlen=self.window)
        self._cusum = 0.0

    def observe(self, passed: bool) -> DriftStatus:
        val = 1.0 if passed else 0.0
        self._outcomes.append(val)
        # Penalize unexpected failures relative to baseline consistency
        deficit = max(0.0, self.baseline - val - 0.15)
        if passed:
            self._cusum = max(0.0, self._cusum - 0.2)
        else:
            self._cusum += deficit
        rate = sum(self._outcomes) / len(self._outcomes)
        drifted = len(self._outcomes) >= 3 and (self._cusum >= self.threshold or rate < 0.5)
        reason = (
            f"consistency dropped ({rate:.2f} < {self.baseline:.2f}, cusum={self._cusum:.2f})"
            if drifted else "stable"
        )
        return DriftStatus(drifted, round(rate, 4), self.baseline, round(self._cusum, 4), reason)


# ---------------------------------------------------------------------------
# EXP-L L3: Behavioral Verification Guard (Anti-Silent-Substitution)
# ---------------------------------------------------------------------------

def verify_behavioral_candidates(
    candidates: Sequence[tuple[str, Callable[[], str]]],
    verifier: Callable[[str], bool],
) -> tuple[str | None, str | None, str]:
    """Evaluate ranked candidates behaviorally before accepting any retrieved match.

    Returns (chosen_name, solution, status) where status in
    {"VERIFIED", "AMBIGUOUS", "REJECTED_ALL"}.
    """
    verified: list[tuple[str, str]] = []
    for name, gen_fn in candidates:
        cand = gen_fn()
        if cand and verifier(cand):
            verified.append((name, cand))
    if len(verified) == 1:
        return verified[0][0], verified[0][1], "VERIFIED"
    if len(verified) > 1:
        # Both pass; still deterministic top-1, but expose ambiguity flag cleanly.
        return verified[0][0], verified[0][1], "AMBIGUOUS"
    return None, None, "REJECTED_ALL"
