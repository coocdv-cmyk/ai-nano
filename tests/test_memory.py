import pytest

import omega_nano.memory as memmod
from omega_nano.memory import MemoryFabric, normalize


@pytest.fixture
def mem(tmp_path):
    m = MemoryFabric(tmp_path / "m.db")
    yield m
    m.close()


def test_arabic_normalization():
    assert normalize("أَحْمَدُ في المدرسةِ") == normalize("احمد فى المدرسه".replace("فى", "في"))
    assert normalize("إسلام آمنة") == "اسلام امنه"
    assert normalize("١٢٣ Hello") == "123 hello"


def test_recall_english_and_arabic(mem):
    mem.remember("The deployment server runs on port 8443", source="ops-doc")
    mem.remember("عاصمة مصر هي القاهرة", source="atlas")
    mem.remember("Bananas are rich in potassium")
    assert mem.recall("which port does deployment use")[0].text.startswith("The deployment")
    top = mem.recall("ما هي عاصمة مِصر")[0]
    assert top.text == "عاصمة مصر هي القاهرة" and top.source == "atlas"


def test_exact_and_near_duplicates(mem):
    a = mem.remember("Paris is the capital of France")
    assert mem.remember("paris is the capital of france!")["status"] == "duplicate"
    assert mem.remember("Paris is the capital of France.")["id"] == a["id"]
    assert mem.stats()["records"] == 1


def test_supersession_keeps_history(mem):
    mem.remember("CEO is Alice", key="CEO")
    r = mem.remember("CEO is Bob", key="ceo")
    assert r["superseded"]
    hits = mem.recall("who is the CEO")
    assert hits[0].text == "CEO is Bob" and all(h.text != "CEO is Alice" for h in hits)
    hist = mem.history("CEO")
    assert [h["text"] for h in hist] == ["CEO is Alice", "CEO is Bob"] and hist[0]["valid_to"] is not None
    assert any(h.text == "CEO is Alice" for h in mem.recall("CEO", include_history=True))


def test_forget_is_a_hard_delete(mem):
    r = mem.remember("my secret code word is pineapple")
    assert mem.forget(r["id"])
    assert mem.recall("secret code word") == []
    assert mem.conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0] == 0


def test_persists_across_reopen(tmp_path):
    m = MemoryFabric(tmp_path / "m.db")
    m.remember("the build uses CMake 3.29")
    m.close()
    m2 = MemoryFabric(tmp_path / "m.db")
    assert m2.recall("cmake version")[0].text == "the build uses CMake 3.29"
    m2.close()


def test_kind_filter_and_validation(mem):
    mem.remember("ran the backup job", kind="episodic")
    mem.remember("backups run nightly", kind="procedural")
    assert [h.kind for h in mem.recall("backup", kind="procedural")] == ["procedural"]
    assert all(h.kind != "episodic" for h in mem.recall("backup", exclude=("episodic",)))
    with pytest.raises(ValueError):
        mem.remember("x", kind="nonsense")
    with pytest.raises(ValueError):
        mem.remember("   ")


def test_pure_python_fallback_matches_numpy(tmp_path, monkeypatch):
    facts = ["alpha protocol uses TLS", "beta service stores logs", "gamma job compresses images"]
    m1 = MemoryFabric(tmp_path / "a.db")
    for f in facts:
        m1.remember(f)
    want = [h.id for h in m1.recall("which service stores logs")]
    m1.close()
    monkeypatch.setattr(memmod, "np", None)
    m2 = MemoryFabric(tmp_path / "a.db")
    assert [h.id for h in m2.recall("which service stores logs")] == want
    m2.close()


def test_provenance_fields_present(mem):
    mem.remember("Water boils at 100 C at sea level", source="textbook", confidence=0.95)
    h = mem.recall("boiling point of water")[0]
    assert h.source == "textbook" and h.confidence == 0.95 and h.valid_to is None and h.valid_from > 0


def test_bulk_ingest_and_incremental_index(mem):
    r = mem.remember_many([{"text": f"record number {i} about topic {i % 7}"} for i in range(3000)]
                          + [{"text": "record number 5 about topic 5"}])
    assert r == {"stored": 3000, "skipped_duplicates": 1}
    mem.remember("a brand new fact about zebras")  # appended without a full reload
    assert not mem._dirty and len(mem._ids) == 3001
    assert mem.recall("zebras")[0].text == "a brand new fact about zebras"
    assert mem.recall("record number 1234")[0].text == "record number 1234 about topic 2"


def test_typo_and_inflection_tolerance(mem):
    mem.remember_many([{"text": f"Entity{i}x{i * 7919 % 1000} is managed by someone"} for i in range(200)])
    mem.remember("Kalvorinsa Brightwell is headquartered in Dorran")
    assert mem.recall("where is Kalvrinsa headquarterd")[0].text.startswith("Kalvorinsa")  # misspellings
    assert mem.recall("Kalvorinsa headquarters")[0].text.startswith("Kalvorinsa")  # inflection
