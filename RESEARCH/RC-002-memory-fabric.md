# RC-002 — A memory fabric that grows with storage, not with model size

**Status:** prototype implemented and measured (EXP-002).

1. **Question.** Can knowledge live outside model weights, so it's updated instantly, versioned, attributable, deletable, and still
   retrieved accurately and fast on a CPU?
2. **Why.** Constitution §3.5 and §4.1: keep persistent knowledge outside the weights when that is cheaper and updateable.
   This is also the main lever for a small core to show large practical capability.
3. **Implemented mechanism.** SQLite (WAL) records carry kind, key, source, confidence, a validity interval and metadata.
   Three retrieval signals are combined with reciprocal-rank fusion: BM25 over normalized text; signed feature-hashing
   embeddings (word uni- and bi-grams plus character trigrams, int8, 512 B) searched by a Numba kernel; and a gated
   trigram "fuzzy" signal for unknown words. Writing again under an existing key supersedes the old version but keeps
   it as history. `forget` is a hard delete across all indexes. Arabic normalization folds diacritics, alef/taa-marbuta/ya
   variants and Arabic-Indic digits.
4. **What is genuinely new here vs borrowed.** All parts are known techniques (BM25, feature hashing, RRF, trigram
   indexes, temporal validity). The contribution is the gating: fuzzy matching applies only to words the index vocabulary
   lacks, and inflections are told apart from misspellings by a vocabulary prefix probe. Ablations show the gating matters.
5. **Open hypotheses.**
   - **H-a:** a learned compact encoder (distilled, int8) beats hashing on semantic paraphrase at under 2x the latency.
     Needs a synonym benchmark first.
   - **H-b:** memory consolidation (merging near-duplicates, summarizing episodic history into procedural memory) reduces
     storage without lowering recall.
   - **H-c:** a coarse quantizer (IVF) keeps p50 under 10 ms at 1M records with recall@5 ≥ 0.98 relative to brute force.
6. **Next experiment.** Build a semantic-paraphrase set (synonyms, reordered relations, cross-lingual EN↔AR) and
   measure the vector signal's contribution honestly.
