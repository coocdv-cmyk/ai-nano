# EXP-013 — N3: learned bilingual encoder

Pre-registration: `RESEARCH/EXP-N-PLAN.md` §5.

**Hypothesis.** A `< 500 KB int8` contrastive bi-encoder generalises across
unseen AR/EN synonyms where the static concept anchors do not.

## Instrument

`concepts.py` authors a fresh 62-concept training inventory and a **disjoint**
30-concept held-out inventory (exact-string disjointness is asserted by
`tests/test_exp_n3.py`). The held-out probe has 90 English documents and 60
queries (30 AR + 30 EN):

- one gold document per query — no translation twins;
- query and gold share no surface token (AR↔EN, and EN query uses a different
  English synonym from the document);
- per gold, two near-duplicate distractors that change only a qualifier or only
  the keyword;
- 16 out-of-scope queries whose concept is absent from the document pool.

The **static baseline is recorded before any learned model is constructed**
(`runs/n3-v1/baseline.json`), as the N3 correction requires. The learned
encoder is a tied character 3/4/5-gram → 4,096-bucket hashed feature vector
projected to 96 dimensions, trained with InfoNCE on the training pairs only,
then quantised to int8 (393,216 bytes). Qualifiers are drawn from the training
inventory; only the concept words are unseen.

## Results

| Encoder | recall@5 (all / AR / EN) | MRR | false-accept (OOS) | bytes |
| --- | --- | --- | --- | --- |
| static anchors | 0.0833 / 0.0667 / 0.1000 | 0.0725 | 0.8125 | — |
| learned int8 | **0.7333 / 0.7000 / 0.7667** | 0.3454 | **0.0625** | 393,216 |

Gates: beats static on the new set ✓; beats the cited N0 strict baseline
`0.6667` ✓; no false-accept increase ✓; under 500 KB int8 ✓ → **overall pass**.

Cost: 5.514 s wall, 57.82 MB peak RSS, 0 repair-verifier calls (N3 does not use
one). Command is recorded verbatim in `results.json`; weights in
`runs/n3-v1/encoder.npz`.

## Negative results and limitations

- The static baseline is **near-useless** on this held-out set (recall@5
  0.0833). Its 81.25% false-accept rate is computed at a 5th-percentile
  in-scope threshold of 0.0472, i.e. the encoder cannot separate in-scope from
  out-of-scope at all here. "No false-accept increase" is therefore a weak
  comparator, not evidence of safety.
- First training attempt (`lr=0.02`, `tau=0.07`) diverged: loss 13.2 against a
  3.43 uniform baseline, recall@5 0.1167, false-accept 1.0. Recorded, not
  deleted; the shipped run uses `lr=0.005`, `tau=0.2`.
- N0's `0.6667` comes from a different suite; comparing it to this new probe
  is a cited sanity check, not a like-for-like measurement.
- This is a synthetic, same-family probe over authored vocabulary. It is **not**
  evidence of open-world bilingual understanding.
- **Hard constraint:** the Arabic byte-model deficit remains `11.12 bpb` versus
  zlib-9's `4.5447` (EXP-005). Retrieval success does not repair the byte-level
  language model.

## Files

| File | Role |
| --- | --- |
| `concepts.py` | disjoint train / held-out bilingual inventory |
| `run_exp.py` | baseline, contrastive trainer, int8 quantiser, evaluation |
| `runs/n3-v1/baseline.json` | probe + static baseline recorded before training |
| `runs/n3-v1/encoder.npz` | quantised weights |
| `runs/n3-v1/results.json` | gate verdict, cost, negative results |
