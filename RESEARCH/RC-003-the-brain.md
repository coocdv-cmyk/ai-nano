# RC-003 — "The brain": building a model that surpasses the strongest models

Research card per constitution §2.2 ("impossible until tested") and §7. **Status:** decomposed; first core trained (EXP-003).

## 1. The goal, stated precisely
An OMEGA-NANO system that beats leading frontier models across a broad, fair benchmark suite (§0.2, §18) while running
on ordinary hardware.

## 2. Which part looks impossible, and why (measured, not assumed)
- **Compute.** This laptop sustains ~178 GFLOPS (fp32 matmul, measured). Frontier pre-training runs are commonly
  estimated at ~10²⁵ FLOPs (a public estimate, not verified in this session). 10²⁵ / 1.78×10¹¹ ≈ 5.6×10¹³ s ≈
  **1.8 million years**, or ~3.6 million years at a realistic ~50% utilization. This is an engineering-resource limit,
  not a law of nature: the same computation is weeks on a large GPU cluster.
- **Data.** Frontier models see on the order of 10¹³ tokens. This machine has ~20 MB of local code (the Python stdlib)
  without downloads.
- **Not a limit:** the architecture. Nothing proves frontier capability needs today's dense-transformer recipe. That is
  the part the constitution asks us to attack.

## 3. Separating the limits
| Constraint | Type | Can be moved by |
|---|---|---|
| FLOPs needed for broad world knowledge in weights | engineering/resource | more compute (cloud, authorized) or *not storing knowledge in weights* |
| Knowledge capacity of a small model | information-theoretic (bits in parameters) | external memory (RC-002), retrieval |
| Reasoning reliability | open research | verification, search, tools, specialized algorithms |
| Inference cost | engineering | kernels, low-bit formats, small cores (RC-001, EXP-001) |

## 4. Three research paths (constitution §2.2 step 6)
- **Path A — System intelligence (cheapest, already started).** Small core + external memory + tools + verification.
  Win in measured narrow dimensions where specialized, verified algorithms beat generation: exact arithmetic (the math
  engine is exact at any size), durable autonomy, verified file operations. *Falsifier:* a 50-task agent benchmark where
  the system doesn't beat an LLM-only baseline on verified success.
- **Path B — Own compact core, from scratch.** Prove the training pipeline, the architecture choices and the inference
  runtime at tiny scale on this CPU, then scale only the validated recipe. *First experiment:* EXP-003 (fp32 vs
  ternary, 3.3M parameters, 30 minutes each). *Falsifier:* ternary quality collapsing relative to fp32 at equal wall-clock.
- **Path C — Distillation from open models.** Use a strong open-weights model as a temporary teacher (§1.3 allows this)
  to produce training signal for the compact core. *Requires:* downloading model weights (needs the user's explicit
  approval and several GB of disk) or a local server.

## 5. Decision
Pursue A and B now, prepare C. Never report a capability that hasn't been measured. The brain is grown by evidence gates,
not declared.
