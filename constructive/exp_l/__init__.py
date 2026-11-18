"""EXP-L: generalization bridge, semantic drift, and selective reasoning reliability.

This is an isolated research package. It is deliberately NOT imported by, and
does NOT modify, OMEGA-NANO, CCM, or any earlier experiment (EXP-H..EXP-K).
All protected systems are inspected read-only.

Research framing: useful reasoning coverage x generalization x reliability /
computational cost.  No claim of general intelligence or superiority over
large models is made or implied by this package.
"""

EXP_L_VERSION = "0.1.0"
GENERATOR_VERSION = "exp-l-gen-1"
SEED_POLICY = "explicit_sha256_derived_random.Random(seed, stream); no hash() of strings"

__all__ = ["EXP_L_VERSION", "GENERATOR_VERSION", "SEED_POLICY"]
