"""OMEGA core: the project's own neural language model (Phase 2).

v0 is a byte-level decoder trained from scratch on a CPU, in two arms that differ
in one variable: full-precision weights vs ternary {-1, 0, +1} weights
(quantization-aware training with a straight-through estimator). Inference runs
in a native Numba engine that loops over tokens without returning to Python.
"""
