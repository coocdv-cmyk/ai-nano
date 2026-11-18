"""OMEGA Constructive Intelligence (OCI) - an isolated research module.

The model learns typed programs from a handful of examples, validates them
against held-out evidence, composes previously validated concepts, and repairs
itself locally.  It has no external LLM dependency and no agent loop.
"""

from .core import Entity, Scene, Term, build_prims, eval_term, same_value, term_cost
from .model import Candidate, ComposeReport, Concept, ConstructiveModel, FitReport, Searcher
from .worlds import Task

__all__ = [
    "Entity",
    "Scene",
    "Term",
    "build_prims",
    "eval_term",
    "same_value",
    "term_cost",
    "Candidate",
    "ComposeReport",
    "Concept",
    "ConstructiveModel",
    "FitReport",
    "Searcher",
    "Task",
]
