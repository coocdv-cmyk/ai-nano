"""Fair, CPU-only baselines for the constructive-learning experiments.

All baselines see exactly the same training examples as the constructive model
and are evaluated on the same test inputs.  Feature extraction is shared, so no
baseline is handicapped by a worse input encoding:

* ``Memorizer``      - exact-match lookup; the pure memorisation control.
* ``NearestNeighbor``- nonparametric transfer, including role transfer for
                       relational (scene -> entity) tasks.
* ``MLP``            - a small numpy network with one hidden layer.  For
                       relational tasks it is a *per-entity scorer* trained with
                       a softmax over the entities of each scene, which is the
                       correct architecture for that output type (a global
                       entity-id classifier would be a straw man).

The baselines are deliberately small: the point is to compare learning
efficiency, not to win with capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import COLORS, SHAPES, Entity, Scene, SIZES

_COLOR_IX = {c: i for i, c in enumerate(COLORS)}
_SHAPE_IX = {s: i for i, s in enumerate(SHAPES)}
_SIZE_IX = {s: i for i, s in enumerate(SIZES)}

SCENE_DIM = 12 + 12 + 16
ENTITY_DIM = 16
LIST_SLOTS = 9
LIST_DIM = 2 * LIST_SLOTS + 1
INT_DIM = 7
PAIR_DIM = 7


class NotSupported(Exception):
    """Raised when a baseline cannot represent an output type."""


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _onehot(value: str, table: dict[str, int], n: int) -> list[float]:
    v = [0.0] * n
    v[table.get(value, 0)] = 1.0
    return v


def scene_features(scene: Scene) -> np.ndarray:
    n = max(1, len(scene.entities))
    q = scene.by_id(scene.query)
    colors = [sum(1 for e in scene.entities if e.color == c) / n for c in COLORS]
    shapes = [sum(1 for e in scene.entities if e.shape == s) / n for s in SHAPES]
    sizes = [sum(1 for e in scene.entities if e.size == s) / n for s in SIZES]
    glob = colors + shapes + sizes + [
        n / 6.0,
        sum(e.x for e in scene.entities) / (n * 4.0),
        sum(e.y for e in scene.entities) / (n * 4.0),
    ]
    query = (_onehot(q.color, _COLOR_IX, 4) + _onehot(q.shape, _SHAPE_IX, 4)
             + _onehot(q.size, _SIZE_IX, 2) + [q.x / 4.0, q.y / 4.0])
    return np.asarray(glob + query, dtype=np.float64)


def entity_features(scene: Scene, e: Entity) -> np.ndarray:
    q = scene.by_id(scene.query)
    return np.asarray(
        _onehot(e.color, _COLOR_IX, 4)
        + _onehot(e.shape, _SHAPE_IX, 4)
        + _onehot(e.size, _SIZE_IX, 2)
        + [
            e.x / 4.0,
            e.y / 4.0,
            (e.x - q.x) / 4.0,
            (e.y - q.y) / 4.0,
            (abs(e.x - q.x) + abs(e.y - q.y)) / 8.0,
            1.0 if e.id == q.id else 0.0,
        ],
        dtype=np.float64,
    )


def list_features(xs: list[int]) -> np.ndarray:
    """Fixed-length slot encoding: value plus presence flag for each position.

    A padded encoding is the standard neural encoding for variable-length
    input.  Unlike a bag of summary statistics it does not force the network to
    *multiply* two features to recover simple aggregates such as ``sum_list``,
    so the baseline is not handicapped by its input representation.
    """
    v = np.zeros(LIST_DIM, dtype=np.float64)
    for i, x in enumerate(xs[:LIST_SLOTS]):
        v[2 * i] = x / 9.0
        v[2 * i + 1] = 1.0
    v[-1] = len(xs) / 10.0
    return v


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def int_features(x: int) -> np.ndarray:
    return np.asarray(
        [x / 20.0, (x % 2), (x % 3) / 2.0, (x % 5) / 4.0,
         1.0 if _is_prime(x) else 0.0,
         sum(int(c) for c in str(abs(x))) / 10.0,
         math.log1p(abs(x)) / 3.0],
        dtype=np.float64,
    )


def pair_features(p: tuple[int, int]) -> np.ndarray:
    a, b = p
    return np.asarray([a / 12.0, b / 12.0, (a + b) / 24.0, abs(a - b) / 12.0,
                       math.gcd(abs(a), abs(b)) / 12.0, min(a, b) / 12.0,
                       max(a, b) / 12.0], dtype=np.float64)


def features_of(x: Any, input_type: str) -> np.ndarray:
    if input_type == "scene":
        return scene_features(x)
    if input_type == "list":
        return list_features(x)
    if input_type == "pair":
        return pair_features(x)
    return int_features(int(x))


# ---------------------------------------------------------------------------
# Memoriser
# ---------------------------------------------------------------------------


class Memorizer:
    """Exact-input lookup.  The control that cannot generalise."""

    name = "memorizer"

    def __init__(self, **kw: Any) -> None:
        self.table: dict[tuple, Any] = {}

    def fit(self, examples: list[tuple[Any, Any]], input_type: str, output_type: str,
            validation: list[tuple[Any, Any]] | None = None, **kw: Any) -> None:
        self.input_type = input_type
        self.output_type = output_type
        for x, y in examples:
            self.table[_key(x, input_type)] = y

    def predict(self, x: Any) -> Any:
        return self.table.get(_key(x, self.input_type))


def _key(x: Any, input_type: str) -> tuple:
    if input_type == "scene":
        return ("s", tuple((e.id, e.color, e.shape, e.size, e.x, e.y) for e in x.entities), x.query)
    if input_type == "list":
        return ("l", tuple(x))
    if input_type == "pair":
        return ("p", tuple(x))
    return ("i", int(x))


# ---------------------------------------------------------------------------
# Nearest neighbour
# ---------------------------------------------------------------------------


class NearestNeighbor:
    """k-NN, with role transfer for relational outputs."""

    name = "nearest_neighbor"

    def __init__(self, k: int = 3, **kw: Any) -> None:
        self.k = k

    def fit(self, examples: list[tuple[Any, Any]], input_type: str, output_type: str,
            validation: list[tuple[Any, Any]] | None = None, **kw: Any) -> None:
        self.input_type = input_type
        self.output_type = output_type
        self.xs = [x for x, _ in examples]
        self.ys = [y for _, y in examples]
        self.fx = [features_of(x, input_type) for x in self.xs]
        self.scene_feats = [scene_features(x) for x in self.xs] if input_type == "scene" else None
        self.target_feats = (
            [entity_features(x, y) for x, y in examples] if input_type == "scene" else None
        )

    def predict(self, x: Any) -> Any:
        if self.input_type == "scene":
            d = [(np.sum((scene_features(x) - s) ** 2), i) for i, s in enumerate(self.scene_feats)]
            d.sort()
            i = d[0][1]
            target_f = self.target_feats[i]
            best, best_d = None, float("inf")
            q = x.by_id(x.query)
            for e in sorted(x.entities, key=lambda e: e.id):
                if e.id == q.id:
                    continue
                dd = float(np.sum((entity_features(x, e) - target_f) ** 2))
                if dd < best_d:
                    best, best_d = e, dd
            return best
        f = features_of(x, self.input_type)
        d = sorted(((float(np.sum((f - g) ** 2)), i) for i, g in enumerate(self.fx)))[: self.k]
        ys = [self.ys[i] for _, i in d]
        if self.output_type == "bool":
            return sum(1 for y in ys if y) * 2 >= len(ys)
        if isinstance(ys[0], (int, float)):
            return int(round(sum(float(y) for y in ys) / len(ys)))
        return ys[0]


# ---------------------------------------------------------------------------
# Small MLP (numpy)
# ---------------------------------------------------------------------------


class MLP:
    """One hidden layer, tanh, full-batch gradient descent with L2.

    Two details make this a fair few-shot baseline rather than a straw man:

    * regression targets are standardised before training and inverted at
      prediction time, so the learning rate is meaningful for outputs whose
      scale differs by orders of magnitude (``sum_list`` vs ``fib``);
    * the scene scorer is trained with *per-scene averaged* gradients.  Summed
      gradients grow with the number of scenes, which silently inflated the
      step size as the support set grew.  Averaging makes the learning rate
      comparable at k=3 and k=20.
    """

    name = "mlp"

    def __init__(self, hidden: int = 24, epochs: int = 400, lr: float = 0.03,
                 l2: float = 1e-3, seed: int = 0, **kw: Any) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # -- setup ----------------------------------------------------------
    def _init(self, d: int, out: int) -> None:
        r = self._rng
        self.W1 = r.normal(0, 1 / math.sqrt(d), (self.hidden, d))
        self.b1 = np.zeros(self.hidden)
        self.W2 = r.normal(0, 1 / math.sqrt(self.hidden), (out, self.hidden))
        self.b2 = np.zeros(out)

    @staticmethod
    def _standardise(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = F.mean(axis=0)
        sd = F.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return (F - mu) / sd, mu, sd

    # -- training -------------------------------------------------------
    def fit(self, examples: list[tuple[Any, Any]], input_type: str, output_type: str,
            validation: list[tuple[Any, Any]] | None = None, **kw: Any) -> None:
        self.input_type = input_type
        self.output_type = output_type
        if output_type == "list":
            raise NotSupported("MLP baseline does not model variable-length list outputs")
        if input_type == "scene":
            self._fit_scene(examples, init=True)
            return
        F = np.stack([features_of(x, input_type) for x, _ in examples])
        F, self.mu, self.sd = self._standardise(F)
        y = np.asarray([float(v) for _, v in examples])
        if output_type == "bool":
            y = (y > 0.5).astype(float)
            self.ymu, self.ysd = 0.0, 1.0
        else:
            self.ymu = float(y.mean())
            self.ysd = float(y.std()) or 1.0
            y = (y - self.ymu) / self.ysd
        self._init(F.shape[1], 1)
        self._train(F, y.reshape(-1, 1))

    def partial_fit(self, examples: list[tuple[Any, Any]], input_type: str,
                    output_type: str, **kw: Any) -> None:
        """Sequential fine-tuning: continue from the current weights.

        Used for the continual-learning baseline, where re-initialising on each
        task would measure nothing but amnesia by construction.
        """
        if (not hasattr(self, "W1") or getattr(self, "input_type", None) != input_type
                or getattr(self, "output_type", None) != output_type
                or input_type != "scene" or self.W1.shape[1] != ENTITY_DIM):
            self.fit(examples, input_type, output_type, **kw)
            return
        self._fit_scene(examples, init=False)

    def _fit_scene(self, examples: list[tuple[Scene, Entity]], init: bool = True) -> None:
        feats = []
        targets = []
        for scene, target in examples:
            ents = sorted(scene.entities, key=lambda e: e.id)
            F = np.stack([entity_features(scene, e) for e in ents])
            t = [i for i, e in enumerate(ents) if e.id == target.id][0]
            feats.append((F, t))
        stacked = np.concatenate([f for f, _ in feats], axis=0)
        if init or not hasattr(self, "mu") or self.mu.shape[0] != stacked.shape[1]:
            stacked, self.mu, self.sd = self._standardise(stacked)
            self._init(stacked.shape[1], 1)
        feats = [((f - self.mu) / self.sd, t) for f, t in feats]
        self._train_scene(feats)

    def _apply_decay(self, dW1: np.ndarray, dW2: np.ndarray) -> None:
        if self.l2:
            dW1 += self.l2 * self.W1
            dW2 += self.l2 * self.W2

    def _train(self, F: np.ndarray, Y: np.ndarray) -> None:
        n = F.shape[0]
        mW1 = np.zeros_like(self.W1); mb1 = np.zeros_like(self.b1)
        mW2 = np.zeros_like(self.W2); mb2 = np.zeros_like(self.b2)
        mom = 0.9
        for _ in range(self.epochs):
            H = np.tanh(self.W1 @ F.T + self.b1[:, None])
            out = self.W2 @ H + self.b2[:, None]
            dout = (out - Y.T) * (2.0 / n)
            dW2 = dout @ H.T
            db2 = dout.sum(axis=1)
            dH = (self.W2.T @ dout) * (1.0 - H ** 2)
            dW1 = dH @ F
            db1 = dH.sum(axis=1)
            self._apply_decay(dW1, dW2)
            for p_, g, m in ((self.W1, dW1, mW1), (self.b1, db1, mb1),
                             (self.W2, dW2, mW2), (self.b2, db2, mb2)):
                m *= mom
                m += g
                p_ -= self.lr * m

    def _train_scene(self, feats: list[tuple[np.ndarray, int]]) -> None:
        mW1 = np.zeros_like(self.W1); mb1 = np.zeros_like(self.b1)
        mW2 = np.zeros_like(self.W2); mb2 = np.zeros_like(self.b2)
        mom = 0.9
        n = max(1, len(feats))
        for _ in range(self.epochs):
            gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
            gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
            for F, t in feats:
                H = np.tanh(self.W1 @ F.T + self.b1[:, None])
                s = (self.W2 @ H + self.b2[:, None]).ravel()
                s = s - s.max()
                p = np.exp(s)
                p = p / p.sum()
                p[t] -= 1.0
                dW2 = p[None, :] @ H.T
                db2 = np.asarray([p.sum()])
                dH = (self.W2.T @ p[None, :]) * (1.0 - H ** 2)
                dW1 = dH @ F
                db1 = dH.sum(axis=1)
                gW1 += dW1; gb1 += db1; gW2 += dW2; gb2 += db2
            gW1 /= n; gb1 /= n; gW2 /= n; gb2 /= n
            self._apply_decay(gW1, gW2)
            for p_, g, m in ((self.W1, gW1, mW1), (self.b1, gb1, mb1),
                             (self.W2, gW2, mW2), (self.b2, gb2, mb2)):
                m *= mom
                m += g
                p_ -= self.lr * m

    # -- inference ------------------------------------------------------
    def predict(self, x: Any) -> Any:
        if self.input_type == "scene":
            ents = sorted(x.entities, key=lambda e: e.id)
            F = np.stack([entity_features(x, e) for e in ents])
            F = (F - self.mu) / self.sd
            H = np.tanh(self.W1 @ F.T + self.b1[:, None])
            s = (self.W2 @ H + self.b2[:, None]).ravel()
            return ents[int(np.argmax(s))]
        f = features_of(x, self.input_type)
        f = (f - self.mu) / self.sd
        H = np.tanh(self.W1 @ f + self.b1)
        v = float((self.W2 @ H + self.b2)[0])
        if self.output_type == "bool":
            return v > 0.5
        return int(round(v * self.ysd + self.ymu))

    # -- size -----------------------------------------------------------
    def n_parameters(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BASELINES = {
    "memorizer": Memorizer,
    "nearest_neighbor": NearestNeighbor,
    "mlp": MLP,
}


def make_baseline(name: str, **kw: Any):
    return BASELINES[name](**kw)
