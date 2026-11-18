"""OmegaCore: a compact byte-level decoder with optional ternary weights.

Bytes as tokens: no tokenizer to train, every language (including Arabic UTF-8)
and every file is representable, and the vocabulary is 256.

Block: pre-RMSNorm -> causal multi-head attention with rotary positions ->
pre-RMSNorm -> SwiGLU MLP. With `ternary=True` every projection inside the
blocks is a BitLinear: weights are quantized to {-1, 0, +1} x absmean scale in
the forward pass, while gradients flow to latent fp32 weights (straight-through
estimator). Embeddings, norms and the output head stay full precision.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoreConfig:
    vocab: int = 256
    d: int = 256
    layers: int = 4
    heads: int = 4
    hidden: int = 704  # SwiGLU width (~2/3 of 4d, multiple of 32)
    ctx: int = 256
    ternary: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def ternarize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmean ternary quantization: returns (codes in {-1,0,1}, scale)."""
    scale = w.abs().mean().clamp(min=1e-8)
    return (w / scale).round().clamp(-1, 1), scale


class BitLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes, scale = ternarize(self.weight)
        wq = self.weight + (codes * scale - self.weight).detach()  # straight-through estimator
        return F.linear(x, wq)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(ctx: int, head_dim: int, base: float = 10000.0):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    ang = torch.outer(torch.arange(ctx).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: [B, H, T, hd]; rotate pairs (even, odd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]], sin[: x.shape[-2]]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        Lin = BitLinear if cfg.ternary else nn.Linear
        self.h, self.hd = cfg.heads, cfg.d // cfg.heads
        self.n1, self.n2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.qkv = Lin(cfg.d, 3 * cfg.d, bias=False)
        self.proj = Lin(cfg.d, cfg.d, bias=False)
        self.gate = Lin(cfg.d, cfg.hidden, bias=False)
        self.up = Lin(cfg.d, cfg.hidden, bias=False)
        self.down = Lin(cfg.hidden, cfg.d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class OmegaCore(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        cos, sin = rope_tables(cfg.ctx, cfg.d // cfg.heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # scale residual outputs by depth (GPT-2 style)
            for lin in (b.proj, b.down):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def n_params(self) -> dict:
        block = sum(p.numel() for b in self.blocks for n, p in b.named_parameters() if p.dim() == 2)
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "block_matrices": block, "other": total - block}

    @torch.no_grad()
    def generate(self, prompt: bytes, n: int, temperature: float = 0.8, top_k: int = 40, seed: int = 0) -> bytes:
        g = torch.Generator().manual_seed(seed)
        idx = torch.tensor([list(prompt)], dtype=torch.long)
        for _ in range(n):
            logits, _ = self(idx[:, -self.cfg.ctx:])
            logits = logits[0, -1] / max(temperature, 1e-5)
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[-1]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)
            idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
        return bytes(idx[0, len(prompt):].tolist())
