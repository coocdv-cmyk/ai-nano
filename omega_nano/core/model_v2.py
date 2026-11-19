"""OmegaCore v2: Multi-Scale Patched, Fast-Weight Plastic, Associative-Copy Looped Core.

Addresses the three structural walls observed in OmegaCore v0/v1 (EXP-003 / EXP-004):
1. Byte & UTF-8 Arabic tax: causal multi-scale n-gram hash stem + depthwise conv +
   strided patch thought stream (S=4) gives 4x effective context & 16x cheaper attention.
2. Finite-context & static-weight wall: Delta-Rule Fast Weights (W_t updated in O(d^2)
   test-time state) + causal O(1) suffix-match CopyGate for long-range exact recall.
3. Shallow fixed depth: weight-shared looped ThoughtBlocks with input anchoring + ACT.

Every component is strictly causal: logits at position t depend solely on bytes <= t.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numba
except ImportError:
    numba = None

if TYPE_CHECKING:
    from .model import CoreConfig


# ---------------------------------------------------------------------------
# Causal Associative Copy-Memory Extractor (Numba-accelerated, pure-Py fallback)
# ---------------------------------------------------------------------------

def _extract_copy_py(arr: np.ndarray, window: int, max_match: int = 16):
    B, T = arr.shape
    b1 = np.zeros((B, T), dtype=np.int64)
    b2 = np.zeros((B, T), dtype=np.int64)
    m1 = np.zeros((B, T), dtype=np.float32)
    m2 = np.zeros((B, T), dtype=np.float32)
    d1 = np.zeros((B, T), dtype=np.float32)

    t3_size = 32768
    t4_size = 32768
    for bi in range(B):
        x = arr[bi]
        last2 = [-1] * 65536
        last3 = [-1] * t3_size
        last4 = [-1] * t4_size
        for t in range(T):
            best_pos = -1
            best_len = 0
            sec_pos = -1
            sec_len = 0

            if t >= 3:
                h4 = ((int(x[t - 3]) * 131 + int(x[t - 2])) * 131 + int(x[t - 1])) * 131 + int(x[t])
                h4 = (h4 * 2654435761) & (t4_size - 1)
                j = last4[h4]
                if j >= 3 and (t - j) <= window:
                    mlen = 0
                    while mlen < max_match and (t - mlen) >= 0 and (j - mlen) >= 0 and x[t - mlen] == x[j - mlen]:
                        mlen += 1
                    if mlen >= 4:
                        best_pos, best_len = j, mlen

            if t >= 2:
                h3 = (int(x[t - 2]) * 131 + int(x[t - 1])) * 131 + int(x[t])
                h3 = (h3 * 2654435761) & (t3_size - 1)
                j = last3[h3]
                if j >= 2 and (t - j) <= window:
                    mlen = 0
                    while mlen < max_match and (t - mlen) >= 0 and (j - mlen) >= 0 and x[t - mlen] == x[j - mlen]:
                        mlen += 1
                    if mlen >= 3:
                        if mlen > best_len:
                            sec_pos, sec_len = best_pos, best_len
                            best_pos, best_len = j, mlen
                        elif j != best_pos and int(x[j + 1]) != int(x[best_pos + 1]):
                            sec_pos, sec_len = j, mlen

            if t >= 1:
                k2 = (int(x[t - 1]) << 8) | int(x[t])
                j = last2[k2]
                if j >= 1 and (t - j) <= window:
                    mlen = 0
                    while mlen < max_match and (t - mlen) >= 0 and (j - mlen) >= 0 and x[t - mlen] == x[j - mlen]:
                        mlen += 1
                    if mlen >= 2:
                        if mlen > best_len:
                            sec_pos, sec_len = best_pos, best_len
                            best_pos, best_len = j, mlen
                        elif sec_len == 0 and j != best_pos and (best_pos < 0 or int(x[j + 1]) != int(x[best_pos + 1])):
                            sec_pos, sec_len = j, mlen

            if best_pos >= 0:
                b1[bi, t] = int(x[best_pos + 1])
                m1[bi, t] = float(best_len)
                d1[bi, t] = float(t - best_pos)
            if sec_pos >= 0:
                b2[bi, t] = int(x[sec_pos + 1])
                m2[bi, t] = float(sec_len)

            # Insert strictly AFTER querying position t so causality is exact.
            if t >= 1:
                last2[(int(x[t - 1]) << 8) | int(x[t])] = t
            if t >= 2:
                h3u = (((int(x[t - 2]) * 131 + int(x[t - 1])) * 131 + int(x[t])) * 2654435761) & (t3_size - 1)
                last3[h3u] = t
            if t >= 3:
                h4u = ((((int(x[t - 3]) * 131 + int(x[t - 2])) * 131 + int(x[t - 1])) * 131 + int(x[t])) * 2654435761) & (t4_size - 1)
                last4[h4u] = t
    return b1, b2, m1, m2, d1


if numba is not None:
    @numba.njit(cache=True)
    def _extract_copy_numba(arr: np.ndarray, window: int, max_match: int = 16):
        B, T = arr.shape
        b1 = np.zeros((B, T), dtype=np.int64)
        b2 = np.zeros((B, T), dtype=np.int64)
        m1 = np.zeros((B, T), dtype=np.float32)
        m2 = np.zeros((B, T), dtype=np.float32)
        d1 = np.zeros((B, T), dtype=np.float32)

        t3_mask = 32767
        t4_mask = 32767
        last2 = np.empty(65536, dtype=np.int32)
        last3 = np.empty(32768, dtype=np.int32)
        last4 = np.empty(32768, dtype=np.int32)

        for bi in range(B):
            last2.fill(-1)
            last3.fill(-1)
            last4.fill(-1)
            for t in range(T):
                best_pos = -1
                best_len = 0
                sec_pos = -1
                sec_len = 0

                if t >= 3:
                    h4 = (((np.int64(arr[bi, t - 3]) * 131 + np.int64(arr[bi, t - 2])) * 131 + np.int64(arr[bi, t - 1])) * 131 + np.int64(arr[bi, t]))
                    h4 = (h4 * np.int64(2654435761)) & t4_mask
                    j = int(last4[h4])
                    if j >= 3 and (t - j) <= window:
                        mlen = 0
                        while mlen < max_match and (t - mlen) >= 0 and (j - mlen) >= 0 and arr[bi, t - mlen] == arr[bi, j - mlen]:
                            mlen += 1
                        if mlen >= 4:
                            best_pos, best_len = j, mlen

                if t >= 2:
                    h3 = ((np.int64(arr[bi, t - 2]) * 131 + np.int64(arr[bi, t - 1])) * 131 + np.int64(arr[bi, t]))
                    h3 = (h3 * np.int64(2654435761)) & t3_mask
                    j = int(last3[h3])
                    if j >= 2 and (t - j) <= window:
                        mlen = 0
                        while mlen < max_match and (t - mlen) >= 0 and (j - mlen) >= 0 and arr[bi, t - mlen] == arr[bi, j - mlen]:
                            mlen += 1
                        if mlen >= 3:
                            if mlen > best_len:
                                sec_pos, sec_len = best_pos, best_len
                                best_pos, best_len = j, mlen
                            elif j != best_pos and arr[bi, j + 1] != arr[bi, best_pos + 1]:
                                sec_pos, sec_len = j, mlen

                if t >= 1:
                    k2 = (int(arr[bi, t - 1]) << 8) | int(arr[bi, t])
                    j = int(last2[k2])
                    if j >= 1 and (t - j) <= window:
                        mlen = 0
                        while mlen < max_match and (t - mlen) >= 0 and (j - mlen) >= 0 and arr[bi, t - mlen] == arr[bi, j - mlen]:
                            mlen += 1
                        if mlen >= 2:
                            if mlen > best_len:
                                sec_pos, sec_len = best_pos, best_len
                                best_pos, best_len = j, mlen
                            elif sec_len == 0 and j != best_pos and (best_pos < 0 or arr[bi, j + 1] != arr[bi, best_pos + 1]):
                                sec_pos, sec_len = j, mlen

                if best_pos >= 0:
                    b1[bi, t] = int(arr[bi, best_pos + 1])
                    m1[bi, t] = np.float32(best_len)
                    d1[bi, t] = np.float32(t - best_pos)
                if sec_pos >= 0:
                    b2[bi, t] = int(arr[bi, sec_pos + 1])
                    m2[bi, t] = np.float32(sec_len)

                if t >= 1:
                    last2[(int(arr[bi, t - 1]) << 8) | int(arr[bi, t])] = t
                if t >= 2:
                    h3u = (((np.int64(arr[bi, t - 2]) * 131 + np.int64(arr[bi, t - 1])) * 131 + np.int64(arr[bi, t])) * np.int64(2654435761)) & t3_mask
                    last3[h3u] = t
                if t >= 3:
                    h4u = ((((np.int64(arr[bi, t - 3]) * 131 + np.int64(arr[bi, t - 2])) * 131 + np.int64(arr[bi, t - 1])) * 131 + np.int64(arr[bi, t])) * np.int64(2654435761)) & t4_mask
                    last4[h4u] = t
        return b1, b2, m1, m2, d1
else:
    _extract_copy_numba = None


def extract_copy_signals(idx: torch.Tensor, window: int) -> tuple[torch.Tensor, ...]:
    arr = idx.detach().cpu().numpy().astype(np.int64, copy=False)
    fn = _extract_copy_numba if _extract_copy_numba is not None else _extract_copy_py
    b1, b2, m1, m2, d1 = fn(arr, int(window))
    dev = idx.device
    return (
        torch.from_numpy(b1).to(dev),
        torch.from_numpy(b2).to(dev),
        torch.from_numpy(m1).to(dev),
        torch.from_numpy(m2).to(dev),
        torch.from_numpy(d1).to(dev),
    )


# ---------------------------------------------------------------------------
# Neural Building Blocks: Delta-Rule Fast Weights + Multi-Scale Thought Core
# ---------------------------------------------------------------------------

class DeltaFastWeights(nn.Module):
    """Causal plastic fast-weight matrix W_t updated via the Delta Rule per step.

    Maintains O(H * d_f^2) associative state independent of sequence length, enabling
    error-corrective in-context key-value binding beyond finite attention windows.
    """

    def __init__(self, d: int, heads: int, d_f: int, ternary: bool = False):
        super().__init__()
        from .model import BitLinear

        Lin = BitLinear if ternary else nn.Linear
        self.heads = heads
        self.d_f = d_f
        fd = heads * d_f
        self.qkv = Lin(d, 3 * fd, bias=False)
        self.gates = nn.Linear(d, 2 * heads, bias=True)
        self.out_proj = Lin(fd, d, bias=False)
        nn.init.constant_(self.gates.bias[:heads], 1.5)   # high initial retention lambda
        nn.init.constant_(self.gates.bias[heads:], -0.5)  # moderate initial write rate beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, _ = x.shape
        H, Df = self.heads, self.d_f
        q, k, v = self.qkv(x).split(H * Df, dim=-1)
        q = F.normalize(q.view(B, P, H, Df), dim=-1)
        k = F.normalize(k.view(B, P, H, Df), dim=-1)
        v = v.view(B, P, H, Df)

        g = self.gates(x)
        lam = torch.sigmoid(g[..., :H]).unsqueeze(-1).unsqueeze(-1)  # [B, P, H, 1, 1]
        beta = torch.sigmoid(g[..., H:]).unsqueeze(-1)               # [B, P, H, 1]

        W = x.new_zeros((B, H, Df, Df))
        outs = []
        for p in range(P):
            kp = k[:, p]  # [B, H, Df]
            vp = v[:, p]
            qp = q[:, p]
            pred = torch.matmul(W, kp.unsqueeze(-1)).squeeze(-1)
            err = beta[:, p] * (vp - pred)
            W = lam[:, p] * W + torch.matmul(err.unsqueeze(-1), kp.unsqueeze(-2))
            mp = torch.matmul(W, qp.unsqueeze(-1)).squeeze(-1)
            outs.append(mp)
        out = torch.stack(outs, dim=1).reshape(B, P, H * Df)
        return self.out_proj(out)


class ThoughtBlock(nn.Module):
    """Coarse thought block: Causal RoPE Attention + optional Delta Fast Weights + SwiGLU."""

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        from .model import BitLinear, RMSNorm

        Lin = BitLinear if cfg.ternary else nn.Linear
        self.h = cfg.heads
        self.hd = cfg.d // cfg.heads
        self.n1 = RMSNorm(cfg.d)
        self.qkv = Lin(cfg.d, 3 * cfg.d, bias=False)
        self.proj = Lin(cfg.d, cfg.d, bias=False)

        self.use_fw = cfg.use_fast_weights
        if self.use_fw:
            self.n_fw = RMSNorm(cfg.d)
            self.fw = DeltaFastWeights(cfg.d, cfg.heads, cfg.fast_dim, ternary=cfg.ternary)

        self.n2 = RMSNorm(cfg.d)
        self.gate = Lin(cfg.d, cfg.hidden, bias=False)
        self.up = Lin(cfg.d, cfg.hidden, bias=False)
        self.down = Lin(cfg.hidden, cfg.d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        from .model import apply_rope

        B, P, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, P, self.h, self.hd).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, P, D))
        if self.use_fw:
            x = x + self.fw(self.n_fw(x))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class OmegaCoreV2(nn.Module):
    """OMEGA Core v2: Hash-Ngram Byte Stem + Strided Looped Thought Core + Fast Weights + Copy Gate."""

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        from .model import BitLinear, RMSNorm, rope_tables

        self.cfg = cfg
        self.patch_size = max(2, cfg.patch_size)
        self.n_loops = max(1, cfg.n_loops)
        self.mtp_k = max(1, cfg.mtp_k)
        self.hash_buckets = max(256, cfg.hash_buckets)
        d = cfg.d
        Lin = BitLinear if cfg.ternary else nn.Linear

        # 1. Multi-scale causal byte + hash-ngram stem (captures UTF-8 multi-byte & keywords at layer 0)
        self.emb = nn.Embedding(cfg.vocab, d)
        d_hash = max(16, d // 4)
        self.emb_h2 = nn.Embedding(self.hash_buckets, d_hash)
        self.emb_h3 = nn.Embedding(self.hash_buckets, d_hash)
        self.emb_h4 = nn.Embedding(self.hash_buckets, d_hash)
        self.ngram_proj = nn.Linear(3 * d_hash, d, bias=False)
        self.stem_norm = RMSNorm(d)
        self.dw_conv = nn.Conv1d(d, d, kernel_size=4, groups=d, bias=True)
        self.dw_proj = nn.Linear(d, d, bias=False)

        # 2. Causal fine-to-coarse boundary-weighted patch summarizer
        self.patch_scorer = nn.Linear(d, 1, bias=True)
        self.patch_proj = nn.Linear(d, d, bias=False)
        self.bos_thought = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # 3. Weight-shared recurrent coarse ThoughtCore with ACT halting
        self.blocks = nn.ModuleList(ThoughtBlock(cfg) for _ in range(cfg.layers))
        self.loop_emb = nn.Parameter(torch.randn(self.n_loops, d) * 0.02)
        self.anchor_gate = nn.Linear(2 * d, d, bias=True)
        self.halt_norm = RMSNorm(d)
        self.halt_head = nn.Linear(d, 1, bias=True)

        # 4. Fine synthesis head fusing local fine byte state + completed prior thought + patch offset
        self.offset_emb = nn.Embedding(self.patch_size, d)
        self.cross_gate = nn.Linear(2 * d, d, bias=True)
        fine_hid = max(64, d * 2)
        self.fine_norm = RMSNorm(d)
        self.fine_gate = Lin(d, fine_hid, bias=False)
        self.fine_up = Lin(d, fine_hid, bias=False)
        self.fine_down = Lin(fine_hid, d, bias=False)

        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, cfg.vocab, bias=False)
        self.mtp_projs = nn.ModuleList(
            nn.Sequential(RMSNorm(d), nn.Linear(d, d, bias=False))
            for _ in range(self.mtp_k - 1)
        )

        # 5. Optional causal associative copy gate
        self.use_copy_gate = cfg.use_copy_gate
        if self.use_copy_gate:
            self.copy_gate_net = nn.Linear(d + 4, 2, bias=True)  # [alpha_copy, beta_sec]

        max_patches = max(256, (max(cfg.ctx, 4096) // self.patch_size) + 8)
        cos, sin = rope_tables(max_patches, d // cfg.heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.last_avg_loops: float = float(self.n_loops)

        self.apply(self._init)
        nn.init.constant_(self.halt_head.bias, -1.0)
        if self.use_copy_gate:
            nn.init.zeros_(self.copy_gate_net.bias)
        eff_depth = max(1, cfg.layers * self.n_loops)
        for b in self.blocks:
            for lin in (b.proj, b.down):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * eff_depth))
            if b.use_fw:
                nn.init.normal_(b.fw.out_proj.weight, std=0.02 / math.sqrt(2 * eff_depth))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(m.weight, std=0.02)

    def _causal_stem(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x0 = self.emb(idx)
        z1 = idx.new_zeros((B, 1))
        m1 = torch.cat([z1, idx[:, :-1]], dim=1)
        m2 = torch.cat([z1, z1, idx[:, :-2]], dim=1) if T > 1 else idx.new_zeros((B, T))
        m3 = torch.cat([z1, z1, z1, idx[:, :-3]], dim=1) if T > 2 else idx.new_zeros((B, T))

        mask = self.hash_buckets - 1
        h2 = ((m1 * 131 + idx) * 2654435761) & mask
        h3 = (((m2 * 131 + m1) * 131 + idx) * 2654435761) & mask
        h4 = ((((m3 * 131 + m2) * 131 + m1) * 131 + idx) * 2654435761) & mask
        e_ng = self.ngram_proj(torch.cat([self.emb_h2(h2), self.emb_h3(h3), self.emb_h4(h4)], dim=-1))

        xf = self.stem_norm(x0 + e_ng)
        # Strict causal depthwise conv: left-pad by kernel-1 = 3
        c_in = F.pad(xf.transpose(1, 2), (3, 0))
        c_out = F.silu(self.dw_conv(c_in)).transpose(1, 2)
        return xf + self.dw_proj(c_out)

    def _run_thought_loops(self, z0: torch.Tensor, early_exit: bool) -> tuple[torch.Tensor, torch.Tensor]:
        B, P, D = z0.shape
        x = z0
        rem = z0.new_ones((B, P, 1))
        out_state = torch.zeros_like(z0)
        ponder = z0.new_zeros((B, P, 1))
        halted = z0.new_zeros((B, P, 1), dtype=torch.bool)
        for t in range(self.n_loops):
            gate = torch.sigmoid(self.anchor_gate(torch.cat([x, z0], dim=-1)))
            h = x + gate * z0 + self.loop_emb[t]
            for b in self.blocks:
                h = b(h, self.cos, self.sin)
            if early_exit and self.n_loops > 1:
                x = torch.where(halted, x, h)
            else:
                x = h
            p = torch.sigmoid(self.halt_head(self.halt_norm(x)))
            is_last = t == self.n_loops - 1
            q = rem if is_last else rem * p
            rem = rem * (1.0 - p)
            out_state = out_state + q * x
            ponder = ponder + (t + 1) * q
            if early_exit and not is_last:
                halted = halted | ((1.0 - rem) >= (1.0 - self.cfg.halt_eps))
                if bool(halted.all()):
                    out_state = out_state + rem * x
                    ponder = ponder + (t + 1) * rem
                    break
        self.last_avg_loops = float(ponder.detach().mean().item())
        return out_state, ponder.mean()

    def _coarse_context_for_bytes(self, xf: torch.Tensor, early_exit: bool) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = xf.shape
        S = self.patch_size
        n_ctx_patches = (T - 1) // S
        if n_ctx_patches == 0:
            self.last_avg_loops = 0.0
            return self.bos_thought.expand(B, T, D), xf.new_zeros(())

        # Only completed patches [0 .. n_ctx_patches*S - 1] enter coarse core -> strictly causal!
        chunk = xf[:, : n_ctx_patches * S, :].view(B, n_ctx_patches, S, D)
        w = torch.softmax(self.patch_scorer(chunk), dim=2)
        z0 = self.patch_proj((w * chunk).sum(dim=2))  # [B, P, D]
        if n_ctx_patches > self.cos.shape[0]:
            from .model import rope_tables
            cos, sin = rope_tables(n_ctx_patches + 16, D // self.cfg.heads)
            self.cos = cos.to(xf.device)
            self.sin = sin.to(xf.device)
        c_patches, mean_ponder = self._run_thought_loops(z0, early_exit=early_exit)

        # Patch p's completed thought c_patches[:, p] is first visible to bytes [(p+1)*S .. (p+2)*S - 1]
        bos = self.bos_thought.expand(B, 1, D)
        shifted = torch.cat([bos, c_patches], dim=1)  # [B, n_ctx_patches + 1, D]
        expanded = shifted.repeat_interleave(S, dim=1)[:, :T, :]
        return expanded, mean_ponder

    def _blend_copy(self, idx: torch.Tensor, h_final: torch.Tensor, logits_neural: torch.Tensor) -> torch.Tensor:
        p_neural = F.softmax(logits_neural, dim=-1)
        b1, b2, m1, m2, d1 = extract_copy_signals(idx, window=self.cfg.copy_window)
        has_m1 = (m1 > 0).to(h_final.dtype)
        has_m2 = (m2 > 0).to(h_final.dtype)

        feats = torch.stack([
            has_m1,
            torch.log1p(m1) / 3.0,
            torch.log1p(m2) / 3.0,
            torch.log1p(d1) / 9.0,
        ], dim=-1)
        g_raw = self.copy_gate_net(torch.cat([h_final, feats], dim=-1))
        alpha = torch.sigmoid(g_raw[..., 0:1]) * has_m1.unsqueeze(-1)
        w_sec = torch.sigmoid(g_raw[..., 1:2]) * 0.45 * has_m2.unsqueeze(-1)
        w_pri = 1.0 - w_sec

        p_copy = torch.zeros_like(p_neural)
        p_copy.scatter_add_(-1, b1.unsqueeze(-1), w_pri)
        p_copy.scatter_add_(-1, b2.unsqueeze(-1), w_sec)

        p_mix = (1.0 - alpha) * p_neural + alpha * p_copy
        return torch.log(p_mix.clamp(min=1e-7))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        early_exit = (not self.training) and (targets is None)

        xf = self._causal_stem(idx)
        coarse_ctx, mean_ponder = self._coarse_context_for_bytes(xf, early_exit=early_exit)

        offsets = torch.arange(T, device=idx.device) % self.patch_size
        xf = xf + self.offset_emb(offsets).unsqueeze(0)
        cg = torch.sigmoid(self.cross_gate(torch.cat([xf, coarse_ctx], dim=-1)))
        hf = xf + cg * coarse_ctx

        fn = self.fine_norm(hf)
        hf = hf + self.fine_down(F.silu(self.fine_gate(fn)) * self.fine_up(fn))
        h_final = self.norm(hf)
        logits_neural = self.head(h_final)

        if self.use_copy_gate:
            log_probs = self._blend_copy(idx, h_final, logits_neural)
            if targets is None:
                return log_probs, None
            loss_0 = F.nll_loss(log_probs.reshape(-1, log_probs.size(-1)), targets.reshape(-1))
            out_scores = log_probs
        else:
            if targets is None:
                return logits_neural, None
            loss_0 = F.cross_entropy(logits_neural.reshape(-1, logits_neural.size(-1)), targets.reshape(-1))
            out_scores = logits_neural

        if not self.training:
            return out_scores, loss_0

        loss = loss_0 + self.cfg.ponder_weight * mean_ponder
        if self.mtp_projs and T > 1:
            mtp_losses = []
            for offset, proj in enumerate(self.mtp_projs, start=1):
                if targets.size(1) <= offset:
                    break
                logits_k = self.head(proj(hf[:, :-offset, :]))
                tgt_k = targets[:, offset:]
                mtp_losses.append(F.cross_entropy(logits_k.reshape(-1, logits_k.size(-1)), tgt_k.reshape(-1)))
            if mtp_losses:
                loss = loss + 0.25 * (sum(mtp_losses) / len(mtp_losses))
        return out_scores, loss

    def n_params(self) -> dict:
        block = sum(p.numel() for b in self.blocks for _, p in b.named_parameters() if p.dim() == 2)
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "block_matrices": block, "other": total - block}

    @torch.no_grad()
    def generate(self, prompt: bytes, n: int, temperature: float = 0.8, top_k: int = 40, seed: int = 0) -> bytes:
        g = torch.Generator().manual_seed(seed)
        idx = torch.tensor([list(prompt)], dtype=torch.long)
        win = max(self.cfg.ctx, self.cfg.copy_window if self.use_copy_gate else self.cfg.ctx)
        for _ in range(n):
            scores, _ = self(idx[:, -win:])
            logits = scores[0, -1] / max(temperature, 1e-5)
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[-1]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)
            idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
        return bytes(idx[0, len(prompt):].tolist())
