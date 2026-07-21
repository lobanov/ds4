#!/usr/bin/env python3
"""Activity 5/6/7 — DSpark drafter HEAD in torch (MPS), RESEARCH INSTRUMENTATION.

Faithful port of dspark_oracle/forward.py forward_head + hc_primitives.hc_head/rmsnorm:
  x [b, BLOCK, HC, DIM]  (frozen pre-head feature, from the numpy oracle body)
  -> hc_head -> rmsnorm -> base_logits = x @ lm_head.T
  -> autoregressive markov rollout -> output_ids [b, BLOCK+1]

Weights loaded once via dspark_oracle/gguf_loader. LoRA adapters (hc_head_fn, norm,
markov_w1, markov_w2) are zero-init -> at construction the head is bit-identical to the
numpy forward_head (Activity 5 fidelity gate); Activity 6/7 train the LoRA params.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent.parent            # issue468/
sys.path.insert(0, str(HERE / "dspark_oracle"))
from gguf_loader import load_gguf_dense_only, index_gguf, read_tensor  # noqa: E402

HC, DIM, BLOCK, VOCAB = 4, 4096, 5, 129280
NORM_EPS, HC_EPS = 1e-6, 1e-6


def _t(a):
    return torch.as_tensor(np.ascontiguousarray(a))


class DrafterHead(nn.Module):
    def __init__(self, hc_fn, hc_scale, hc_base, norm_w, markov_w1, markov_w2, conf_proj, lm_head,
                 lora_rank: int = 0):
        super().__init__()
        self.register_buffer("hc_fn", _t(hc_fn))              # [HC, HC*DIM]
        self.register_buffer("hc_scale", _t(hc_scale))        # [1]
        self.register_buffer("hc_base", _t(hc_base))          # [HC]
        self.register_buffer("norm_w", _t(norm_w))            # [DIM]
        self.register_buffer("markov_w1", _t(markov_w1))      # [VOCAB, rank_m]
        self.register_buffer("markov_w2", _t(markov_w2))      # [VOCAB, rank_m]
        self.register_buffer("conf_proj", _t(conf_proj))      # [DIM + rank_m]
        self.register_buffer("lm_head", _t(lm_head))          # [VOCAB, DIM]
        self.lora_rank = lora_rank
        self.rank_m = markov_w1.shape[1]
        r = lora_rank
        if r > 0:
            # LoRA: delta = B @ A. Init A ~ small random, B = 0 so delta starts at 0 but
            # grad_B = grad @ A.T is non-zero (A!=0) -> B learns first, then A. (Both-zero
            # init is a dead-LoRA bug: zero gradient forever.)
            self.lora_hc_A = nn.Parameter(torch.empty(r, HC * DIM)); nn.init.normal_(self.lora_hc_A, std=0.02)
            self.lora_hc_B = nn.Parameter(torch.zeros(HC, r))
            self.lora_norm = nn.Parameter(torch.zeros(DIM))
            self.lora_mw1_A = nn.Parameter(torch.empty(r, self.rank_m)); nn.init.normal_(self.lora_mw1_A, std=0.02)
            self.lora_mw1_B = nn.Parameter(torch.zeros(VOCAB, r))
            self.lora_mw2_A = nn.Parameter(torch.empty(r, self.rank_m)); nn.init.normal_(self.lora_mw2_A, std=0.02)
            self.lora_mw2_B = nn.Parameter(torch.zeros(VOCAB, r))
            # conf_proj LoRA (for the confidence-head calibration — the verify_n lever)
            self.lora_conf_A = nn.Parameter(torch.empty(r, DIM + self.rank_m)); nn.init.normal_(self.lora_conf_A, std=0.02)
            self.lora_conf_B = nn.Parameter(torch.zeros(1, r))

    def _hc_fn(self):
        return self.hc_fn + (self.lora_hc_B @ self.lora_hc_A) if self.lora_rank else self.hc_fn

    def _norm_w(self):
        return self.norm_w * (1.0 + self.lora_norm) if self.lora_rank else self.norm_w

    def _mw1(self, ids):
        w1 = self.markov_w1 + (self.lora_mw1_B @ self.lora_mw1_A) if self.lora_rank else self.markov_w1
        return w1[ids]

    def _mw2_T(self, dtype):
        w2 = self.markov_w2 + (self.lora_mw2_B @ self.lora_mw2_A) if self.lora_rank else self.markov_w2
        return w2.T.to(dtype)

    def _conf_proj(self, dtype):
        base = self.conf_proj
        if self.lora_rank:
            delta = (self.lora_conf_B @ self.lora_conf_A).squeeze(0)  # [DIM + rank_m]
            base = base + delta.to(base.dtype)
        return base.to(dtype)

    def hc_head(self, x):  # x [b, BLOCK, HC, DIM] -> [b, BLOCK, DIM]
        # F32-internal (faithful to the ds4's dspark_hc_head_one: rms_norm_no_weight +
        # matvec_f16 + sigmoid all run in F32) — avoids the F16 overflow on the large
        # body-output values (up to 1721 -> x^2 > 65504 -> F16 inf -> the rsqrt + the
        # matmul-result cast collapse). The weights are F16 in the GGUF; cast to F32 for
        # the compute, cast the output back to the input dtype.
        b, s, hc, d = x.shape
        out_dt = x.dtype
        xf = x.float()
        flat = xf.reshape(b, s, hc * d)
        rsqrt = 1.0 / torch.sqrt((flat * flat).mean(-1, keepdim=True) + NORM_EPS)
        mixes = (flat @ self._hc_fn().float().T) * rsqrt       # [b, BLOCK, HC]
        pre = torch.sigmoid(mixes * self.hc_scale[0] + self.hc_base) + HC_EPS
        return ((pre.unsqueeze(-1) * xf).sum(dim=2)).to(out_dt)  # [b, BLOCK, DIM]

    def k_scores(self, x, prev_tok):  # x [N,BLOCK,HC,DIM], prev_tok [N,K] -> [N,K,VOCAB] (teacher-forced, LoRA-applied)
        dev = self.lm_head.device; dt = self.lm_head.dtype
        h = self.hc_head(x.to(dev)).to(dev, dt)            # [N,BLOCK,DIM] (LoRA via _hc_fn)
        hf = h.float()  # rms-norm in F32 (faithful to ds4; avoids h^2 F16 overflow)
        hf = hf * (1.0 / torch.sqrt((hf * hf).mean(-1, keepdim=True) + NORM_EPS)) * self._norm_w().to(dev, torch.float32)
        h = hf.to(dt)
        base = h @ self.lm_head.T                            # [N,BLOCK,VOCAB]
        bias = self._mw1(prev_tok.to(dev)).to(dev, dt) @ self._mw2_T(dt)  # [N,K,VOCAB]
        return base + bias                                   # BLOCK==K -> [N,K,VOCAB]

    def p1_scores(self, x0, anchor):  # x0 [N,HC,DIM], anchor [N] -> [N,VOCAB] (LoRA-applied)
        dev = self.lm_head.device; dt = self.lm_head.dtype
        x = x0.to(dev).unsqueeze(1)                       # [N,1,HC,DIM]
        h = self.hc_head(x)[:, 0].to(dev, dt)             # [N,DIM] (uses _hc_fn with LoRA)
        # the rms-norm in F32 (avoids the h^2 F16 overflow when h is large)
        hf = h.float()
        hf = hf * (1.0 / torch.sqrt((hf * hf).mean(-1, keepdim=True) + NORM_EPS)) * self._norm_w().to(dev, torch.float32)
        h = hf.to(dt)
        base = h @ self.lm_head.T                          # [N,VOCAB]
        bias = self._mw1(anchor.to(dev)).to(dev, dt) @ self._mw2_T(dt)  # [N,VOCAB]
        return base + bias

    def confidence_logits(self, h, prev_tok):
        """Per-position confidence logits / scores from normalized hidden + Markov embedding.

        h        : [b, BLOCK, DIM] normalized hidden states
        prev_tok : [b, BLOCK] previous-token ids for each draft position
        returns  : (logits [b, BLOCK], scores [b, BLOCK])
        """
        dev = h.device
        dt = h.dtype
        emb = self._mw1(prev_tok.to(dev)).to(dev, dt)                     # [b, BLOCK, rank_m]
        conf_inp = torch.cat((h, emb), dim=-1)                            # [b, BLOCK, DIM+rank_m]
        logits = (conf_inp * self._conf_proj(dt).to(dev)).sum(dim=-1)     # [b, BLOCK]
        return logits, torch.sigmoid(logits)

    def forward(self, x, anchor, temp=1.0, return_conf=False):  # x [b,BLOCK,HC,DIM], anchor [b]
        dev = x.device
        dt = self.lm_head.dtype
        h = self.hc_head(x.to(dev)).to(dev, dt)
        hf = h.float()  # rms-norm in F32 (faithful to ds4; avoids h^2 F16 overflow)
        hf = hf * (1.0 / torch.sqrt((hf * hf).mean(-1, keepdim=True) + NORM_EPS)) * self._norm_w().to(dev, torch.float32)
        h = hf.to(dt)
        base = h @ self.lm_head.T                                # [b, BLOCK, VOCAB]
        b = x.shape[0]
        out = torch.zeros(b, BLOCK + 1, dtype=torch.long, device=dev)
        out[:, 0] = anchor.to(dev)
        prev_tok = torch.zeros(b, BLOCK, dtype=torch.long, device=dev)
        mw2_T = self._mw2_T(dt)                                  # [rank_m, VOCAB]
        for i in range(BLOCK):
            prev_tok[:, i] = out[:, i]
            emb = self._mw1(out[:, i]).to(dev, dt)               # [b, rank_m]
            li = (base[:, i] + emb @ mw2_T) / max(temp, 1e-5)
            out[:, i + 1] = li.argmax(-1)
        if return_conf:
            conf_logits, conf_scores = self.confidence_logits(h, prev_tok)
            return out, base, conf_logits, conf_scores
        return out, base


def build_head(dspark_path: str, target_path: str, device, lora_rank: int = 0,
               dtype=torch.float32) -> DrafterHead:
    _, T, _infos, _doff, _ = load_gguf_dense_only(dspark_path)
    hc_fn = T["mtp.2.hc_head_fn.weight"][0]
    hc_scale = T["mtp.2.hc_head_scale.weight"][0]
    hc_base = T["mtp.2.hc_head_base.weight"][0]
    norm_w = T["mtp.2.norm.weight"][0]
    mw1 = T["mtp.2.markov_head.markov_w1.weight"][0]
    mw2 = T["mtp.2.markov_head.markov_w2.weight"][0]
    conf_proj = T["mtp.2.confidence_head.proj.weight"][0]
    _, tinfos, tdoff = index_gguf(target_path)
    lm_head = read_tensor(target_path, tinfos, tdoff, "output.weight").astype(np.float32)
    head = DrafterHead(hc_fn, hc_scale, hc_base, norm_w, mw1, mw2, conf_proj, lm_head, lora_rank=lora_rank)
    return head.to(device=device, dtype=dtype)
