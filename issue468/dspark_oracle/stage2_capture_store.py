#!/usr/bin/env python3
"""Stage2CaptureStore — read DSpark target main_hidden + greedy token streams
directly from the Stage 2 sharded safetensors (few-file; no per-position .npy
or bundle dirs materialized).

Used by measure_acceptance_bundle's additive `store` path (Lead 03) to run the
acceptance harness over the 240-prompt Stage 2 corpus (temp=0, 128-tok) without
emitting per-prompt bundle dirs. Schema (per shards/index.json):
  main_hidden [N,12288] f16, token_ids [N] i32, prompt_idx [N] i32,
  pos_in_prompt [N] i32; prompts[]: {prompt_id, source, split, prompt_tokens,
  n_positions, pos_range [P, P+G-1]}.

Semantics (index note): main_hidden[p] is the POST-token hidden at absolute
position p; token_ids at pos_in_prompt=j is the target token at pos0+j;
drafter input = main_hidden[p], anchor = token_ids[p], predicts token[p+1..p+K].
This matches measure_acceptance_bundle's bundle convention exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


class Stage2CaptureStore:
    def __init__(self, shards_dir):
        # accept a single dir or a list of dirs (merged by unique prompt_id; Lead 3
        # combines Stage 2's 240 shards with newly-captured shards).
        dirs = [shards_dir] if not isinstance(shards_dir, (list, tuple)) else list(shards_dir)
        dirs = [Path(d) for d in dirs]
        idx0 = json.loads((dirs[0] / "index.json").read_text())
        self.shard_names = list(idx0["shards"])
        self.prompts = list(idx0["prompts"])
        self.shards_dir = dirs[0]
        # merge additional dirs (re-index prompt_idx to follow the first dir's prompts)
        base = len(self.prompts)
        for d in dirs[1:]:
            more = json.loads((d / "index.json").read_text())
            self.shard_names.extend(f"{d}/{s}" for s in more["shards"])
            for p in more["prompts"]:
                pp = dict(p)
                self.prompts.append(pp)
            base += len(more["prompts"])
        self.id_to_idx = {p["prompt_id"]: i for i, p in enumerate(self.prompts)}

        # load each dir's shards; remap its local prompt_idx to the merged position
        # via prompt_id (so prompt_idx stays correct across merged dirs).
        self._mh = {}; self._tok = {}
        for d in dirs:
            di = json.loads((d / "index.json").read_text())
            local_to_merged = {li: self.id_to_idx[p["prompt_id"]] for li, p in enumerate(di["prompts"])}
            mh_c, tk_c, pi_c, po_c = [], [], [], []
            for name in di["shards"]:
                dd = load_file(str(d / name))
                mh_c.append(dd["main_hidden"].astype(np.float32))
                tk_c.append(dd["token_ids"].astype(np.int64))
                pi_c.append(dd["prompt_idx"].astype(np.int64))
                po_c.append(dd["pos_in_prompt"].astype(np.int64))
            mh = np.concatenate(mh_c); tk = np.concatenate(tk_c)
            pi = np.concatenate(pi_c); po = np.concatenate(po_c)
            for k_local in range(len(di["prompts"])):
                sel = np.where(pi == k_local)[0]
                if sel.size == 0:
                    continue
                order = np.argsort(po[sel], kind="stable"); sel = sel[order]
                km = local_to_merged[k_local]
                self._mh[km] = mh[sel]
                self._tok[km] = tk[sel].tolist()

    def prompt_ids(self):
        return [p["prompt_id"] for p in self.prompts]

    def prompt_meta(self, prompt_id):
        return self.prompts[self.id_to_idx[prompt_id]]

    def target_tokens(self, prompt_id):
        return self._tok[self.id_to_idx[prompt_id]]

    def main_hidden_at(self, prompt_id, abs_pos):
        """Return main_hidden [12288] at absolute position abs_pos for prompt_id."""
        i = self.id_to_idx[prompt_id]
        meta = self.prompts[i]
        j = abs_pos - int(meta["prompt_tokens"])  # pos_in_prompt
        return self._mh[i][j]

    def n_positions(self, prompt_id):
        i = self.id_to_idx[prompt_id]
        return len(self._tok[i])
