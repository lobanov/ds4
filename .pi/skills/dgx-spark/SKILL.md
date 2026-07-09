---
name: dgx-spark
description: Use the local DGX Spark (ssh dgx-direct) — an NVIDIA GB10 Grace-Blackwell, aarch64, ~121 GB unified memory. Use for FREE local GPU iteration (debugging instrumentation, small/medium models, source reading) before paid cloud runs; know its ~118 GB memory ceiling and unified-memory caveats.
---

# Local DGX Spark — access, profile, and when to use it

The DGX Spark (`ssh dgx-direct`) is a free, local NVIDIA **GB10 Grace-Blackwell** box.
It is the right place to iterate for free before paying for cloud GPUs — but it has a
hard ~118 GB memory ceiling and a unified-memory model that changes the usual playbook.

## When to use

- **Free local GPU iteration** before any paid cloud (Modal) run: debugging
  instrumentation (hooks, patches, capture scripts), small/medium model tests, reading
  installed engine source.
- The target model **fits in ~118 GB resident** (e.g. a ~≤100 GB checkpoint), or the
  engine has **SSD/expert streaming** (e.g. ds4 `--ssd-streaming`) that pages the rest.
- You need **aarch64 + CUDA 13** specifically (the GB10).

## Hardware profile (verify on connect — it's a shared box)

- **SoC:** NVIDIA GB10 Grace-Blackwell, **aarch64** (ARM), 20 cores.
- **Memory:** ~121 GB **unified** (CPU + GPU share one pool), ~118 GB available. The GPU
  has no separate VRAM — `nvidia-smi` reports memory as `[N/A]`; use **`free -g`** for
  the real pool.
- **CUDA:** 13.0 (`nvcc`); driver ~580.
- **Disk:** ~3.7 TB NVMe (~3.3 TB free) — plenty for large model caches.
- Recon one-liner:
  ```
  ssh dgx-direct 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader; free -g | head -2; df -h / | tail -1; nproc'
  ```

## Access patterns

- **Profile:** `ssh dgx-direct` (BatchMode OK; `ssh -o ConnectTimeout=15 dgx-direct ...`).
- **Multi-line scripts:** use a **heredoc** — `ssh dgx-direct 'bash -s' <<'REMOTE' ...
  REMOTE`. Do **not** use `bash -lc '...'` with nested single-quotes (it breaks on `(`,
  etc.). Quote the heredoc delimiter (`<<'REMOTE'`) to prevent local expansion.
- **Long tasks** (model downloads, big pip installs): background with `nohup ... >
  ~/log 2>&1 &` inside the ssh, then **poll the log** from subsequent calls (a single
  ssh call should not block for a 30-min download).
- **Files to/from:** `scp dgx-direct:~/path ./` and `ssh dgx-direct "cat > ~/f" < localf`
  both work for staging scripts/prompts.

## Core lessons

1. **~118 GB is a hard resident ceiling; there is no TP.** It's one GPU. A model larger
   than ~118 GB **does not fit resident** (e.g. a 129 GB checkpoint OOMs). vLLM
   **eager-loads** with no engine-level SSD/expert streaming → such a model can't run in
   vLLM here. Either use a smaller/quantized checkpoint that fits, or an engine with
   genuine streaming (ds4 `--ssd-streaming` pages routed experts from the 3.3 TB disk).

2. **Unified memory defeats CPU-offload tricks.** Because CPU and GPU share one pool,
   `--cpu-offload-gb` (and any "offload weights to host RAM" knob) gives **no extra
   capacity** — host RAM *is* the GPU pool. Don't reach for offload to fit a too-big
   model here; it won't help. (It only helps on discrete-GPU hosts with separate host RAM.)

3. **aarch64 wheels are required (and mostly available).** `pip install vllm` resolves an
  aarch64 wheel (`manylinux_2_28_aarch64`) and `torch+cu130` aarch64 exist. But **check
  other libs** — not every package ships aarch64 wheels; some build from source slowly or
  fail. Probe with `pip install --dry-run <pkg>` before committing.

4. **It's a shared/dev box — inventory before you install/download.** Other projects live
   here (home dir has multiple venvs, model dirs, download scripts, an HF token). Before
   re-downloading a model or installing vLLM: `ls ~`, `find ~ -maxdepth N -iname
   '*.gguf' -o -iname '*.safetensors'`, check existing venvs (`~/<venv>/bin/python -c
   'import vllm,torch; ...'`). **Use an isolated venv** for your install — vLLM pins its
   own torch (e.g. 2.11) and will downgrade/clobber a shared venv's torch (e.g. 2.12),
   breaking other tools there.

5. **Its superpower is FREE iteration + co-located reference data.** Use it to (a) read
   the installed engine source for free, (b) debug capture/instrumentation scripts on a
   small loadable model, (c) validate the V1/deployment mechanics — all before a paid
   cloud run. A ~90 GB model that fits here can be iterated on for $0 instead of $/hr.

6. **Downloads are ~25-40 MB/s.** A ~90 GB model is ~40-60 min; a ~130 GB model ~60-90
   min. Always resume-friendly (`huggingface_hub.snapshot_download` with
   `allow_patterns`; or `curl -C -`). Don't redownload what's cached.

## Workflow (free-first, before cloud)

1. **Inventory** the box: GPU/mem/disk, existing venvs + models + tokens (don't
   re-download/install what's there).
2. **Does the target fit ~118 GB?** If yes → iterate here. If no, and the engine has SSD
   streaming → try it. If no streaming → this box can't run it; go to cloud (TP≥2).
3. **Isolated venv** (`python3 -m venv ~/myvenv`) for your deps; `pip install --dry-run`
   first to confirm aarch64 wheels.
4. **Read the installed engine source** here (free) — don't fight your local Mac for an
   engine that isn't installed locally.
5. **Debug instrumentation on a small loadable model** here (e.g. a ~few-GB model) to
   prove the capture/deployment mechanism, then take the validated script to cloud for
   the big model.

## Guardrails

- **Never assume a >~118 GB model runs here** (no TP, no useful offload, vLLM doesn't
  stream) — it OOMs.
- **Never install into a shared venv** if it pins a different torch/deps — use an
  isolated venv.
- **Never block one ssh call on a long download/install** — background + log + poll.
- **Never assume a package has an aarch64 wheel** — `--dry-run` first.
- **Never use `bash -lc '...'` for multi-line ssh scripts** — use a quoted heredoc.

## Anti-patterns (from experience)

- **Loading a 129 GB native model in vLLM on the GB10** → OOM (no TP, no streaming,
  offload useless on unified memory).
- **`pip install vllm` into a shared venv** → downgrades its torch, breaks the other tool.
- **`ssh dgx-direct 'bash -lc "for ... ( ..."'`** → `syntax error near unexpected token '('`
  (nested-quote breakage) — use `bash -s <<'REMOTE'`.
- **Re-downloading an 87 GB model** that was already in `~/.../gguf/` → wasted an hour;
  inventory first.
- **Blocking a single tool call on a 60-min download** → timeout/cancel; background it.
