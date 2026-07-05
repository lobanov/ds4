# Retained numpy oracle

## Summary

The DSpark numpy oracle has been copied into the research worktree under `issue468/dspark_oracle/`, given a local virtual environment, and verified to remain operational for retained smoke validation.

## What was retained

- `issue468/dspark_oracle/attention.py`
- `issue468/dspark_oracle/expert_store.py`
- `issue468/dspark_oracle/forward.py`
- `issue468/dspark_oracle/gguf_loader.py`
- `issue468/dspark_oracle/hc_primitives.py`
- `issue468/dspark_oracle/moe.py`
- `issue468/dspark_oracle/.gitignore`

## Local environment

A local venv was created at:

- `issue468/dspark_oracle/.venv/`

It is ignored and is not part of the retained git artifacts.

Installed and used for the smoke check:

- `numpy`

Retained environment spec:

- `issue468/dspark_oracle/requirements.txt`

## Operational check performed

From `issue468/dspark_oracle/`:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy
python attention.py
python hc_primitives.py
python moe.py
python forward.py --smoke
```

## Result

Operational check passed.

The retained oracle successfully completed:

- primitive self-tests in `attention.py`
- primitive self-tests in `hc_primitives.py`
- primitive self-tests in `moe.py`
- smoke forward run in `forward.py --smoke`

The smoke forward run loaded:

- `../../../ds4/gguf/dspark.gguf`
- `../../../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

and produced draft output ids and logits without error.

## Notes

Two small portability fixes were applied in the retained oracle copy so the smoke procedure works cleanly in the research worktree:

- `forward.py` now accepts `DSPARK_GGUF` and `TARGET_GGUF` overrides and uses research-worktree-relative defaults.
- self-test tensor shapes in `hc_primitives.py` and `moe.py` were corrected so their retained module-level smoke checks execute successfully.

These changes preserve the oracle's retained usability as a compact validation tool inside the dossier.
