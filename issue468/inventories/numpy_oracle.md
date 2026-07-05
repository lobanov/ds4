# Numpy Oracle Inventory

This inventory tracks the retained DSpark numpy oracle under `issue468/dspark_oracle/`.

## Purpose

The numpy oracle is the reference implementation used to validate DSpark drafter mechanics independently of the live Metal path.

It is retained because it supports:

- forward-shape and primitive correctness checks,
- tensor-layout and dequantization cross-check work,
- compact smoke validation of the retained oracle code,
- future reproduction of DSpark drafter investigations without reviving the full historical dossier.

## Location

- code: `issue468/dspark_oracle/`
- requirements: `issue468/dspark_oracle/requirements.txt`
- local virtualenv: `issue468/dspark_oracle/.venv/` (ignored)

## Expected operational checks

At minimum, the retained oracle should be able to run:

- module self-tests in `attention.py`
- module self-tests in `hc_primitives.py`
- module self-tests in `moe.py`
- `forward.py --smoke`

## Environment notes

The retained oracle uses a local Python virtual environment inside the oracle directory.
The venv is for local execution only and must not be committed.

Minimal recreation:

```sh
cd issue468/dspark_oracle
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Retention rule

Keep the oracle code and this inventory page active.
If future work changes the smoke procedure or dependencies, update this page and `issue468/STATUS.md`.
