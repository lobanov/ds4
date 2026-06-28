#!/usr/bin/env bash
# A small, deterministic coding prompt (HumanEval-style). Used as a fixed
# input for baseline runs so MTP draft acceptance is reproducible across
# commits. Keep this file stable; edit only to add more tasks at the end.
set -euo pipefail
cat <<'PROMPT'
Complete the following Python functions. Output only code.

def fib(n):
    """Return the n-th Fibonacci number, with fib(0)=0 and fib(1)=1."""

def is_prime(n):
    """Return True if n is a prime number, else False."""

def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""

def flatten(nested):
    """Flatten an arbitrarily nested list of ints into a flat list of ints."""

def group_by(items, key):
    """Group items by the value returned by key(item), preserving first-seen order."""

Now write a short paragraph explaining when you would use a Merkle tree.
PROMPT
