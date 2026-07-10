#!/usr/bin/env python3
"""FREE test 1 — unit-test the buffer-patch CAPTURE LOGIC (numpy, no torch).

Verifies the parts most likely to harbor an off-by-one / wrong-axis bug:
  - slot_for_layer: only layers 40/41/42 map to slots 0/1/2 (given a start_layer offset).
  - capture_into_buffer: mhc_post -> flatten -> copy into buffer[:n, slot] correctly;
    only the captured layers populate; shape [n,3,16384].

This is the patch CODE, not the torch-hook plumbing / V1 deployment (those are free
test 2 on the DGX-Spark + the Modal decisive smoke).
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import dspark_hc_patch as P


def fake_mhc_post(hs, residual, post_mix, res_mix):
    """Stand-in for mhc_post_tilelang: returns hs unchanged (so buffer == hs.flatten)."""
    return hs


def fake_mhc_post_scaled(hs, residual, post_mix, res_mix):
    """Stand-in that depends on residual too (verifies we pass all 4 args correctly)."""
    return hs + 0.0 * residual  # identity-ish but touches residual


def test_slot_mapping():
    # The forward loop uses islice(layers, start_layer, end_layer); global idx = start+i.
    # Only global 40/41/42 capture. Verify the mapping for a few start_layer offsets.
    for start in (0, 1, 5):
        for i in range(43 - start):
            gidx = start + i
            s = P.slot_for_layer(gidx)
            if gidx == 40:
                assert s == 0, (start, gidx, s)
            elif gidx == 41:
                assert s == 1
            elif gidx == 42:
                assert s == 2
            else:
                assert s is None, (start, gidx, s)
    # explicit
    assert P.slot_for_layer(39) is None
    assert P.slot_for_layer(40) == 0
    assert P.slot_for_layer(41) == 1
    assert P.slot_for_layer(42) == 2
    assert P.slot_for_layer(43) is None
    print("[mock] slot_for_layer OK (only 40/41/42 -> slots 0/1/2)")


def test_capture_one_slot():
    n = 5
    buf = np.zeros((10, 3, P.HC_DIM), dtype=np.float32)
    hs = np.random.randn(n, 4, 4096).astype(np.float32)
    res = np.random.randn(n, 4, 4096).astype(np.float32)
    pm = np.random.randn(n, 4).astype(np.float32)
    rm = np.random.randn(n, 4, 4096).astype(np.float32)
    nn = P.capture_into_buffer(buf, 1, hs, res, pm, rm, fake_mhc_post)
    assert nn == n
    # slot 1 populated with hs.flatten (= mhc_post identity)
    assert buf[:n, 1].shape == (n, P.HC_DIM)
    assert np.allclose(buf[:n, 1], hs.reshape(n, -1)), "slot1 != hs.flatten"
    # other slots untouched
    assert np.all(buf[:, 0] == 0) and np.all(buf[:, 2] == 0)
    # beyond n untouched
    assert np.all(buf[n:, 1] == 0)
    print(f"[mock] capture_into_buffer OK (slot1 <- hs.flatten, shape [{n},{P.HC_DIM}], others 0)")


def test_simulate_loop():
    """Simulate the patched forward loop: 43 layers, capture only 40/41/42."""
    start_layer, end_layer = 0, 43
    max_tok = 8
    buf = np.zeros((max_tok, 3, P.HC_DIM), dtype=np.float32)
    n_seen = [0, 0, 0]
    # fake "forward": each layer produces a distinctive hs (layer-indexed) so we can tell
    # which layers were captured.
    for i in range(end_layer - start_layer):
        gidx = start_layer + i
        hs = np.full((3, 4, 4096), gidx, dtype=np.float32)  # value = global layer idx
        res = np.zeros_like(hs); pm = np.zeros((3, 4), dtype=np.float32); rm = np.zeros_like(hs)
        slot = P.slot_for_layer(gidx)
        if slot is not None:
            n_seen[slot] = P.capture_into_buffer(buf, slot, hs, res, pm, rm, fake_mhc_post)
    # only layers 40/41/42 captured -> buf values 40/41/42 in slots 0/1/2
    assert n_seen == [3, 3, 3]
    assert np.allclose(buf[:3, 0], 40.0) and np.allclose(buf[:3, 1], 41.0) and np.allclose(buf[:3, 2], 42.0)
    # beyond the 3 captured tokens untouched
    assert np.all(buf[3:, :] == 0)
    print("[mock] simulated loop OK (captured exactly layers 40/41/42 -> slots 0/1/2, values 40/41/42)")


if __name__ == "__main__":
    test_slot_mapping()
    test_capture_one_slot()
    test_simulate_loop()
    print("\n[mock] ALL FREE-TEST-1 CHECKS PASSED")
