# In-Memory Payload Handoff: Investigation

## Question

> Can the worker's KV cache be loaded directly from the wire as bytes arrive,
> eliminating the temp-file detour that the current snapshot framing uses?

## Short answer

Yes. The current path makes the KV shard hit `/tmp` four times per handoff
(once written and once read on each side, plus the wire transfer). The
streaming path eliminates the temp file entirely on each side, replacing it
with a fixed staging buffer that is reused for every chunk. For a 77-220 MB
payload this is roughly 0.5-1.5 GB of redundant data movement saved per
handoff, plus removal of disk-flush latency on each side, plus bounded
per-handoff memory regardless of payload size. Phase 5 should keep the
staging size as a compile-time constant, not a user-visible tuning flag.

## Current data path

Reading the code in `ds4.c` and `ds4_distributed.c`:

**Coordinator (save) side**

`ds4_session_save_layer_payload(session, FILE *fp, ...)` calls
`payload_write_tensor_span(fp, gpu_tensor, offset, bytes, buf, chunk, ...)`
for each tensor. `payload_write_tensor_span`:

1. Reads `bytes` from `gpu_tensor` into `buf` using `ds4_gpu_tensor_read`.
2. Writes `buf` to `fp` with `fwrite`.

`dist_save_remote_shard_to_file()` then streams `fp` to the wire via
`dist_send_snapshot_file_chunks()`.

So the coordinator flow is:

```
GPU -> buf -> FILE (temp) -> [later] FILE -> buf -> socket
```

**Worker (load) side**

`dist_worker_handle_snapshot_load()` reads chunks from the wire into `buf`,
then `fwrite`s them to a temp file. After all chunks arrive it `fflush`es,
`fseeko`s to the start, and calls
`ds4_session_load_layer_payload(session, FILE *fp, ...)` which calls
`payload_read_tensor_span(fp, gpu_tensor, ...)`:

1. Reads `bytes` from `fp` into `buf` with `fread`.
2. Writes `buf` to `gpu_tensor` with `ds4_gpu_tensor_write`.

So the worker flow is:

```
socket -> buf -> FILE (temp) -> [later] FILE -> buf -> GPU
```

**Total IO per handoff for a 77 MB payload:**

- Coordinator: 77 MB GPU read + 77 MB temp file write + 77 MB temp file read + 77 MB wire send = ~308 MB of data movement, plus file flush latency
- Worker: 77 MB wire receive + 77 MB temp file write + 77 MB temp file read + 77 MB GPU write = ~308 MB of data movement, plus file flush latency

The temp file is hit twice on each side, with no functional reason.

For the README prompt (`ctx=16384`, 14,318 tokens, payload 221 MB) the
multipliers make the cost larger: ~884 MB of data movement per side.

## Proposed change: streaming payload with a fixed staging buffer

Add streaming variants of the save and load paths that operate on a
fixed reusable staging buffer instead of either a `FILE *` or a
full-payload in-memory buffer. The staging buffer is a compile-time
constant for Phase 5 and is the only memory that needs to be live
during a handoff. The data flows:

- Sender: GPU → staging buffer → wire chunk
- Receiver: wire chunk → staging buffer → GPU

The temp file is gone. The full-payload in-memory buffer is gone. The
single staging buffer is allocated once at the start of the transfer
and reused for every chunk.

### New functions in `ds4.c`

```c
typedef int (*ds4_payload_chunk_out_fn)(void *ctx,
                                        const uint8_t *buf, uint64_t bytes,
                                        char *err, size_t errlen);

int ds4_session_save_layer_payload_stream(
    ds4_session *s,
    ds4_payload_chunk_out_fn write_chunk,
    void *ctx,
    uint32_t layer_start,
    uint32_t layer_end,
    uint64_t *bytes_out,            // total payload size for the BEGIN frame
    char *err, size_t errlen);

typedef int (*ds4_payload_chunk_in_fn)(void *ctx,
                                       uint8_t *buf, uint64_t bytes,
                                       char *err, size_t errlen);

int ds4_session_load_layer_payload_stream(
    ds4_session *s,
    ds4_payload_chunk_in_fn read_chunk,
    void *ctx,
    uint64_t payload_bytes,
    const int *tokens,
    uint32_t n_tokens,
    uint32_t layer_start,
    uint32_t layer_end,
    char *err, size_t errlen);
```

Both functions walk the layer range in the same order as the existing
`ds4_session_save_layer_payload` / `ds4_session_load_layer_payload`
(header + per-layer `n_comp` / `n_index_comp` + raw cache rows +
compressed cache + state tensors + indexer tensors). For each tensor,
they transfer it in fixed-size staging chunks, calling the callback
once per chunk. The chunk callback is responsible for moving the bytes
to or from the wire.

The save function pre-computes `*bytes_out` (the total payload size)
by summing the per-layer sizes. This lets the caller put the value in
the `DS4_DIST_MSG_SNAPSHOT_BEGIN` frame before any data is sent.

The load function validates the header and per-layer metadata on the
first callback calls, then walks the layer range and writes to GPU.
Validation (token hash, model id, layer range, ctx, prefill_cap) all
happens before the first GPU write, exactly as today.

### Internal refactor

`payload_write_tensor_span(fp, ...)` and
`payload_read_tensor_span(fp, ...)` are replaced by
`payload_chunked_read(gpu_tensor, offset, n, staging_buf, ...)` and
`payload_chunked_write(gpu_tensor, offset, n, staging_buf, ...)`. The
chunked `ds4_gpu_tensor_read` / `ds4_gpu_tensor_write` calls are
unchanged; only the `FILE *` I/O around them moves to a streaming
callback.

The header / per-layer metadata writes (`payload_write_u32`,
`payload_read_u32`) also become `payload_chunked_write_u32` /
`payload_chunked_read_u32` that use the same callback.

The existing `FILE *`-based `ds4_session_save_layer_payload` /
`ds4_session_load_layer_payload` are re-implemented as thin wrappers
that provide a `FILE *`-based chunk callback. The local save/load tests
in `ds4_test.c` continue to work without changes.

### Wire code

The coordinator side change:

- Compute `payload_bytes` by calling `ds4_session_save_layer_payload_stream`
  with a counting callback, or by adding a pure
  `ds4_session_layer_payload_size()` helper.
- Send `DS4_DIST_MSG_SNAPSHOT_BEGIN` with the computed size.
- Call `ds4_session_save_layer_payload_stream` for real. The
  `write_chunk` callback sends `DS4_DIST_MSG_SNAPSHOT_CHUNK` frames.
  `DS4_DIST_SNAPSHOT_CHUNK_BYTES` is the natural compile-time staging
  size unless implementation constraints require a smaller internal
  buffer.

The worker side change:

- Receive `DS4_DIST_MSG_SNAPSHOT_BEGIN`. Validate token count, model
  id, layer range, ctx, prefill_cap.
- Receive `DS4_DIST_MSG_SNAPSHOT_CHUNK` frames into the wire receive
  buffer (8 MB, unchanged).
- Call `ds4_session_load_layer_payload_stream` with a `read_chunk`
  callback that fills the staging buffer from the wire receive buffer,
  receiving more wire chunks as needed.
- After the load function returns, send `DS4_DIST_MSG_SNAPSHOT_DONE`.

The wire framing is unchanged. The only difference is the source/sink
of the chunk payload: data flows through the existing wire chunk buffer
and the fixed staging buffer instead of through a temp file.

### Memory cost

- Sender: 1 fixed staging buffer + 1 wire chunk buffer.
- Receiver: 1 fixed staging buffer + 1 wire chunk buffer.

Total per-handoff memory is bounded by the fixed buffers across both
processes regardless of payload size. For a 1M-token context the
savings versus the temp-file approach are not just disk IO but also
~6 GB of disk-resident temp files and the corresponding page cache
pressure.

### Configuration

Do not add `--dist-staging-bytes` in Phase 5. Use a compile-time
constant for the staging buffer, preferably aligned with
`DS4_DIST_SNAPSHOT_CHUNK_BYTES` unless code-level constraints require
a smaller internal buffer. Runtime tuning can be revisited after the
first performance artifacts show it matters.

### Why streaming from day one

A staging buffer is strictly better than a full-payload in-memory
buffer because:

- Bounded memory regardless of payload size.
- No risk of a failed `malloc(payload_size)` on a constrained device.
- The staging buffer is reusable for future pipelined KV return
  (Phase 6) — the same buffer can be passed across prefill chunks
  with no additional allocation.
- A compile-time staging size keeps the Phase 5 operator surface small.
- The implementation is only marginally more complex than the
  full-buffer version. The chunk callback abstraction is the same
  whether the callback sends to a wire or to a FILE*.

## Impact on Phase 5 plan

The Phase 5 plan currently treats in-memory payload handling as
"rejected for Phase 5; only revisit if temp-file staging becomes a
measurable bottleneck." This investigation shows the bottleneck is
already measurable for the target prompt sizes (200+ MB redundant
disk I/O on each side per handoff) and the change is small (two new
functions plus minor refactoring of the wire code).

**Recommendation: do the streaming refactor as part of Phase 5.** It
fits naturally into the new worker-pushed / HELLO-advertised path, and
it removes the only significant runtime cost of the current snapshot
framing. Phase 6 can then target pipelining or wire-to-GPU streaming
for further optimization.

## Code touchpoints

- `ds4.c`
  - Add `ds4_session_save_layer_payload_stream()` and
    `ds4_session_load_layer_payload_stream()` with chunk callbacks.
  - Refactor `payload_write_tensor_span` / `payload_read_tensor_span`
    into chunked GPU read / GPU write helpers that take a staging
    buffer and a chunk callback. Keep the `FILE *`-based path as a
    thin wrapper that provides a `FILE *`-based chunk callback.
  - The header / per-layer metadata read / write also moves to the
    chunk callback.
- `ds4_distributed.c`
  - Coordinator: replace `dist_tmpfile_or_err(...)` +
    `ds4_session_save_layer_payload(fp, ...)` with a fixed staging
    buffer and `ds4_session_save_layer_payload_stream(...)`. The
    `write_chunk` callback sends a `DS4_DIST_MSG_SNAPSHOT_CHUNK` per
    staging buffer fill.
  - Worker: replace the chunked `fwrite` to a temp file and
    `ds4_session_load_layer_payload(tmp, ...)` with the streaming
    load function. The `read_chunk` callback receives the next wire
    chunk and copies bytes into the staging buffer.
  - Wire framing (`DS4_DIST_MSG_SNAPSHOT_*`) is unchanged.

## Validation

- Same-pass comparison: existing `tests/issue304_phase4_handoff` and
  `tests/issue304_phase4_diagnose` runs must produce identical token
  sequences with the streaming path.
- Timing: the streaming path should be measurably faster than the
  temp-file path for prompts that produce > 50 MB payloads, and at
  least no slower for small payloads.
- Memory: per-handoff memory growth is bounded by the fixed staging
  buffer plus the existing wire chunk buffer. No temp file is created.
- Failure cases: token-hash mismatch, layer-range mismatch, model-id
  mismatch must still reject the handoff before any GPU write happens.
- No staging-size CLI: verify Phase 5 does not add
  `--dist-staging-bytes`; staging-size tuning is deferred.
