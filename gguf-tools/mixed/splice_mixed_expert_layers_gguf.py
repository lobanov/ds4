#!/usr/bin/env python3
"""Build a mixed DeepSeek V4 Flash GGUF by splicing routed-expert layers.

The base GGUF supplies metadata and all tensors by default.  For selected layer
IDs, this copies the routed expert tensors from a donor GGUF, rewriting the GGUF
tensor directory and streaming tensor payloads without dequantizing or
requantizing.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


GGUF_HEADER_SIZE = 24
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

GGUF_SCALAR_SIZES = {
    GGUF_VALUE_UINT8: 1,
    GGUF_VALUE_INT8: 1,
    GGUF_VALUE_UINT16: 2,
    GGUF_VALUE_INT16: 2,
    GGUF_VALUE_UINT32: 4,
    GGUF_VALUE_INT32: 4,
    GGUF_VALUE_FLOAT32: 4,
    GGUF_VALUE_BOOL: 1,
    GGUF_VALUE_UINT64: 8,
    GGUF_VALUE_INT64: 8,
    GGUF_VALUE_FLOAT64: 8,
}

# GGML quant type -> (block elements, bytes per block, display name).
# This intentionally includes the formats used by the current DeepSeek V4 Flash
# GGUFs.  Add entries here if future recipes introduce new tensor types.
GGML_QUANT_SIZES = {
    0: (1, 4, "F32"),
    1: (1, 2, "F16"),
    30: (1, 2, "BF16"),
    8: (32, 34, "Q8_0"),
    10: (256, 84, "Q2_K"),
    12: (256, 144, "Q4_K"),
    16: (256, 66, "IQ2_XXS"),
    26: (1, 4, "I32"),
}

EXPERT_TENSOR_RE = re.compile(r"^(?:blk|mtp)\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    rel_offset: int
    data_offset: int
    n_bytes: int


@dataclass(frozen=True)
class GGUFInfo:
    path: Path
    version: int
    tensor_count: int
    kv_count: int
    kv_blob: bytes
    alignment: int
    tensors: list[TensorInfo]
    tensor_by_name: dict[str, TensorInfo]


@dataclass(frozen=True)
class SplicePlan:
    name: str
    source: str
    tensor: TensorInfo
    new_rel_offset: int
    donor_tensor: TensorInfo | None = None
    donor_block_ids: tuple[int, ...] = ()
    donor_expert_ids: tuple[int, ...] = ()


def read_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def read_u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def pad_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def read_exact(src: BinaryIO, n_bytes: int) -> bytes:
    data = src.read(n_bytes)
    if len(data) != n_bytes:
        raise EOFError("short read while parsing GGUF")
    return data


def read_u32_file(src: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(src, 4))[0]


def read_u64_file(src: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(src, 8))[0]


def skip_gguf_string_file(src: BinaryIO) -> None:
    n = read_u64_file(src)
    src.seek(n, os.SEEK_CUR)


def read_gguf_string_file(src: BinaryIO) -> str:
    n = read_u64_file(src)
    return read_exact(src, n).decode("utf-8")


def skip_value_payload_file(src: BinaryIO, value_type: int) -> None:
    if value_type == GGUF_VALUE_STRING:
        skip_gguf_string_file(src)
        return
    if value_type == GGUF_VALUE_ARRAY:
        subtype = read_u32_file(src)
        count = read_u64_file(src)
        if subtype == GGUF_VALUE_STRING:
            for _ in range(count):
                skip_gguf_string_file(src)
            return
        if subtype == GGUF_VALUE_ARRAY:
            for _ in range(count):
                skip_value_payload_file(src, subtype)
            return
        size = GGUF_SCALAR_SIZES.get(subtype)
        if size is None:
            raise ValueError(f"unsupported GGUF array subtype {subtype}")
        src.seek(count * size, os.SEEK_CUR)
        return
    size = GGUF_SCALAR_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    src.seek(size, os.SEEK_CUR)


def tensor_nbytes(dims: tuple[int, ...], ggml_type: int) -> int:
    q = GGML_QUANT_SIZES.get(ggml_type)
    if q is None:
        raise ValueError(f"unsupported GGML tensor type {ggml_type}; add it to GGML_QUANT_SIZES")
    block_elems, block_bytes, _name = q
    n_elems = 1
    for dim in dims:
        n_elems *= dim
    if n_elems % block_elems != 0:
        raise ValueError(f"tensor element count {n_elems} is not divisible by block size {block_elems}")
    return (n_elems // block_elems) * block_bytes


def parse_gguf(path: Path) -> GGUFInfo:
    with path.open("rb") as src:
        if read_exact(src, 4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = read_u32_file(src)
        tensor_count = read_u64_file(src)
        kv_count = read_u64_file(src)

        alignment = GGUF_DEFAULT_ALIGNMENT
        for _ in range(kv_count):
            key = read_gguf_string_file(src)
            value_type = read_u32_file(src)
            value_start = src.tell()
            skip_value_payload_file(src, value_type)
            value_end = src.tell()
            if key == "general.alignment" and value_type == GGUF_VALUE_UINT32:
                src.seek(value_start)
                alignment = read_u32_file(src)
                src.seek(value_end)

        kv_end = src.tell()
        src.seek(GGUF_HEADER_SIZE)
        kv_blob = read_exact(src, kv_end - GGUF_HEADER_SIZE)
        src.seek(kv_end)

        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = read_gguf_string_file(src)
            n_dims = read_u32_file(src)
            dims = tuple(read_u64_file(src) for _ in range(n_dims))
            ggml_type = read_u32_file(src)
            rel_offset = read_u64_file(src)
            raw_tensors.append((name, dims, ggml_type, rel_offset))

        data_start = pad_to(src.tell(), alignment)
    tensors: list[TensorInfo] = []
    for name, dims, ggml_type, rel_offset in raw_tensors:
        n_bytes = tensor_nbytes(dims, ggml_type)
        tensors.append(TensorInfo(name, dims, ggml_type, rel_offset, data_start + rel_offset, n_bytes))

    return GGUFInfo(
        path=path,
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        kv_blob=kv_blob,
        alignment=alignment,
        tensors=tensors,
        tensor_by_name={t.name: t for t in tensors},
    )


def parse_layer_set(spec: str) -> set[int]:
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"bad descending range {part!r}")
            layers.update(range(lo, hi + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError("no layers selected")
    return layers


def parse_selection_specs(spec: str) -> dict[int, set[str] | None]:
    allowed_parts = {"gate", "up", "down"}
    selections: dict[int, set[str] | None] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            layer_s, tensor_part = part.split(":", 1)
            layer = int(layer_s)
            if tensor_part not in allowed_parts:
                raise ValueError(
                    f"unsupported tensor part {tensor_part!r}; expected one of {sorted(allowed_parts)}"
                )
            chosen = selections.setdefault(layer, set())
            if chosen is None:
                continue
            chosen.add(tensor_part)
        else:
            layer = int(part)
            selections[layer] = None
    if not selections:
        raise ValueError("no routed-expert selections provided")
    return selections


def should_take_donor(name: str, selections: dict[int, set[str] | None]) -> bool:
    match = EXPERT_TENSOR_RE.match(name)
    if match is None:
        return False
    layer = int(match.group(1))
    tensor_part = match.group(2)
    chosen = selections.get(layer)
    if chosen is None:
        return layer in selections
    return tensor_part in chosen


def parse_block_delta_specs(specs: list[str] | None) -> dict[str, int]:
    parsed: dict[str, int] = {}
    if not specs:
        return parsed
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"bad --block-delta-top spec {spec!r}; expected <tensor_name>:<count>")
        name, count_s = spec.rsplit(":", 1)
        name = name.strip()
        count = int(count_s)
        if not name or count <= 0:
            raise ValueError(f"bad --block-delta-top spec {spec!r}")
        parsed[name] = count
    return parsed


def parse_expert_specs(specs: list[str] | None) -> dict[str, tuple[int, ...]]:
    parsed: dict[str, tuple[int, ...]] = {}
    if not specs:
        return parsed
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"bad --expert-select spec {spec!r}; expected <tensor_name>:id,id,...")
        name, ids_s = spec.split(":", 1)
        ids = tuple(sorted({int(part) for part in ids_s.split(",") if part.strip()}))
        if not name.strip() or not ids:
            raise ValueError(f"bad --expert-select spec {spec!r}")
        parsed[name.strip()] = ids
    return parsed


def qtype_name(ggml_type: int) -> str:
    return GGML_QUANT_SIZES.get(ggml_type, (0, 0, f"type_{ggml_type}"))[2]


def top_delta_block_ids(base: GGUFInfo, donor: GGUFInfo, tensor_name: str, top_n: int) -> tuple[int, ...]:
    base_tensor = base.tensor_by_name.get(tensor_name)
    donor_tensor = donor.tensor_by_name.get(tensor_name)
    if base_tensor is None or donor_tensor is None:
        raise ValueError(f"missing tensor for block delta selection: {tensor_name}")
    if base_tensor.ggml_type != donor_tensor.ggml_type:
        raise ValueError(f"type mismatch for {tensor_name}")
    q = GGML_QUANT_SIZES.get(base_tensor.ggml_type)
    if q is None:
        raise ValueError(f"unsupported block-delta type for {tensor_name}: {base_tensor.ggml_type}")
    _block_elems, block_bytes, qname = q
    if qname != "Q4_K":
        raise ValueError(f"--block-delta-top currently supports Q4_K only, got {qname} for {tensor_name}")
    n_blocks = base_tensor.n_bytes // block_bytes
    scores: list[tuple[int, int]] = []
    with base.path.open("rb") as base_file, donor.path.open("rb") as donor_file:
        base_file.seek(base_tensor.data_offset)
        donor_file.seek(donor_tensor.data_offset)
        for block_id in range(n_blocks):
            b = read_exact(base_file, block_bytes)
            d = read_exact(donor_file, block_bytes)
            diff_l1 = sum(abs(x - y) for x, y in zip(b, d))
            scores.append((diff_l1, block_id))
    scores.sort(reverse=True)
    return tuple(sorted(block_id for _score, block_id in scores[:top_n]))


def expert_slice_bytes(tensor: TensorInfo) -> tuple[int, int]:
    if len(tensor.dims) != 3:
        raise ValueError(f"tensor {tensor.name} is not a 3D expert tensor")
    in_dim, out_dim, n_exp = tensor.dims
    elems_per_exp = in_dim * out_dim
    q = GGML_QUANT_SIZES.get(tensor.ggml_type)
    if q is None:
        raise ValueError(f"unsupported expert tensor type for {tensor.name}: {tensor.ggml_type}")
    block_elems, block_bytes, _name = q
    if elems_per_exp % block_elems != 0:
        raise ValueError(f"expert slice for {tensor.name} is not block-aligned")
    return n_exp, (elems_per_exp // block_elems) * block_bytes


def build_plan(
    base: GGUFInfo,
    donor: GGUFInfo,
    selections: dict[int, set[str] | None],
    block_delta_specs: dict[str, int],
    expert_specs: dict[str, tuple[int, ...]],
) -> list[SplicePlan]:
    if base.version != donor.version:
        raise ValueError(f"GGUF version mismatch: base={base.version} donor={donor.version}")
    if base.tensor_count != donor.tensor_count:
        raise ValueError(f"tensor count mismatch: base={base.tensor_count} donor={donor.tensor_count}")
    if base.alignment != donor.alignment:
        raise ValueError(f"alignment mismatch: base={base.alignment} donor={donor.alignment}")

    plan: list[SplicePlan] = []
    next_rel = 0
    for base_tensor in base.tensors:
        donor_tensor = donor.tensor_by_name.get(base_tensor.name)
        if donor_tensor is None:
            raise ValueError(f"donor is missing tensor {base_tensor.name}")
        if base_tensor.dims != donor_tensor.dims:
            raise ValueError(f"shape mismatch for {base_tensor.name}: {base_tensor.dims} vs {donor_tensor.dims}")
        use_donor = should_take_donor(base_tensor.name, selections)
        donor_block_ids: tuple[int, ...] = ()
        donor_expert_ids: tuple[int, ...] = ()
        source = "base"
        source_tensor = base_tensor
        donor_tensor_ref: TensorInfo | None = None
        if use_donor:
            source = "donor"
            source_tensor = donor_tensor
        elif base_tensor.name in expert_specs:
            donor_expert_ids = expert_specs[base_tensor.name]
            source = "mixed_experts"
            donor_tensor_ref = donor_tensor
        elif base_tensor.name in block_delta_specs:
            donor_block_ids = top_delta_block_ids(base, donor, base_tensor.name, block_delta_specs[base_tensor.name])
            source = "mixed_blocks"
            donor_tensor_ref = donor_tensor
        plan.append(SplicePlan(
            name=base_tensor.name,
            source=source,
            tensor=source_tensor,
            new_rel_offset=next_rel,
            donor_tensor=donor_tensor_ref,
            donor_block_ids=donor_block_ids,
            donor_expert_ids=donor_expert_ids,
        ))
        next_rel += pad_to(source_tensor.n_bytes, base.alignment)
    return plan


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_tensor_info(out: BinaryIO, item: SplicePlan) -> None:
    t = item.tensor
    out.write(pack_string(item.name))
    out.write(struct.pack("<I", len(t.dims)))
    for dim in t.dims:
        out.write(struct.pack("<Q", dim))
    out.write(struct.pack("<I", t.ggml_type))
    out.write(struct.pack("<Q", item.new_rel_offset))


def copy_exact(src: BinaryIO, dst: BinaryIO, n_bytes: int, bufsize: int = 64 * 1024 * 1024) -> None:
    remaining = n_bytes
    while remaining:
        chunk = src.read(min(bufsize, remaining))
        if not chunk:
            raise EOFError("short read while copying tensor payload")
        dst.write(chunk)
        remaining -= len(chunk)


def write_padding(out: BinaryIO, n_bytes: int, alignment: int) -> None:
    pad = pad_to(n_bytes, alignment) - n_bytes
    if pad:
        out.write(b"\0" * pad)


def write_mixed(base: GGUFInfo, donor: GGUFInfo, plan: list[SplicePlan], out_path: Path, force: bool) -> None:
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; pass --force to overwrite")
    if tmp_path.exists():
        raise FileExistsError(f"temporary file already exists: {tmp_path}")

    total_payload = sum(item.tensor.n_bytes for item in plan)
    copied = 0
    next_report = 0

    with base.path.open("rb") as base_file, donor.path.open("rb") as donor_file, tmp_path.open("wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", base.version))
        out.write(struct.pack("<Q", base.tensor_count))
        out.write(struct.pack("<Q", base.kv_count))
        out.write(base.kv_blob)
        for item in plan:
            write_tensor_info(out, item)
        write_padding(out, out.tell(), base.alignment)

        for item in plan:
            if item.source == "mixed_experts":
                assert item.donor_tensor is not None
                n_exp, bytes_per_exp = expert_slice_bytes(item.tensor)
                if any(expert < 0 or expert >= n_exp for expert in item.donor_expert_ids):
                    raise ValueError(f"expert selection out of range for {item.name}")
                next_expert = 0
                base_file.seek(item.tensor.data_offset)
                donor_file.seek(item.donor_tensor.data_offset)
                for donor_expert in item.donor_expert_ids:
                    if donor_expert < next_expert:
                        continue
                    base_skip = (donor_expert - next_expert) * bytes_per_exp
                    if base_skip:
                        copy_exact(base_file, out, base_skip)
                        donor_file.seek(base_skip, os.SEEK_CUR)
                    donor_chunk = read_exact(donor_file, bytes_per_exp)
                    out.write(donor_chunk)
                    base_file.seek(bytes_per_exp, os.SEEK_CUR)
                    next_expert = donor_expert + 1
                remaining = item.tensor.n_bytes - next_expert * bytes_per_exp
                if remaining:
                    copy_exact(base_file, out, remaining)
                    donor_file.seek(remaining, os.SEEK_CUR)
            elif item.source == "mixed_blocks":
                assert item.donor_tensor is not None
                q = GGML_QUANT_SIZES[item.tensor.ggml_type]
                block_bytes = q[1]
                base_file.seek(item.tensor.data_offset)
                donor_file.seek(item.donor_tensor.data_offset)
                if item.tensor.n_bytes % block_bytes != 0:
                    raise ValueError(f"tensor {item.name} size is not block-aligned")
                next_block = 0
                for donor_block in item.donor_block_ids:
                    if donor_block < next_block:
                        continue
                    base_skip = (donor_block - next_block) * block_bytes
                    if base_skip:
                        copy_exact(base_file, out, base_skip)
                        donor_file.seek(base_skip, os.SEEK_CUR)
                    donor_chunk = read_exact(donor_file, block_bytes)
                    out.write(donor_chunk)
                    base_file.seek(block_bytes, os.SEEK_CUR)
                    next_block = donor_block + 1
                remaining = item.tensor.n_bytes - next_block * block_bytes
                if remaining:
                    copy_exact(base_file, out, remaining)
                    donor_file.seek(remaining, os.SEEK_CUR)
            else:
                src = donor_file if item.source == "donor" else base_file
                src.seek(item.tensor.data_offset)
                copy_exact(src, out, item.tensor.n_bytes)
            write_padding(out, item.tensor.n_bytes, base.alignment)
            copied += item.tensor.n_bytes
            pct = int(copied * 100 / total_payload) if total_payload else 100
            if pct >= next_report:
                print(f"copied {copied / (1024 ** 3):.2f} GiB / {total_payload / (1024 ** 3):.2f} GiB ({pct}%)", flush=True)
                next_report += 5

        out.flush()
        os.fsync(out.fileno())

    os.replace(tmp_path, out_path)


def summarize(base: GGUFInfo, donor: GGUFInfo, plan: list[SplicePlan]) -> None:
    selected = [item for item in plan if item.source == "donor"]
    block_selected = [item for item in plan if item.source == "mixed_blocks"]
    expert_selected = [item for item in plan if item.source == "mixed_experts"]
    base_bytes = sum(t.n_bytes for t in base.tensors)
    out_bytes = sum(item.tensor.n_bytes for item in plan)
    delta = out_bytes - base_bytes

    print(f"base:  {base.path}")
    print(f"donor: {donor.path}")
    print(f"selected donor tensors: {len(selected)}")
    by_type: dict[str, int] = {}
    for item in selected:
        by_type[qtype_name(item.tensor.ggml_type)] = by_type.get(qtype_name(item.tensor.ggml_type), 0) + 1
    print("selected donor types:", ", ".join(f"{name}:{count}" for name, count in sorted(by_type.items())) or "none")
    if expert_selected:
        for item in expert_selected:
            print(f"selected donor experts: {item.name}:{len(item.donor_expert_ids)}")
    if block_selected:
        for item in block_selected:
            print(f"selected donor blocks: {item.name}:{len(item.donor_block_ids)}")
    print(f"base tensor payload: {base_bytes:,} bytes ({base_bytes / (1024 ** 3):.2f} GiB)")
    print(f"mixed tensor payload: {out_bytes:,} bytes ({out_bytes / (1024 ** 3):.2f} GiB)")
    print(f"payload delta: {delta:,} bytes ({delta / (1024 ** 3):.2f} GiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Splice selected DeepSeek V4 Flash routed-expert layers from a donor GGUF.")
    parser.add_argument("--base", required=True, type=Path, help="base GGUF used for metadata and default tensors")
    parser.add_argument("--donor", required=True, type=Path, help="donor GGUF used for selected routed expert layers")
    parser.add_argument("--out", required=True, type=Path, help="output mixed GGUF")
    parser.add_argument("--q4-layers", help="comma-separated layer IDs/ranges to take from donor, e.g. 37-42")
    parser.add_argument(
        "--q4-select",
        help="comma-separated routed selections to take from donor, e.g. 2:down,2:gate or 37",
    )
    parser.add_argument(
        "--block-delta-top",
        action="append",
        help="copy only the top-N donor-vs-base Q4_K payload blocks for one tensor, format <tensor_name>:<count>",
    )
    parser.add_argument(
        "--expert-select",
        action="append",
        help="copy only selected expert slices from one routed tensor, format <tensor_name>:id,id,...",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing the output")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it already exists")
    args = parser.parse_args()

    if not any([args.q4_layers, args.q4_select, args.block_delta_top, args.expert_select]):
        parser.error("pass at least one selection mode")
    if args.q4_layers and args.q4_select:
        parser.error("--q4-layers and --q4-select are mutually exclusive")
    if args.q4_select:
        selections = parse_selection_specs(args.q4_select)
        print("q4 select:", args.q4_select)
    elif args.q4_layers:
        q4_layers = parse_layer_set(args.q4_layers)
        print("q4 layers:", ",".join(str(x) for x in sorted(q4_layers)))
        selections = {layer: None for layer in q4_layers}
    else:
        selections = {}
    block_delta_specs = parse_block_delta_specs(args.block_delta_top)
    for name, count in sorted(block_delta_specs.items()):
        print(f"block delta top: {name}:{count}")
    expert_specs = parse_expert_specs(args.expert_select)
    for name, ids in sorted(expert_specs.items()):
        print(f"expert select: {name}:{','.join(str(x) for x in ids)}")

    base = parse_gguf(args.base)
    donor = parse_gguf(args.donor)
    plan = build_plan(base, donor, selections, block_delta_specs, expert_specs)
    summarize(base, donor, plan)

    if args.dry_run:
        return 0

    write_mixed(base, donor, plan, args.out, args.force)
    final_size = args.out.stat().st_size
    print(f"wrote: {args.out}")
    print(f"file size: {final_size:,} bytes ({final_size / (1024 ** 3):.2f} GiB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
