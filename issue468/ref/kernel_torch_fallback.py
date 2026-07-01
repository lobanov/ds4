from functools import lru_cache

import torch
import torch.nn.functional as F


FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def _round_pow2_scales(scale: torch.Tensor) -> torch.Tensor:
    scale = torch.clamp(scale, min=torch.finfo(torch.float32).tiny)
    return torch.pow(2.0, torch.ceil(torch.log2(scale)))


def _cast_scale(scale: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    if scale_dtype == torch.float8_e8m0fnu:
        return scale.to(torch.float8_e8m0fnu)
    return scale.to(scale_dtype)


def _restore_scale(scale: torch.Tensor) -> torch.Tensor:
    return scale.float()


def _reshape_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    n = x.size(-1)
    if n % block_size != 0:
        raise ValueError(f"last dimension {n} is not divisible by block_size={block_size}")
    return x.reshape(-1, n).view(-1, n // block_size, block_size)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    x_flat = x.reshape(-1, x.size(-1))
    blocks = _reshape_blocks(x, block_size).float()
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-4)
    scale = amax / FP8_MAX
    if scale_fmt is not None:
        scale = _round_pow2_scales(scale)
    scale_out = _cast_scale(scale, scale_dtype)
    q = torch.clamp(blocks / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        dq = (q.float() * _restore_scale(scale_out).unsqueeze(-1)).reshape_as(x_flat).to(x.dtype)
        x.copy_(dq.reshape_as(x))
        return x
    return q.reshape_as(x_flat).reshape_as(x), scale_out.reshape(*x.shape[:-1], x.size(-1) // block_size)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    x_flat = x.reshape(-1, x.size(-1))
    blocks = _reshape_blocks(x, block_size).float()
    amax = blocks.abs().amax(dim=-1).clamp_min(6.0 * (2 ** -126))
    scale = _round_pow2_scales(amax / FP4_MAX)
    scale_out = scale.to(torch.float8_e8m0fnu)

    x_scaled = blocks / scale.unsqueeze(-1)
    table = FP4_TABLE.to(x.device)
    table_view = table.view(1, 1, 1, -1)
    diffs = (x_scaled.unsqueeze(-1) - table_view).abs()
    idx = diffs.argmin(dim=-1).to(torch.uint8)
    low = idx[..., 0::2]
    high = idx[..., 1::2]
    packed = (low | (high << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    if inplace:
        unpacked = _unpack_fp4(packed).view_as(blocks)
        dq = (unpacked * _restore_scale(scale_out).unsqueeze(-1)).reshape_as(x_flat).to(x.dtype)
        x.copy_(dq.reshape_as(x))
        return x
    return packed.reshape(*x.shape[:-1], x.size(-1) // 2), scale_out.reshape(*x.shape[:-1], x.size(-1) // block_size)


@lru_cache(maxsize=16)
def _expanded_fp8_weight_shape(shape: tuple[int, int], scale_shape: tuple[int, int]) -> tuple[int, int]:
    if scale_shape[0] * 128 < shape[0] or scale_shape[1] * 128 < shape[1]:
        raise ValueError(f"scale shape {scale_shape} is incompatible with weight shape {shape}")
    return shape


def _expand_fp8_scales(scale: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    _expanded_fp8_weight_shape((out_features, in_features), tuple(scale.shape))
    expanded = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return expanded[:out_features, :in_features]


def _unpack_fp4(weight: torch.Tensor) -> torch.Tensor:
    raw = weight.view(torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    table = FP4_TABLE.to(weight.device)
    vals = torch.stack([table[low], table[high]], dim=-1)
    return vals.flatten(-2)


def _expand_fp4_scales(scale: torch.Tensor, in_features: int) -> torch.Tensor:
    expanded = scale.float().repeat_interleave(32, dim=1)
    return expanded[:, :in_features]


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    del scale_dtype
    a_flat = a.reshape(-1, a.size(-1))
    a_scale = _restore_scale(a_s.reshape(-1, a_s.size(-1))).repeat_interleave(128, dim=1)[:, : a.size(-1)]
    a_dq = a_flat.float() * a_scale
    b_scale = _expand_fp8_scales(b_s, b.size(0), b.size(1))
    b_dq = b.float() * b_scale
    out = F.linear(a_dq, b_dq)
    return out.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    del scale_dtype
    a_flat = a.reshape(-1, a.size(-1))
    a_scale = _restore_scale(a_s.reshape(-1, a_s.size(-1))).repeat_interleave(128, dim=1)[:, : a.size(-1)]
    a_dq = a_flat.float() * a_scale
    b_dq = _unpack_fp4(b)
    b_scale = _expand_fp4_scales(b_s, b_dq.size(1))
    out = F.linear(a_dq, b_dq.float() * b_scale)
    return out.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    bsz, seqlen, n_heads, head_dim = q.shape
    valid = topk_idxs >= 0
    gather = topk_idxs.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kv_expanded = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)
    gathered = torch.gather(kv_expanded, 2, gather)
    gathered = gathered.masked_fill(~valid.unsqueeze(-1), 0)

    scores = torch.einsum("bshd,bstd->bsht", q.float(), gathered.float()).mul_(softmax_scale)
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    finite_scores = torch.where(valid.unsqueeze(2), scores, torch.full_like(scores, float("-inf")))
    max_score = finite_scores.amax(dim=-1)
    sink_max = torch.maximum(max_score, attn_sink.view(1, 1, -1))
    numer = torch.exp(finite_scores - sink_max.unsqueeze(-1)) * valid.unsqueeze(2)
    denom = numer.sum(dim=-1) + torch.exp(attn_sink.view(1, 1, -1) - sink_max)
    out = torch.einsum("bsht,bstd->bshd", numer, gathered.float()) / denom.unsqueeze(-1)
    return out.to(q.dtype)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
