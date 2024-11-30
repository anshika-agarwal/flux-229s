import torch
from einops import rearrange
from torch import Tensor
import math


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}, pe shape: {pe.shape}")

    heavy_hitters_ratio = 0.4

    q, k = apply_rope(q, k, pe)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    cumulative_attention = attention_weights.sum(dim=-2)  # aggregate across queries
    num_top_k = max(1, int(heavy_hitters_ratio * attention_weights.size(-1)))
    _, top_k_indices = torch.topk(cumulative_attention, k=num_top_k, largest=True, dim=-1)

    # print(f"Using heavy_hitters_ratio {heavy_hitters_ratio}, top-k: {num_top_k}")
    # print(f"Top-k indices: {top_k_indices[0, 0]}")

    k_topk = torch.gather(k, dim=2, index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, k.size(-1)))
    v_topk = torch.gather(v, dim=2, index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, v.size(-1)))

    scores_topk = torch.matmul(q, k_topk.transpose(-2, -1)) / math.sqrt(q.size(-1))
    attention_weights_topk = torch.nn.functional.softmax(scores_topk, dim=-1)

    # print(f"Reduced attention weights size: {attention_weights_topk.size()}")

    x = torch.matmul(attention_weights_topk, v_topk)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
