"""Native FA4 varlen with state-prefix causal attention for decoder questions.

For a question of length Q and its state of length S, keys are [state, question].
FA4's bottom-right causal mask exposes S+i+1 keys to query i: the complete state
and exactly its own prefix. One kernel therefore computes the original joint
softmax and its complete gradient, without separate self/cross attention merges.
Active sliding windows remain in the exact Triton implementation.
"""

import torch


@torch.library.custom_op('bongard::flash_packed_forward', mutates_args=())
def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             cuq: torch.Tensor, cuk: torch.Tensor, scale: float, causal: bool
             ) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_fwd

    out, lse, _, _ = _flash_attn_fwd(
        q, k, v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
        softmax_scale=scale, causal=causal, return_lse=True)
    return out, lse


@_forward.register_fake
def _forward_fake(q, k, v, cuq, cuk, scale, causal):
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)


@torch.library.custom_op('bongard::flash_packed_backward', mutates_args=())
def _backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              cuq: torch.Tensor, cuk: torch.Tensor, out: torch.Tensor, lse: torch.Tensor,
              dout: torch.Tensor, scale: float, causal: bool
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_bwd

    return _flash_attn_bwd(
        q, k, v, out, dout.contiguous(), lse,
        cu_seqlens_q=cuq, cu_seqlens_k=cuk, softmax_scale=scale, causal=causal)[:3]


@_backward.register_fake
def _backward_fake(q, k, v, cuq, cuk, out, lse, dout, scale, causal):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _context(ctx, inputs, output):
    q, k, v, cuq, cuk, ctx.scale, ctx.causal = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, cuq, cuk, out, lse)
    ctx.mark_non_differentiable(lse)


def _grad(ctx, dout, ignored):
    return (*_backward(*ctx.saved_tensors, dout, ctx.scale, ctx.causal), None, None, None, None)


_forward.register_autograd(_grad, setup_context=_context)


def _prefix_keys(k, v, ck, cv, indices):
    return tuple(torch.cat((state, own), dim=0).index_select(0, indices)
                 for state, own in ((ck, k), (cv, v)))


@torch.library.custom_op('bongard::flash_prefix_forward', mutates_args=())
def _prefix(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            ck: torch.Tensor, cv: torch.Tensor, indices: torch.Tensor,
            cuq: torch.Tensor, cuk: torch.Tensor, scale: float
            ) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_fwd

    joined_k, joined_v = _prefix_keys(k, v, ck, cv, indices)
    out, lse, _, _ = _flash_attn_fwd(
        q, joined_k, joined_v, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
        softmax_scale=scale, causal=True, return_lse=True)
    return out, lse


@_prefix.register_fake
def _prefix_fake(q, k, v, ck, cv, indices, cuq, cuk, scale):
    return torch.empty_like(q), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)


@torch.library.custom_op('bongard::flash_prefix_backward', mutates_args=())
def _prefix_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     ck: torch.Tensor, cv: torch.Tensor, indices: torch.Tensor,
                     cuq: torch.Tensor, cuk: torch.Tensor, out: torch.Tensor, lse: torch.Tensor,
                     dout: torch.Tensor, scale: float
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_bwd

    joined_k, joined_v = _prefix_keys(k, v, ck, cv, indices)
    dq, dk, dv = _flash_attn_bwd(
        q, joined_k, joined_v, out, dout.contiguous(), lse,
        cu_seqlens_q=cuq, cu_seqlens_k=cuk, softmax_scale=scale, causal=True)[:3]
    # Keep the gather/scatter workspace in the activation dtype. Promoting the
    # whole expanded KV gradient to FP32 doubled the largest temporary buffer.
    gradients = []
    for expanded in (dk, dv):
        base = torch.zeros((ck.shape[0] + k.shape[0], *k.shape[1:]),
                           device=k.device, dtype=k.dtype)
        base.index_add_(0, indices, expanded)
        gradients.append((base[ck.shape[0]:].clone(), base[:ck.shape[0]].clone()))
    return dq, gradients[0][0], gradients[1][0], gradients[0][1], gradients[1][1]


@_prefix_backward.register_fake
def _prefix_backward_fake(q, k, v, ck, cv, indices, cuq, cuk, out, lse, dout, scale):
    return tuple(torch.empty_like(t) for t in (q, k, v, ck, cv))


def _prefix_context(ctx, inputs, output):
    *tensors, ctx.scale = inputs
    out, lse = output
    # Never retain state K/V expanded once per question in every decoder layer.
    # Rebuild only that gather in backward; preserve projections and FFN work.
    ctx.save_for_backward(*tensors, out, lse)
    ctx.mark_non_differentiable(lse)


def _prefix_grad(ctx, dout, ignored):
    return (*_prefix_backward(*ctx.saved_tensors, dout, ctx.scale), None, None, None, None)


_prefix.register_autograd(_prefix_grad, setup_context=_prefix_context)


def attention(q, k, v, layout, scale, window, cross=None):
    if window is not None:
        raise ValueError('FA4 HD256 requires an inactive local window')
    q, k, v = (t[0].transpose(0, 1).contiguous() for t in (q, k, v))
    if cross is not None:
        ck, cv = (state[0].transpose(0, 1).contiguous() for state in cross)
        out = _prefix(q, k, v, ck, cv, layout.key_indices,
                      layout.cu_seqlens, layout.key_seqlens, scale)[0]
    else:
        out = _forward(q, k, v, layout.cu_seqlens, layout.cu_seqlens, scale, False)[0]
    return out.transpose(0, 1)[None]
