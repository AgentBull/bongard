"""Unpadded record/question packs with native positions and attention visibility."""

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import accumulate

import torch
from torch.nn import functional as F


@lru_cache(maxsize=1)
def _block_mask_factory():
    from torch.nn.attention.flex_attention import create_block_mask

    return torch.compile(create_block_mask, fullgraph=True, dynamic=True)


@lru_cache(maxsize=1)
def _flex_option_overrides():
    """BONGARD_FLEX_OPTIONS='{"fwd_BLOCK_M": 128, ...}' tunes the FlexAttention tiles."""
    import json
    import os

    return json.loads(os.environ.get('BONGARD_FLEX_OPTIONS', '{}'))


@lru_cache(maxsize=1)
def _flex_block_size():
    """BONGARD_FLEX_BLOCK_SIZE=256 is required by the FlashAttention-4 flex backend."""
    import os

    return int(os.environ.get('BONGARD_FLEX_BLOCK_SIZE', '128'))


@lru_cache(maxsize=1)
def _attention_kernel():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, fullgraph=True, dynamic=True)


@dataclass
class PackedLayout:
    positions: torch.Tensor
    masks: dict
    backend: str = 'triton'
    cu_seqlens: torch.Tensor | None = None
    key_seqlens: torch.Tensor | None = None
    key_indices: torch.Tensor | None = None
    flash_windows: frozenset = frozenset()
    partitions: dict = field(default_factory=dict)


def packed_layout(lengths, windows, device, *, owners=None, state_lengths=None, backend='triton'):
    """Build once per pack; every layer and checkpoint replay shares the masks."""
    segments = torch.tensor(
        [index for index, length in enumerate(lengths) for _ in range(length)], device=device
    )
    positions = torch.tensor([p for length in lengths for p in range(length)], device=device)
    cu = key_cu = key_indices = None
    flash_windows = frozenset()
    partitions = {}
    if backend == 'flash':
        offsets = list(accumulate(lengths, initial=0))
        for window in set(windows) - {None}:
            limit = window if owners is not None else (window + 1) // 2
            short = [i for i, length in enumerate(lengths) if length <= limit]
            long = [i for i, length in enumerate(lengths) if length > limit]
            if short and long:
                partitions[window] = []
                for members in (short, long):
                    indices = torch.tensor([j for i in members for j in range(offsets[i], offsets[i + 1])],
                                           device=device)
                    child = packed_layout(
                        [lengths[i] for i in members], [window], device,
                        owners=[owners[i] for i in members] if owners is not None else None,
                        state_lengths=state_lengths, backend=backend,
                    )
                    partitions[window].append((indices, child))
    if backend == 'flash' and device.type == 'cuda':
        cu = torch.tensor(list(accumulate(lengths, initial=0)), dtype=torch.int32, device=device)
        # FA4's SM100 HD256 kernels lack local/custom masks. A window that
        # covers each complete sequence is exactly native dense/causal FA4.
        flash_windows = frozenset(w for w in windows if w is None or
                                  max(lengths) <= (w if owners is not None else (w + 1) // 2))
        if owners is not None:
            state_offsets = list(accumulate(state_lengths, initial=0))
            # Put the complete state before each question's own K/V. FA4's
            # bottom-right causal alignment then implements the original
            # joint self/state softmax in one attention call.
            query_offsets = list(accumulate(lengths, initial=0))
            state_total = sum(state_lengths)
            indices = []
            for index, owner in enumerate(owners):
                indices.extend(range(state_offsets[owner], state_offsets[owner + 1]))
                indices.extend(range(state_total + query_offsets[index],
                                     state_total + query_offsets[index + 1]))
            key_cu = torch.tensor(list(accumulate(
                (state_lengths[o] + length for o, length in zip(owners, lengths)), initial=0)),
                dtype=torch.int32, device=device)
            key_indices = torch.tensor(indices, device=device)
        if flash_windows == set(windows):
            return PackedLayout(positions[None], {}, backend, cu, key_cu, key_indices, flash_windows)
    query_length = len(segments)
    decoder = owners is not None
    if decoder:
        query_owners = torch.tensor(
            [owner for owner, length in zip(owners, lengths) for _ in range(length)], device=device
        )
        state_owners = torch.tensor(
            [index for index, length in enumerate(state_lengths) for _ in range(length)], device=device
        )
        key_length = query_length + len(state_owners)
    else:
        key_length = query_length
    def make_mask(window):
        def mask_mod(batch, head, query, key):
            qi = query.clamp(max=query_length - 1)
            ki = key.clamp(max=query_length - 1)
            distance = positions[qi] - positions[ki]
            visible = segments[qi] == segments[ki]
            if decoder:
                visible = visible & (distance >= 0)
                if window is not None:
                    visible = visible & (distance < window)
                state_index = (key - query_length).clamp(min=0, max=len(state_owners) - 1)
                visible = torch.where(
                    key < query_length, visible,
                    query_owners[qi] == state_owners[state_index],
                )
            elif window is not None:
                # Native T5Gemma2 bidirectional window, including its odd-width rule.
                visible = visible & (distance < (window + 1) // 2) & (-distance < window // 2 + 1)
            return visible & (query < query_length) & (key < key_length)

        return mask_mod

    masks = {}
    for window in set(windows) - flash_windows - set(partitions):
        mask_mod = make_mask(window)
        if device.type == 'cuda':
            masks[window] = _block_mask_factory()(
                mask_mod, 1, None, query_length, key_length, device=device,
                BLOCK_SIZE=_flex_block_size(),
            )
        else:
            queries = torch.arange(query_length, device=device)[:, None]
            keys = torch.arange(key_length, device=device)[None, :]
            masks[window] = mask_mod(0, 0, queries, keys)[None, None]
    return PackedLayout(positions[None], masks, backend, cu, key_cu, key_indices,
                        flash_windows, partitions)


def packed_attention(q, k, v, mask, scale, dropout=0.0):
    if q.device.type == 'cuda':
        if dropout:
            raise ValueError('Packed CUDA attention requires zero attention dropout')
        # The automatic short-query decode tile can be 256 on B200, which
        # does not divide our 128-token sparse blocks. Both paths accept 64.
        options = {'fwd_BLOCK_M': 64, 'fwd_BLOCK_N': 64, **_flex_option_overrides()}
        if options.get('BACKEND') == 'FLASH':
            # The FlashAttention-4 backend picks its own tiles (block masks of 256).
            options = {k: v for k, v in options.items() if not k.startswith(('fwd_', 'bwd_'))}
        if torch.compiler.is_compiling():
            from torch.nn.attention.flex_attention import flex_attention

            return flex_attention(
                q, k, v, block_mask=mask, scale=scale, enable_gqa=True, kernel_options=options,
            )
        return _attention_kernel()(
            q, k, v, block_mask=mask, scale=scale, enable_gqa=True, kernel_options=options,
        )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, dropout_p=dropout, enable_gqa=True,
    )


def layout_attention(q, k, v, layout, scale, window, cross=None, dropout=0.0):
    if window in layout.partitions:
        # Partition only attention. QKV projections and FFNs still consume the
        # complete pack, retaining large FP8 GEMMs and one decoder layer call.
        output = torch.zeros_like(q)
        for indices, child in layout.partitions[window]:
            part = layout_attention(
                q.index_select(2, indices), k.index_select(2, indices), v.index_select(2, indices),
                child, scale, window, cross, dropout,
            )
            output = output.index_copy(2, indices, part)
        return output
    if window in layout.flash_windows:
        from .flash_varlen import attention

        if dropout:
            raise ValueError('Packed CUDA attention requires zero attention dropout')
        return attention(q, k, v, layout, scale, None, cross)
    if cross is not None:
        k, v = torch.cat((k, cross[0]), dim=2), torch.cat((v, cross[1]), dim=2)
    return packed_attention(q, k, v, layout.masks[window], scale, dropout)


def encoder_attention(module, hidden_states, position_embeddings, layout):
    from .attention import project_qk

    q, k = project_qk(module, hidden_states, position_embeddings, 'reference')
    shape = (*hidden_states.shape[:-1], -1, module.head_dim)
    v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
    output = layout_attention(q, k, v, layout, module.scaling, module.sliding_window,
                              dropout=module.attention_dropout if module.training else 0.0)
    output = output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
    return module.o_proj(output), None


def decoder_attention(module, hidden_states, position_embeddings, cache, layout, packed_cross=None):
    from .attention import project_qk

    q, k = project_qk(module, hidden_states, position_embeddings, 'reference')
    shape = (*hidden_states.shape[:-1], -1, module.head_dim)
    v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
    if packed_cross is None:
        cross = cache.cross_attention_cache.layers[module.layer_idx]
        packed_cross = (cross.keys, cross.values)
    output = layout_attention(q, k, v, layout, module.scaling, module.sliding_window,
                              packed_cross, module.attention_dropout if module.training else 0.0)
    output = output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
    return module.o_proj(output), None, None


def forward_packed(model, requests, branch_batch_size, state_owners=None):
    owners = list(range(len(requests))) if state_owners is None else state_owners
    if len(owners) != len(requests):
        raise ValueError('state_owners must match requests')
    states, state_rows, request_rows = [], {}, []
    for owner, request in zip(owners, requests):
        key = (owner, request.state_key)
        if key not in state_rows:
            state_rows[key] = len(states)
            states.append(request)
        request_rows.append(state_rows[key])
    encoder_lengths = [len(r.state) for r in states]
    # Round a pack up to a multiple of model.pack_multiple tokens with one trailing dummy
    # segment (its own attention segment, owned by nobody, never read out): Transformer
    # Engine projections then take every hidden state without an alignment pad copy.
    multiple = getattr(model, 'pack_multiple', 1)
    filler = model.compiler.eos
    encoder_pad = -sum(encoder_lengths) % multiple
    encoder_tokens = [token for r in states for token in r.state] + [filler] * encoder_pad
    encoder_layout = packed_layout(
        encoder_lengths + ([encoder_pad] if encoder_pad else []),
        [layer.self_attn.sliding_window for layer in model.backbone.encoder.text_model.layers],
        model.device,
        backend=model.packing_attention,
    )
    empty = torch.empty(0, device=model.device)
    empty_masks = {'full_attention': empty, 'sliding_attention': empty}
    pixels = [r.pixel_values for r in states if r.pixel_values is not None]
    state = model.backbone.encoder(
        input_ids=model.ids(encoder_tokens),
        position_ids=encoder_layout.positions, attention_mask=empty_masks,
        packed_layout=encoder_layout, use_cache=False,
        pixel_values=torch.cat(pixels).to(model.device) if pixels else None,
    ).last_hidden_state
    spans = state.split(encoder_lengths + ([encoder_pad] if encoder_pad else []), dim=1)
    # A JEPA view shares its encoded state but owns its projection/dropout draws.
    state_bank = torch.cat([spans[index] for index in request_rows], dim=1)
    shared = model.state_kv(state_bank)
    questions = {
        (index, qid): q for index, request in enumerate(requests)
        for qid, q in request.questions.items()
    }
    decoder_windows = [layer.self_attn.sliding_window for layer in model.backbone.decoder.layers]
    members = list(questions.items())
    readouts = {}
    for start in range(0, len(members), branch_batch_size):
        group = members[start:start + branch_batch_size]
        lengths = [len(q.tokens) for _, q in group]
        owners_of_group = [key[0] for key, _ in group]
        decoder_tokens = [token for _, q in group for token in q.tokens]
        decoder_pad = -len(decoder_tokens) % multiple
        if decoder_pad:
            # The dummy question reads its owner's state like any question; nothing reads it.
            lengths = lengths + [decoder_pad]
            owners_of_group = owners_of_group + [owners_of_group[-1]]
            decoder_tokens = decoder_tokens + [filler] * decoder_pad
        layout = packed_layout(
            lengths, decoder_windows,
            model.device, owners=owners_of_group,
            state_lengths=[len(r.state) for r in requests],
            backend=model.packing_attention,
        )
        decoder = model.backbone.decoder
        hidden = decoder.embed_tokens(model.ids(decoder_tokens))
        embeddings = {
            layer_type: decoder.rotary_emb(hidden, layout.positions, layer_type)
            for layer_type in set(decoder.config.layer_types)
        }
        hidden = decoder.dropout(hidden)
        for index, layer in enumerate(decoder.layers):
            cross = shared.cross_attention_cache.layers[index]
            # Select the layer's tensors outside its compiled region. Passing
            # the whole cache specializes a new graph for every layer index.
            hidden = layer(
                hidden, embeddings[decoder.config.layer_types[index]], empty,
                layout.positions, None, False, state_bank,
                question_layout=layout, packed_cross=(cross.keys, cross.values),
            )
        hidden = decoder.dropout(decoder.norm(hidden))[0]
        indices, sizes, offset = [], [], 0
        for (_, question), length in zip(group, lengths):
            indices.extend(offset + p for p in question.candidate_positions)
            indices.append(offset + question.decision_position)
            sizes.append(len(question.candidate_positions) + 1)
            offset += length
        selected = hidden.index_select(0, torch.tensor(indices, device=hidden.device))
        for ((key, _), row) in zip(group, selected.split(sizes)):
            readouts[key] = row
    scored = model.score_readouts(questions, readouts)
    return [{qid: scored[index, qid] for qid in request.questions}
            for index, request in enumerate(requests)]
