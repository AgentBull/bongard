// Exact tiled attention. Each SIMD group owns eight rows; score tiles stay
// in registers / threadgroup memory. FP32 accumulation, online softmax, no atomics.
// HEAD_DIM and SIMD_GROUPS are supplied by flash_attention.py.
#include <metal_stdlib>
using namespace metal;

inline float row_sum(float x) {
    x += simd_shuffle_xor(x, 1);
    return x + simd_shuffle_xor(x, 2);
}

// A[8,D] @ B[32,D].T -> S[8,32]. All device rows are padded by the caller.
inline void scores(device const float* a, device const float* b,
                   threadgroup float* s) {
    simdgroup_float8x8 sum[4];
    #pragma unroll
    for (uint j = 0; j < 4; ++j) sum[j] = simdgroup_float8x8(0.0f);
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM; d += 8) {
        simdgroup_float8x8 x;
        simdgroup_load(x, a + d, HEAD_DIM);
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            simdgroup_float8x8 y;
            simdgroup_load(y, b + j * 8 * HEAD_DIM + d, HEAD_DIM, ulong2(0), true);
            simdgroup_multiply_accumulate(sum[j], x, y, sum[j]);
        }
    }
    #pragma unroll
    for (uint j = 0; j < 4; ++j) simdgroup_store(sum[j], s + j * 8, 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// Matrix fragments have two adjacent columns per lane. This is the Apple
// SIMD layout also used by PyTorch's MPS PrefillAttention.h / MLX. Four lanes
// share a row; XOR 1 and XOR 8 perform that row's reduction.
inline float matrix_row_sum(float x) {
    x += simd_shuffle_xor(x, 1);
    return x + simd_shuffle_xor(x, 8);
}
inline float matrix_row_max(float x) {
    x = max(x, simd_shuffle_xor(x, 1));
    return max(x, simd_shuffle_xor(x, 8));
}

kernel void attention_forward(
    device const float* q, device const float* k, device const float* v,
    device const bool* mask, device float* out, device float* lse,
    constant uint& rows, constant uint& keys,
    constant uint& padded_rows, constant uint& padded_keys,
    constant uint& masked, constant float& scale,
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint first = (group.x * SIMD_GROUPS + sg) * 8;
    uint local_row = ((lane / 4) & 4) + ((lane / 2) % 4);
    uint col = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    uint row = first + local_row;
    q += (group.y * padded_rows + first) * HEAD_DIM;
    k += group.y * padded_keys * HEAD_DIM;
    v += group.y * padded_keys * HEAD_DIM;
    out += (group.y * padded_rows + first) * HEAD_DIM;
    lse += group.y * padded_rows;
    simdgroup_float8x8 acc[HEAD_DIM / 8];
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 8; ++d) acc[d] = simdgroup_float8x8(0.0f);
    float maximum = -INFINITY, normalizer = 0;
    for (uint start = 0; start < padded_keys; start += 32) {
        simdgroup_float8x8 s[4];
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            s[j] = simdgroup_float8x8(0.0f);
            for (uint d = 0; d < HEAD_DIM; d += 8) {
                simdgroup_float8x8 query, key;
                simdgroup_load(query, q + d, HEAD_DIM);
                simdgroup_load(key, k + (start + j * 8) * HEAD_DIM + d,
                               HEAD_DIM, ulong2(0), true);
                simdgroup_multiply_accumulate(s[j], query, key, s[j]);
            }
        }
        float tile_max = -INFINITY;
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            for (uint e = 0; e < 2; ++e) {
                uint key = start + j * 8 + col + e;
                bool visible = row < rows && key < keys;
                if (masked && visible) visible = mask[(group.y * rows + row) * keys + key];
                float value = visible ? s[j].thread_elements()[e] * scale : -INFINITY;
                s[j].thread_elements()[e] = value;
                tile_max = max(tile_max, value);
            }
        }
        float next = max(maximum, matrix_row_max(tile_max));
        float alpha = isfinite(maximum) ? exp(maximum - next) : 0;
        float total = 0;
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            for (uint e = 0; e < 2; ++e) {
                float p = isfinite(next) ? exp(s[j].thread_elements()[e] - next) : 0;
                s[j].thread_elements()[e] = p;
                total += p;
            }
        }
        normalizer = normalizer * alpha + matrix_row_sum(total);
        maximum = next;
        #pragma unroll
        for (uint d = 0; d < HEAD_DIM / 8; ++d) {
            acc[d].thread_elements()[0] *= alpha;
            acc[d].thread_elements()[1] *= alpha;
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                simdgroup_float8x8 value;
                simdgroup_load(value, v + (start + j * 8) * HEAD_DIM + d * 8, HEAD_DIM);
                simdgroup_multiply_accumulate(acc[d], s[j], value, acc[d]);
            }
        }
    }
    float inverse = normalizer > 0 ? 1.0f / normalizer : 0;
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 8; ++d) {
        acc[d].thread_elements()[0] *= inverse;
        acc[d].thread_elements()[1] *= inverse;
        simdgroup_store(acc[d], out + d * 8, HEAD_DIM);
    }
    if (col == 0) lse[row] = normalizer > 0 ? maximum + log(normalizer) : -INFINITY;
}

kernel void attention_backward_q(
    device const float* q, device const float* k, device const float* v,
    device const bool* mask, device const float* out, device const float* lse,
    device const float* grad_out, device const float* grad_lse,
    device float* dq, device float* delta,
    constant uint& rows, constant uint& keys,
    constant uint& padded_rows, constant uint& padded_keys,
    constant uint& masked, constant float& scale,
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float workspace[SIMD_GROUPS * 512];
    threadgroup float* s = workspace + sg * 512;
    threadgroup float* dp = s + 256;
    uint first = (group.x * SIMD_GROUPS + sg) * 8;
    uint row = first + lane / 4, col = lane % 4;
    uint offset = (group.y * padded_rows + first) * HEAD_DIM;
    q += offset; out += offset; grad_out += offset; dq += offset;
    k += group.y * padded_keys * HEAD_DIM;
    v += group.y * padded_keys * HEAD_DIM;
    uint row_offset = group.y * padded_rows + row;
    float z = lse[row_offset], gz = grad_lse[row_offset], dot = 0;
    #pragma unroll
    for (uint d = col; d < HEAD_DIM; d += 4)
        dot += out[(lane / 4) * HEAD_DIM + d] * grad_out[(lane / 4) * HEAD_DIM + d];
    dot = row_sum(dot);
    if (col == 0) delta[row_offset] = dot;
    simdgroup_float8x8 acc[HEAD_DIM / 8];
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 8; ++d) acc[d] = simdgroup_float8x8(0.0f);
    for (uint start = 0; start < padded_keys; start += 32) {
        scores(q, k + start * HEAD_DIM, s);
        scores(grad_out, v + start * HEAD_DIM, dp);
        for (uint t = 0; t < 8; ++t) {
            uint j = col + 4 * t, key = start + j, at = (lane / 4) * 32 + j;
            bool visible = row < rows && key < keys && isfinite(z);
            if (masked && visible) visible = mask[(group.y * rows + row) * keys + key];
            float p = visible ? exp(s[at] * scale - z) : 0;
            s[at] = p * (dp[at] - dot + gz) * scale;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            simdgroup_float8x8 ds;
            simdgroup_load(ds, s + j * 8, 32);
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM / 8; ++d) {
                simdgroup_float8x8 key;
                simdgroup_load(key, k + (start + j * 8) * HEAD_DIM + d * 8, HEAD_DIM);
                simdgroup_multiply_accumulate(acc[d], ds, key, acc[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 8; ++d) simdgroup_store(acc[d], dq + d * 8, HEAD_DIM);
}

// Assign complete key rows to a SIMD group and loop over query tiles. This
// reduces shared-state gradients without atomics or repeated K/V tensors.
kernel void attention_backward_kv(
    device const float* q, device const float* k, device const float* v,
    device const bool* mask, device const float* lse, device const float* delta,
    device const float* grad_out, device const float* grad_lse,
    device float* dk, device float* dv,
    constant uint& rows, constant uint& keys,
    constant uint& padded_rows, constant uint& padded_keys,
    constant uint& masked, constant float& scale,
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float workspace[SIMD_GROUPS * 512];
    threadgroup float* s = workspace + sg * 512;
    threadgroup float* p = s + 256;
    uint first = (group.x * SIMD_GROUPS + sg) * 8;
    uint key = first + lane / 4, col = lane % 4;
    uint offset = (group.y * padded_keys + first) * HEAD_DIM;
    k += offset; v += offset; dk += offset; dv += offset;
    q += group.y * padded_rows * HEAD_DIM;
    grad_out += group.y * padded_rows * HEAD_DIM;
    lse += group.y * padded_rows; delta += group.y * padded_rows;
    grad_lse += group.y * padded_rows;
    simdgroup_float8x8 acc_k[HEAD_DIM / 8], acc_v[HEAD_DIM / 8];
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 8; ++d) {
        acc_k[d] = simdgroup_float8x8(0.0f);
        acc_v[d] = simdgroup_float8x8(0.0f);
    }
    for (uint start = 0; start < padded_rows; start += 32) {
        scores(k, q + start * HEAD_DIM, s);
        scores(v, grad_out + start * HEAD_DIM, p);
        for (uint t = 0; t < 8; ++t) {
            uint j = col + 4 * t, row = start + j, at = (lane / 4) * 32 + j;
            bool visible = row < rows && key < keys && isfinite(lse[row]);
            if (masked && visible) visible = mask[(group.y * rows + row) * keys + key];
            float prob = visible ? exp(s[at] * scale - lse[row]) : 0;
            s[at] = prob * (p[at] - delta[row] + grad_lse[row]) * scale;
            p[at] = prob;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint j = 0; j < 4; ++j) {
            simdgroup_float8x8 ds, prob;
            simdgroup_load(ds, s + j * 8, 32);
            simdgroup_load(prob, p + j * 8, 32);
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM / 8; ++d) {
                simdgroup_float8x8 query, go;
                simdgroup_load(query, q + (start + j * 8) * HEAD_DIM + d * 8, HEAD_DIM);
                simdgroup_load(go, grad_out + (start + j * 8) * HEAD_DIM + d * 8, HEAD_DIM);
                simdgroup_multiply_accumulate(acc_k[d], ds, query, acc_k[d]);
                simdgroup_multiply_accumulate(acc_v[d], prob, go, acc_v[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 8; ++d) {
        simdgroup_store(acc_k[d], dk + d * 8, HEAD_DIM);
        simdgroup_store(acc_v[d], dv + d * 8, HEAD_DIM);
    }
}
