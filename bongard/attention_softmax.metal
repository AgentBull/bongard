// GEMM supplies QK^T. Fuse scale, mask, stable softmax and natural-log LSE;
// overwrite the bounded score tile with probabilities for the second GEMM.
#include <metal_stdlib>
using namespace metal;

[[kernel, max_total_threads_per_threadgroup(THREADS)]]
void softmax_lse(device float* scores, device const bool* mask, device float* lse,
                constant uint& keys, constant uint& masked, constant float& scale,
                uint row [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]],
                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float workspace[THREADS / 32];
    float values[ITEMS];
    float maximum = -INFINITY;
    #pragma unroll
    for (uint i = 0; i < ITEMS; ++i) {
        uint col = tid + i * THREADS;
        float value = -INFINITY;
        if (col < keys && (!masked || mask[row * keys + col]))
            value = scores[row * keys + col] * scale;
        values[i] = value;
        maximum = max(maximum, value);
    }
    maximum = simd_max(maximum);
    if (lane == 0) workspace[sg] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = simd_max(lane < THREADS / 32 ? workspace[lane] : -INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    #pragma unroll
    for (uint i = 0; i < ITEMS; ++i) {
        // Keep the maximum in natural-log units: converting a large maximum
        // to log2 and back adds rounding error to the LSE used by backward.
        values[i] = isfinite(maximum) ? fast::exp2((values[i] - maximum) * M_LOG2E_F) : 0;
        total += values[i];
    }
    total = simd_sum(total);
    if (lane == 0) workspace[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(lane < THREADS / 32 ? workspace[lane] : 0);
    #pragma unroll
    for (uint i = 0; i < ITEMS; ++i) {
        uint col = tid + i * THREADS;
        if (col < keys) scores[row * keys + col] = total > 0 ? values[i] / total : 0;
    }
    if (tid == 0) lse[row] = total > 0 ? maximum + log(total) : -INFINITY;
}
