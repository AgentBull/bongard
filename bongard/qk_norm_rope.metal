#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
[[kernel, max_total_threads_per_threadgroup(32)]]
void qk_norm_rope(device const float* q, device const float* k,
                 device const float* qw, device const float* kw,
                 device const float* cosine, device const float* sine,
                 device float* oq, device float* ok,
                 constant uint& length, constant uint& qheads, constant uint& kheads,
                 constant uint& dim, constant uint& cos_batches,
                 constant float& qeps, constant float& keps,
                 uint row [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]]) {
    uint h = row % (qheads + kheads), token = row / (qheads + kheads);
    uint b = token / length, pos = token % length;
    bool query = h < qheads;
    uint heads = query ? qheads : kheads;
    h = query ? h : h - qheads;
    device const float* x = (query ? q : k) + (token * heads + h) * dim;
    device const float* w = query ? qw : kw;
    device float* out = (query ? oq : ok) + ((b * heads + h) * length + pos) * dim;
    uint ci = ((cos_batches == 1 ? 0 : b) * length + pos) * dim;
    float sum = 0;
    for (uint i = tid; i < dim; i += 32) sum += x[i] * x[i];
    float r = rsqrt(simd_sum(sum) / float(dim) + (query ? qeps : keps));
    for (uint i = tid; i < dim; i += 32) {
        uint j = (i + dim / 2) % dim;
        float a = (x[i] * r) * (1.0f + w[i]);
        float other = (x[j] * r) * (1.0f + w[j]);
        float rotated = i < dim / 2 ? -other : other;
        out[i] = a * cosine[ci + i] + rotated * sine[ci + i];
    }
}
