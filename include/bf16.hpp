#pragma once

#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define DF_HD __host__ __device__
#else
#define DF_HD
#endif

DF_HD inline uint32_t f32_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

DF_HD inline float bits_f32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

DF_HD inline float bf16_to_f32(uint16_t bits){
    __nv_bfloat16 h;
    memcpy(&h, &bits, sizeof(h));
    return __bfloat162float(h);
}

DF_HD inline uint16_t f32_to_bf16(float f){
    const uint32_t u = f32_bits(f);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) return 0xFFFFu;
    const __nv_bfloat16 h = __float2bfloat16(f);
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

DF_HD inline uint32_t bf16_widened_bits(uint16_t bits) {
    return f32_bits(bf16_to_f32(bits));
}

