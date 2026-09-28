/*
 *	Name: NeonIntrinsics
 *	Author: Pawel Mrochen
 */

#pragma once

#if defined(__aarch64__) || defined(_M_ARM64)
#define SIMD_NEON 8
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SIMD_NEON 7
#endif

#ifdef SIMD_NEON

#include <arm_neon.h>
#include <type_traits>
#include <tuple>

#define SIMD_NEON_HAS_FLOAT4 0/*1*/ // #TODO Change to 1 after full implementation

namespace simd::neon {
namespace detail {

// typedef union f4128
// {
// 	float32x4_t v;
// 	float f[4];
// } f4128;

const float32x4_t ZERO4 = { 0.0f, 0.0f, 0.0f, 0.0f };
const float32x4_t ONE4 = { 1.0f, 1.0f, 1.0f, 1.0f };
const float32x4_t HALF4 = { 0.5f, 0.5f, 0.5f, 0.5f };

} // namespace detail

using float4 = float32x4_t;

constexpr int X = 0;
constexpr int Y = 1;
constexpr int Z = 2;
constexpr int W = 3;

// #TODO

template<typename T>
	requires std::is_same_v<T, float32x4_t>
inline T zero()
{
	return vdupq_n_f32(0.0f);
}

template<typename T, int N /*= 1*/>
	requires (std::is_same_v<T, float32x4_t> && (N >= 1) && (N <= 4))
inline T set(float s)
{
	if constexpr (N == 1)
	{
		return float32x4_t{}; // #TODO
	}
	else if constexpr (N == 2)
	{
		return float32x4_t{}; // #TODO
	} 
	else if constexpr (N == 3)
	{
		return float32x4_t{}; // #TODO
	}
	else //if constexpr (N == 4)
	{
		return vdupq_n_f32(s);
	}
}

template<typename T, int N>
	requires (std::is_same_v<T, float32x4_t> && (N >= 1) && (N <= 4))
inline T load(const float* v)
{
	if constexpr (N == 1)
	{
		return float32x4_t{}; // #TODO
	}
	else if constexpr (N == 2)
	{
		float t[4] = { v[0], v[1], 0.0f, 0.0f };
		return vld1q_f32(t);
	}
	else if constexpr (N == 3)
	{
		float t[4] = { v[0], v[1], v[2], 0.0f };
		return vld1q_f32(t);
	}
	else //if constexpr (N == 4)
		return vld1q_f32(v); 
}

template<int N>
	requires ((N >= 1) && (N <= 4))
inline void store(float32x4_t u, float* v) 
{
	if constexpr (N == 4)
	{
		// #TODO
	}
	else
	{
		v[0] = vgetq_lane_f32(u, 0);
		if constexpr (N >= 2)
			v[1] = vgetq_lane_f32(u, 1);
		if constexpr (N >= 3)
			v[2] = vgetq_lane_f32(u, 2);
	} 
}

inline std::tuple<float32x4_t, float32x4_t, float32x4_t>/*float32x4x3_t*/ transpose(float32x4_t row0, float32x4_t row1, float32x4_t row2)
{
	//float32x4x3_t m = vld3q_f32(&row0);
	float32x4_t t0 = vzip1q_f32(row0, row1);
	float32x4_t t1 = vzip2q_f32(row0, row1);
	return { vsetq_lane_f32(vgetq_lane_f32(row2, 0), t0, 2),
		vsetq_lane_f32(vgetq_lane_f32(row2, 1), vcombine_f32(vget_high_f32(t0), vget_high_f32(t0)), 2),
		vsetq_lane_f32(vgetq_lane_f32(row2, 2), t1, 2) };
}

template<int I = 0>
	requires ((I & ~3) == 0)
inline float extract(float32x4_t v)
{
	return vgetq_lane_f32(v, I);
}

// inline float get(float32x4_t v, int index)
// {
// 	f4128 b;
// 	b.v = v;
// 	return b.f[index];
// }

template<int N = 4>
	requires ((N >= 1) && (N <= 4))
inline bool all(uint32x4_t b)
{
	auto v = vgetq_lane_u32(b, 0);
	if constexpr (N >= 2)
		v &= vgetq_lane_u32(b, 1);
	if constexpr (N >= 3)
		v &= vgetq_lane_u32(b, 2);
	if constexpr (N >= 4)
		v &= vgetq_lane_u32(b, 3);
	return (bool)v;
}

template<int N = 4>
	requires ((N >= 1) && (N <= 4))
inline bool any(uint32x4_t b)
{
	auto v = vgetq_lane_u32(b, 0);
	if constexpr (N >= 2)
		v |= vgetq_lane_u32(b, 1);
	if constexpr (N >= 3)
		v |= vgetq_lane_u32(b, 2);
	if constexpr (N >= 4)
		v |= vgetq_lane_u32(b, 3);
	return (bool)v;
}

inline uint32x4_t equal(float32x4_t v1, float32x4_t v2)
{
	return vceqq_f32(v1, v2);
}

inline uint32x4_t lessThan(float32x4_t v1, float32x4_t v2)
{
	return vcltq_f32(v1, v2);
}

inline uint32x4_t lessThanEqual(float32x4_t v1, float32x4_t v2)
{
	return vcleq_f32(v1, v2);
}

inline uint32x4_t greaterThan(float32x4_t v1, float32x4_t v2)
{
	return vcgtq_f32(v1, v2);
}

inline uint32x4_t greaterThanEqual(float32x4_t v1, float32x4_t v2)
{
	return vcgeq_f32(v1, v2);
}

inline float32x4_t min(float32x4_t v1, float32x4_t v2)
{
	return vminq_f32(v1, v2);
}

inline float32x4_t max(float32x4_t v1, float32x4_t v2)
{
	return vmaxq_f32(v1, v2);
}

inline float32x4_t neg(float32x4_t v)
{
	return vnegq_f32(v);
}

inline float32x4_t abs(float32x4_t v)
{
	return vabsq_f32(v);
}

inline float32x4_t add(float32x4_t v1, float32x4_t v2)
{
	return vaddq_f32(v1, v2);
}

inline float32x4_t sub(float32x4_t v1, float32x4_t v2)
{
	return vsubq_f32(v1, v2);
}

inline float32x4_t mul(float32x4_t v1, float32x4_t v2)
{
	return vmulq_f32(v1, v2);
}

//template<int N = 4> // #TODO
inline float32x4_t div(float32x4_t v1, float32x4_t v2)
{
#if SIMD_NEON >= 8
	return vdivq_f32(v1, v2);
#else
	#error // #TODO
#endif
}

// #TODO

} // namespace simd::neon

#endif /* SIMD_NEON */
