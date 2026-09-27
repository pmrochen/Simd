/*
 *	Name: AvxIntrinsics
 *	Author: Pawel Mrochen
 */

#pragma once

#if defined(__AVX2__)
#define SIMD_AVX 2
#elif defined(__AVX__)
#define SIMD_AVX 1
#endif

#ifdef SIMD_AVX

#include "SseIntrinsics.hpp"
#include <immintrin.h>
#include <type_traits>
#include <tuple>

#define SIMD_AVX_HAS_DOUBLE4 0/*1*/ // #TODO Change to 1 after full implementation

#if !defined(__clang__) && !defined(__GNUC__) && defined(_MSC_VER)

inline __m256d operator+(const __m256d v) { return v; }
inline __m256d operator-(const __m256d v) { return _mm256_xor_pd(v, _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000ll))); }
inline __m256d& operator+=(__m256d& v1, const __m256d v2) { v1 = _mm256_add_pd(v1, v2); return v1; }
inline __m256d& operator-=(__m256d& v1, const __m256d v2) { v1 = _mm256_sub_pd(v1, v2); return v1; }
inline __m256d& operator*=(__m256d& v1, const __m256d v2) { v1 = _mm256_mul_pd(v1, v2); return v1; }
inline __m256d& operator/=(__m256d& v1, const __m256d v2) { v1 = _mm256_div_pd(v1, v2); return v1; }
inline __m256d& operator*=(__m256d& v1, double s) { v1 = _mm256_mul_pd(v1, _mm256_set1_pd(s)); return v1; }
inline __m256d& operator/=(__m256d& v1, double s) { v1 = _mm256_div_pd(v1, _mm256_set1_pd(s)); return v1; }
inline __m256d operator+(const __m256d v1, const __m256d v2) { return _mm256_add_pd(v1, v2); }
inline __m256d operator-(const __m256d v1, const __m256d v2) { return _mm256_sub_pd(v1, v2); }
inline __m256d operator*(const __m256d v1, const __m256d v2) { return _mm256_mul_pd(v1, v2); }
inline __m256d operator/(const __m256d v1, const __m256d v2) { return _mm256_div_pd(v1, v2); }
inline __m256d operator*(const __m256d v, double s) { return _mm256_mul_pd(v, _mm256_set1_pd(s)); }
inline __m256d operator*(double s, const __m256d v) { return _mm256_mul_pd(_mm256_set1_pd(s), v); }
inline __m256d operator/(const __m256d v, double s) { return _mm256_div_pd(v, _mm256_set1_pd(s)); }

#endif

namespace simd::avx {

using double4 = __m256d;

using simd::sse::X;
using simd::sse::Y;
using simd::sse::Z;
using simd::sse::W;

// #TODO

template<typename T, int N /*= 1*/>
	requires (std::is_same_v<T, __m256d> && (N >= 1) && (N <= 4))
inline T set(double s)
{
	if constexpr (N == 1)
	{
		return __m256d{}; // #TODO
	}
	else if constexpr (N == 2)
	{
		return __m256d{}; // #TODO
	} 
	else if constexpr (N == 3)
	{
		return __m256d{}; // #TODO
	}
	else //if constexpr (N == 4)
	{
		return _mm256_set1_pd(s);
	}
}

template<typename T>
	requires std::is_same_v<T, __m256d>
inline T set(double x, double y, double z, double w)
{ 
	return _mm256_setr_pd(x, y, z, w); 
}

template<typename T, int N>
	requires (std::is_same_v<T, __m256d> && (N >= 1) && (N <= 4))
inline T load(const double* v)
{
	if constexpr (N == 1)
		return __m256d{}; // #TODO
	else if constexpr (N == 2)
		return __m256d{}; // #TODO
	else if constexpr (N == 3)
		return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(&v[0])), _mm_load_sd(&v[2]), 1);
	else //if constexpr (N == 4)
		return __m256d{}; // #TODO
}

template<int N>
	requires ((N >= 1) && (N <= 4))
inline void store(__m256d u, double* v) 
{
	if constexpr (N == 1)
	{
		// #TODO 
	}
	else if constexpr (N == 4)
	{
		// #TODO 
	}
	else
	{
		_mm_storeu_pd(&v[0], _mm256_castpd256_pd128(u));
		if constexpr (N >= 3)
			_mm_store_sd(&v[2], _mm256_extractf128_pd(u, 1));
	} 
}

inline __m256d add(__m256d v1, __m256d v2)
{
	return _mm256_add_pd(v1, v2);
}

inline __m256d sub(__m256d v1, __m256d v2)
{
	return _mm256_sub_pd(v1, v2);
}

inline __m256d mul(__m256d v1, __m256d v2)
{
	return _mm256_mul_pd(v1, v2);
}

// #TODO

} // namespace simd::avx

#endif /* SIMD_AVX */
