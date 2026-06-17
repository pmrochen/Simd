/*
 *	Name: Functional
 *	Author: Pawel Mrochen
 */

#pragma once

#include <functional>
#include <cstddef>
#include "Intrinsics.hpp"

namespace std {

template<typename T>
struct equal_to;

#if SIMD_HAS_FLOAT4
template<>
struct equal_to<::simd::float4>
{
	bool operator()(const ::simd::float4& v1, const ::simd::float4& v2) const noexcept
	{
		return ::simd::all(::simd::equal(v1, v2));
	}
};
#endif /* SIMD_HAS_FLOAT4 */

#if SIMD_HAS_INT4
template<>
struct equal_to<::simd::int4>
{
	bool operator()(const ::simd::int4& v1, const ::simd::int4& v2) const noexcept
	{
		return ::simd::all(::simd::equal(v1, v2));
	}
};
#endif /* SIMD_HAS_INT4 */

template<typename T>
struct hash;

#if SIMD_HAS_FLOAT4
template<>
struct hash<::simd::float4>
{
	size_t operator()(const ::simd::float4& v) const noexcept
	{
        return ::simd::hash(v);
    }
};
#endif /* SIMD_HAS_FLOAT4 */

#if SIMD_HAS_INT4
template<>
struct hash<::simd::int4>
{
	size_t operator()(const ::simd::int4& v) const noexcept
	{
        return ::simd::hash(v);
    }
};
#endif /* SIMD_HAS_INT4 */

} // namespace std
