/*
 *	Name: Functional
 *	Author: Pawel Mrochen
 */

#pragma once

#include <functional>
#include <cstddef>
#include "Intrinsics.hpp"

#if SIMD_HAS_FLOAT4

namespace std {

template<typename T>
struct equal_to;

template<>
struct equal_to<::simd::float4>
{
	bool operator()(const ::simd::float4& v1, const ::simd::float4& v2) const noexcept
	{
		return ::simd::all4(::simd::equal(v1, v2));
	}
};

template<typename T>
struct hash;

template<>
struct hash<::simd::float4>
{
	size_t operator()(const ::simd::float4& v) const noexcept
	{
        return ::simd::hash(v);
    }
};

} // namespace std

#endif /* SIMD_HAS_FLOAT4 */
