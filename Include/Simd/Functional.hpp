/*
 *	Name: Functional
 *	Author: Pawel Mrochen
 */

#pragma once

#include <functional>
#include <cstddef>
#include <cstdint>
#include "Intrinsics.hpp"

namespace std {

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

#if SIMD_HAS_FLOAT4
template<>
struct hash<::simd::float4>
{
	// Consistent with equal(): lanes comparing equal hash equally, i.e. negative zero is hashed as positive zero.
	size_t operator()(const ::simd::float4& v) const noexcept
	{
		std::uint64_t h[2];
		::simd::store(::simd::normalize(v), reinterpret_cast<float*>(h));

		// Chain both halves through a bijective 64-bit finalizer (SplitMix64), so that the result depends on the position
		// of each lane (a plain XOR of the lanes cancels out equal lanes and ignores their order).
		auto mix = [](std::uint64_t x)
		{
			x ^= x >> 30;
			x *= 0xBF58476D1CE4E5B9ull;
			x ^= x >> 27;
			x *= 0x94D049BB133111EBull;
			x ^= x >> 31;
			return x;
		};

		const std::uint64_t x = mix(mix(h[0] + 0x9E3779B97F4A7C15ull) ^ h[1]);
		if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t))
			return static_cast<std::size_t>(x);
		else
			return static_cast<std::size_t>(x ^ (x >> 32));
	}
};
#endif /* SIMD_HAS_FLOAT4 */

#if SIMD_HAS_INT4
template<>
struct hash<::simd::int4>
{
	// Hashes all 128 bits of the vector; vectors with identical bits hash to the same value.
	size_t operator()(const ::simd::int4& v) const noexcept
	{
		std::uint64_t h[2];
		::simd::store(v, reinterpret_cast<int*>(h));

		// Chain both halves through a bijective 64-bit finalizer (SplitMix64), so that the result depends on the position
		// of each lane (a plain XOR of the lanes cancels out equal lanes and ignores their order).
		auto mix = [](std::uint64_t x)
		{
			x ^= x >> 30;
			x *= 0xBF58476D1CE4E5B9ull;
			x ^= x >> 27;
			x *= 0x94D049BB133111EBull;
			x ^= x >> 31;
			return x;
		};

		const std::uint64_t x = mix(mix(h[0] + 0x9E3779B97F4A7C15ull) ^ h[1]);
		if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t))
			return static_cast<std::size_t>(x);
		else
			return static_cast<std::size_t>(x ^ (x >> 32));
	}
};
#endif /* SIMD_HAS_INT4 */

} // namespace std
