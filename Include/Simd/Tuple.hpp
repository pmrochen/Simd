/*
 *	Name: Tuple
 *	Author: Pawel Mrochen
 */

#pragma once

#include <type_traits>
#include <concepts>
#include <functional>
#include <tuple>
#include <cstddef>
#include "Intrinsics.hpp"

namespace simd {
namespace templates {

template<typename T, int N>
struct Storage;

#if SIMD_HAS_FLOAT4
template<>
struct Storage<float, 4> { using Type = float4; };
#endif

#if SIMD_HAS_INT4
template<>
struct Storage<int, 4> { using Type = int4; };
#endif

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
struct alignas(sizeof(T)*N) Tuple
{
	using ComponentType = T;
	using DataType = typename Storage<T, N>::Type;
	
	static constexpr int NUM_COMPONENTS = N;
	static constexpr int ALIGNMENT = sizeof(T)*N;

	struct alignas(sizeof(T)*N) Bool
	{
		Bool() = default;
        explicit Bool(bool s) noexcept { value = set4/*set*/(s); } // #FIXME Support N != 4
        Bool(bool x, bool y, bool z, bool w) noexcept requires (N == 4) { value = set4/*set*/(x, y, z, w); }
		Bool(DataType b) noexcept : value(b) {}

		operator DataType() const noexcept { return value; }
		Bool operator!() const noexcept { return Bool(logicalNot(value)); }

		DataType value;
	};

	Tuple() = default;
	explicit Tuple(T s) noexcept { value = set4/*set*/(s); } // #FIXME Support N != 4
	Tuple(T x, T y, T z, T w) noexcept requires (N == 4) { value = set4/*set*/(x, y, z, w); }
	Tuple(DataType v) noexcept : value(v) {}

	operator DataType() const noexcept { return value; }
	//T operator[](int i) const noexcept { /*TODO*/ }
	Tuple operator+() const noexcept { return *this; }
	Tuple operator-() const noexcept { return Tuple(neg(value)); }
	Tuple& operator+=(const Tuple v) noexcept { value = add(value, v); return *this; }
	Tuple& operator-=(const Tuple v) noexcept { value = subtract(value, v); return *this; }
	Tuple& operator*=(const Tuple v) noexcept { value = multiply(value, v); return *this; }
	Tuple& operator*=(T s) noexcept { value = multiply(value, set4/*set*/(s)); return *this; } // #FIXME Support N != 4
	Tuple& operator/=(const Tuple v) noexcept { value = divide(value, v); return *this; }
	Tuple& operator/=(T s) noexcept { value = divide(value, set4/*set*/(s)); return *this; } // #FIXME Support N != 4
	Bool operator<(const Tuple& v) const noexcept { return Bool(less(value, v)); }
	Bool operator<=(const Tuple& v) const noexcept { return Bool(lessEqual(value, v)); }
	Bool operator>(const Tuple& v) const noexcept { return Bool(greater(value, v)); }
	Bool operator>=(const Tuple& v) const noexcept { return Bool(greaterEqual(value, v)); }
	Bool operator==(const Tuple& v) const noexcept { return Bool(equal(value, v)); }
	Bool operator!=(const Tuple& v) const noexcept { return Bool(not4(equal(value, v))); }

	//template<std::size_t I> T& get() noexcept; // #TODO
	template<std::size_t I> T get() const noexcept { return extract<I>(value); }

	T x() const noexcept { return extract<X>(value); }
	T y() const noexcept { return extract<Y>(value); }
	T z() const noexcept { return extract<Z>(value); }
	T w() const noexcept { return extract<W>(value); }
	Tuple xxxx() const noexcept requires (N == 4) { return Tuple(broadcast<X>(value)); }
	Tuple yyyy() const noexcept requires (N == 4) { return Tuple(broadcast<Y>(value)); }
 	Tuple zzzz() const noexcept requires (N == 4) { return Tuple(broadcast<Z>(value)); }
	Tuple wwww() const noexcept requires (N == 4) { return Tuple(broadcast<W>(value)); }
	Tuple xyyy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Y, Y>(value)); }
	Tuple xyxy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, X, Y>(value)); }
	Tuple xxyy() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Y, Y>(value)); }
	Tuple xzzz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Z, Z>(value)); }
	Tuple xwww() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, W, W>(value)); }
	Tuple xyzz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Z, Z>(value)); }
	Tuple xyww() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, W, W>(value)); }
	Tuple xzww() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, W, W>(value)); }
	Tuple yxxx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, X, X>(value)); }
	Tuple yzzz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Z, Z>(value)); }
	Tuple yzxx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, X, X>(value)); }
	Tuple yzxw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, X, W>(value)); }
	Tuple yzww() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, W, W>(value)); }
	Tuple ywww() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, W, W>(value)); }
	Tuple yxzz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Z, Z>(value)); }
	Tuple yxzw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Z, W>(value)); }
	Tuple yxwz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, W, Z>(value)); }
	Tuple zxxx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, X, X>(value)); }
	Tuple zxyy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Y, Y>(value)); }
	Tuple zxyw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Y, W>(value)); }
	Tuple zyyy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Y, Y>(value)); }
	Tuple zyww() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, W, W>(value)); }
	Tuple zzww() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, W, W>(value)); }
	Tuple zwww() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, W, W>(value)); }
	Tuple zwxy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, X, Y>(value)); }
	Tuple zwzw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Z, W>(value)); }
	Tuple wxxx() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, X, X>(value)); }
	Tuple wyyy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Y, Y>(value)); }
	Tuple wzzz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Z, Z>(value)); }
	Tuple wzyx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Y, X>(value)); }

	DataType value;
};

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline typename Tuple<T, N>::Bool operator&&(const typename Tuple<T, N>::Bool b1, const typename Tuple<T, N>::Bool b2) noexcept 
{ 
	return Tuple<T, N>::Bool(logicalAnd(b1, b2));
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline typename Tuple<T, N>::Bool operator||(const typename Tuple<T, N>::Bool b1, const typename Tuple<T, N>::Bool b2) noexcept 
{ 
	return Tuple<T, N>::Bool(logicalOr(b1, b2));
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline bool any(const typename Tuple<T, N>::Bool b) noexcept 
{ 
	return any(b.value);
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline bool all(const typename Tuple<T, N>::Bool b) noexcept 
{ 
	return all(b.value); 
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator+(const Tuple<T, N> v1, const Tuple<T, N> v2) noexcept 
{ 
	return Tuple<T, N>(add(v1, v2));
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator-(const Tuple<T, N> v1, const Tuple<T, N> v2) noexcept 
{ 
	return Tuple<T, N>(subtract(v1, v2)); 
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator*(const Tuple<T, N> v1, const Tuple<T, N> v2) noexcept 
{ 
	return Tuple<T, N>(multiply(v1, v2)); 
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator*(T s, const Tuple<T, N> v) noexcept 
{ 
	return Tuple<T, N>(multiply(set4(s), v)); // #FIXME Support N != 4
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator*(const Tuple<T, N> v, T s) noexcept 
{ 
	return Tuple<T, N>(multiply(v, set4(s))); // #FIXME Support N != 4
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator/(const Tuple<T, N> v1, const Tuple<T, N> v2) noexcept 
{ 
	return Tuple<T, N>(divide(v1, v2)); 
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator/(T s, const Tuple<T, N> v) noexcept 
{ 
	return Tuple<T, N>(divide(set4(s), v)); // #FIXME Support N != 4
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator/(const Tuple<T, N> v, T s) noexcept 
{ 
	return Tuple<T, N>(divide(v, set4(s))); // #FIXME Support N != 4
}

//template<std::size_t I, typename T, int N>
//	requires (std::floating_point<T> || std::integral<T>)
//inline T& get(Tuple<T, N>& v) noexcept; // #TODO

template<std::size_t I, typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline T get(const Tuple<T, N>& v) noexcept 
{ 
	return extract<I>(v.value); 
}

} // namespace templates

#if SIMD_HAS_FLOAT4
using Float4 = templates::Tuple<float, 4>;
#endif

#if SIMD_HAS_INT4
using Int4 = templates::Tuple<int, 4>;
#endif

} // namespace simd

namespace std {

template<size_t I, typename T>
struct tuple_element;

template<size_t I, typename T, int N>
struct tuple_element<I, ::simd::templates::Tuple<T, N>>
{
	using type = T;
};

template<typename T>
struct tuple_size;

template<typename T, int N>
struct tuple_size<::simd::templates::Tuple<T, N>> : integral_constant<size_t, N>
{
};

template<typename T>
struct equal_to;

template<typename T, int N>
struct equal_to<::simd::templates::Tuple<T, N>>
{
	bool operator()(const ::simd::templates::Tuple<T, N>& v1, const ::simd::templates::Tuple<T, N>& v2) const noexcept
	{
		return ::simd::all(::simd::equal(v1.value, v2.value));
	}
};

template<typename T>
struct hash;

template<typename T, int N>
struct hash<::simd::templates::Tuple<T, N>>
{
	size_t operator()(const ::simd::templates::Tuple<T, N>& v) const noexcept
	{
		return ::simd::hash(v.value);
	}
};

} // namespace std
