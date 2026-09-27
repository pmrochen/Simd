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

#if SIMD_HAS_DOUBLE2
template<>
struct Storage<double, 2> { using Type = double2; };
#endif

#if SIMD_HAS_DOUBLE4
template<>
struct Storage<double, 4> { using Type = double4; };
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
        explicit Bool(bool s) noexcept { value = set<DataType, N>(s); }
        Bool(bool x, bool y) noexcept requires (N == 2) { value = set<DataType>(x, y); }
        Bool(bool x, bool y, bool z, bool w) noexcept requires (N == 4) { value = set<DataType>(x, y, z, w); }
		Bool(DataType b) noexcept : value(b) {}

		operator DataType() const noexcept { return value; }
		Bool operator!() const noexcept { return Bool(logicalNot(value)); }

		DataType value;
	};

	Tuple() = default;
	explicit Tuple(T s) noexcept { value = set<DataType, N>(s); }
	Tuple(T x, T y) noexcept requires (N == 2) { value = set<DataType>(x, y); }
	Tuple(T x, T y, T z, T w) noexcept requires (N == 4) { value = set<DataType>(x, y, z, w); }
	Tuple(DataType v) noexcept : value(v) {}

	operator DataType() const noexcept { return value; }
	//T operator[](int i) const noexcept { /*TODO*/ }
	Tuple operator+() const noexcept { return *this; }
	Tuple operator-() const noexcept { return Tuple(neg(value)); }
	Tuple& operator+=(const Tuple v) noexcept { value = add(value, v); return *this; }
	Tuple& operator-=(const Tuple v) noexcept { value = sub(value, v); return *this; }
	Tuple& operator*=(const Tuple v) noexcept { value = mul(value, v); return *this; }
	Tuple& operator*=(T s) noexcept { value = mul(value, set<DataType, N>(s)); return *this; }
	Tuple& operator/=(const Tuple v) noexcept { value = div(value, v); return *this; }
	Tuple& operator/=(T s) noexcept { value = div(value, set<DataType, N>(s)); return *this; }
	Bool operator<(const Tuple& v) const noexcept { return Bool(less(value, v)); }
	Bool operator<=(const Tuple& v) const noexcept { return Bool(lessEqual(value, v)); }
	Bool operator>(const Tuple& v) const noexcept { return Bool(greater(value, v)); }
	Bool operator>=(const Tuple& v) const noexcept { return Bool(greaterEqual(value, v)); }
	Bool operator==(const Tuple& v) const noexcept { return Bool(equal(value, v)); }
	Bool operator!=(const Tuple& v) const noexcept { return Bool(logicalNot(equal(value, v))); }

	//template<std::size_t I> T& get() noexcept; // #TODO
	template<std::size_t I> T get() const noexcept { return extract<I>(value); }

	T x() const noexcept requires (N >= 1) { return extract<X>(value); }
	T y() const noexcept requires (N >= 2) { return extract<Y>(value); }
	T z() const noexcept requires (N >= 3) { return extract<Z>(value); }
	T w() const noexcept requires (N >= 4) { return extract<W>(value); }
	Tuple xx() const noexcept requires (N == 2) { return Tuple(broadcast<X>(value)); }
	Tuple yy() const noexcept requires (N == 2) { return Tuple(broadcast<Y>(value)); }
	Tuple xy() const noexcept requires (N == 2) { return Tuple(swizzle<X, Y>(value)); }
	Tuple yx() const noexcept requires (N == 2) { return Tuple(swizzle<Y, X>(value)); }
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
	return Tuple<T, N>(sub(v1, v2)); 
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator*(const Tuple<T, N> v1, const Tuple<T, N> v2) noexcept 
{ 
	return Tuple<T, N>(mul(v1, v2)); 
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator*(T s, const Tuple<T, N> v) noexcept 
{ 
	return Tuple<T, N>(mul(set<DataType, N>(s), v));
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator*(const Tuple<T, N> v, T s) noexcept 
{ 
	return Tuple<T, N>(mul(v, set<DataType, N>(s)));
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator/(const Tuple<T, N> v1, const Tuple<T, N> v2) noexcept 
{ 
	return Tuple<T, N>(div(v1, v2)); 
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator/(T s, const Tuple<T, N> v) noexcept 
{ 
	return Tuple<T, N>(div(set<DataType, N>(s), v));
}

template<typename T, int N>
	requires (std::floating_point<T> || std::integral<T>)
inline Tuple<T, N> operator/(const Tuple<T, N> v, T s) noexcept 
{ 
	return Tuple<T, N>(div(v, set<DataType, N>(s)));
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

#if SIMD_HAS_DOUBLE2
using Double2 = templates::Tuple<double, 2>;
#endif

#if SIMD_HAS_DOUBLE4
using Double4 = templates::Tuple<double, 4>;
#endif

#if SIMD_HAS_INT4
using Int4 = templates::Tuple<int, 4>;
#endif

} // namespace simd

namespace std {

template<size_t I, typename T, int N>
struct tuple_element<I, ::simd::templates::Tuple<T, N>>
{
	using type = T;
};

template<typename T, int N>
struct tuple_size<::simd::templates::Tuple<T, N>> : public integral_constant<size_t, N>
{
};

template<typename T, int N>
struct equal_to<::simd::templates::Tuple<T, N>>
{
	bool operator()(const ::simd::templates::Tuple<T, N>& v1, const ::simd::templates::Tuple<T, N>& v2) const noexcept
	{
		return ::simd::all(::simd::equal(v1.value, v2.value));
	}
};

template<typename T, int N>
struct hash<::simd::templates::Tuple<T, N>>
{
	size_t operator()(const ::simd::templates::Tuple<T, N>& v) const noexcept
	{
		return ::simd::hash(v.value);
	}
};

} // namespace std
