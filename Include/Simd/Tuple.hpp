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
	requires (std::floating_point<T> || std::integral<T>)
struct alignas(sizeof(T)*N) Tuple
{
	using ComponentType = T;
	using DataType = typename scalar::Storage<T, N>::Type;
	
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
	Tuple xxxy() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, X, Y>(value)); }
	Tuple xxxz() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, X, Z>(value)); }
	Tuple xxxw() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, X, W>(value)); }
	Tuple xxyx() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Y, X>(value)); }
	Tuple xxyy() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Y, Y>(value)); }
	Tuple xxyz() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Y, Z>(value)); }
	Tuple xxyw() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Y, W>(value)); }
	Tuple xxzx() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Z, X>(value)); }
	Tuple xxzy() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Z, Y>(value)); }
	Tuple xxzz() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Z, Z>(value)); }
	Tuple xxzw() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, Z, W>(value)); }
	Tuple xxwx() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, W, X>(value)); }
	Tuple xxwy() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, W, Y>(value)); }
	Tuple xxwz() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, W, Z>(value)); }
	Tuple xxww() const noexcept requires (N == 4) { return Tuple(swizzle<X, X, W, W>(value)); }
	Tuple xyxx() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, X, X>(value)); }
	Tuple xyxy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, X, Y>(value)); }
	Tuple xyxz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, X, Z>(value)); }
	Tuple xyxw() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, X, W>(value)); }
	Tuple xyyx() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Y, X>(value)); }
	Tuple xyyy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Y, Y>(value)); }
	Tuple xyyz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Y, Z>(value)); }
	Tuple xyyw() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Y, W>(value)); }
	Tuple xyzx() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Z, X>(value)); }
	Tuple xyzy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Z, Y>(value)); }
	Tuple xyzz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Z, Z>(value)); }
	Tuple xyzw() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, Z, W>(value)); }
	Tuple xywx() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, W, X>(value)); }
	Tuple xywy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, W, Y>(value)); }
	Tuple xywz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, W, Z>(value)); }
	Tuple xyww() const noexcept requires (N == 4) { return Tuple(swizzle<X, Y, W, W>(value)); }
	Tuple xzxx() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, X, X>(value)); }
	Tuple xzxy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, X, Y>(value)); }
	Tuple xzxz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, X, Z>(value)); }
	Tuple xzxw() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, X, W>(value)); }
	Tuple xzyx() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Y, X>(value)); }
	Tuple xzyy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Y, Y>(value)); }
	Tuple xzyz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Y, Z>(value)); }
	Tuple xzyw() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Y, W>(value)); }
	Tuple xzzx() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Z, X>(value)); }
	Tuple xzzy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Z, Y>(value)); }
	Tuple xzzz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Z, Z>(value)); }
	Tuple xzzw() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, Z, W>(value)); }
	Tuple xzwx() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, W, X>(value)); }
	Tuple xzwy() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, W, Y>(value)); }
	Tuple xzwz() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, W, Z>(value)); }
	Tuple xzww() const noexcept requires (N == 4) { return Tuple(swizzle<X, Z, W, W>(value)); }
	Tuple xwxx() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, X, X>(value)); }
	Tuple xwxy() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, X, Y>(value)); }
	Tuple xwxz() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, X, Z>(value)); }
	Tuple xwxw() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, X, W>(value)); }
	Tuple xwyx() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, Y, X>(value)); }
	Tuple xwyy() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, Y, Y>(value)); }
	Tuple xwyz() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, Y, Z>(value)); }
	Tuple xwyw() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, Y, W>(value)); }
	Tuple xwzx() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, Z, X>(value)); }
	Tuple xwzy() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, Z, Y>(value)); }
	Tuple xwzz() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, Z, Z>(value)); }
	Tuple xwzw() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, Z, W>(value)); }
	Tuple xwwx() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, W, X>(value)); }
	Tuple xwwy() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, W, Y>(value)); }
	Tuple xwwz() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, W, Z>(value)); }
	Tuple xwww() const noexcept requires (N == 4) { return Tuple(swizzle<X, W, W, W>(value)); }
	Tuple yxxx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, X, X>(value)); }
	Tuple yxxy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, X, Y>(value)); }
	Tuple yxxz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, X, Z>(value)); }
	Tuple yxxw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, X, W>(value)); }
	Tuple yxyx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Y, X>(value)); }
	Tuple yxyy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Y, Y>(value)); }
	Tuple yxyz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Y, Z>(value)); }
	Tuple yxyw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Y, W>(value)); }
	Tuple yxzx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Z, X>(value)); }
	Tuple yxzy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Z, Y>(value)); }
	Tuple yxzz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Z, Z>(value)); }
	Tuple yxzw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, Z, W>(value)); }
	Tuple yxwx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, W, X>(value)); }
	Tuple yxwy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, W, Y>(value)); }
	Tuple yxwz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, W, Z>(value)); }
	Tuple yxww() const noexcept requires (N == 4) { return Tuple(swizzle<Y, X, W, W>(value)); }
	Tuple yyxx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, X, X>(value)); }
	Tuple yyxy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, X, Y>(value)); }
	Tuple yyxz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, X, Z>(value)); }
	Tuple yyxw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, X, W>(value)); }
	Tuple yyyx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, Y, X>(value)); }
	Tuple yyyy() const noexcept requires (N == 4) { return Tuple(broadcast<Y>(value)); }
	Tuple yyyz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, Y, Z>(value)); }
	Tuple yyyw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, Y, W>(value)); }
	Tuple yyzx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, Z, X>(value)); }
	Tuple yyzy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, Z, Y>(value)); }
	Tuple yyzz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, Z, Z>(value)); }
	Tuple yyzw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, Z, W>(value)); }
	Tuple yywx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, W, X>(value)); }
	Tuple yywy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, W, Y>(value)); }
	Tuple yywz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, W, Z>(value)); }
	Tuple yyww() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Y, W, W>(value)); }
	Tuple yzxx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, X, X>(value)); }
	Tuple yzxy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, X, Y>(value)); }
	Tuple yzxz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, X, Z>(value)); }
	Tuple yzxw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, X, W>(value)); }
	Tuple yzyx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Y, X>(value)); }
	Tuple yzyy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Y, Y>(value)); }
	Tuple yzyz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Y, Z>(value)); }
	Tuple yzyw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Y, W>(value)); }
	Tuple yzzx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Z, X>(value)); }
	Tuple yzzy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Z, Y>(value)); }
	Tuple yzzz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Z, Z>(value)); }
	Tuple yzzw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, Z, W>(value)); }
	Tuple yzwx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, W, X>(value)); }
	Tuple yzwy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, W, Y>(value)); }
	Tuple yzwz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, W, Z>(value)); }
	Tuple yzww() const noexcept requires (N == 4) { return Tuple(swizzle<Y, Z, W, W>(value)); }
	Tuple ywxx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, X, X>(value)); }
	Tuple ywxy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, X, Y>(value)); }
	Tuple ywxz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, X, Z>(value)); }
	Tuple ywxw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, X, W>(value)); }
	Tuple ywyx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, Y, X>(value)); }
	Tuple ywyy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, Y, Y>(value)); }
	Tuple ywyz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, Y, Z>(value)); }
	Tuple ywyw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, Y, W>(value)); }
	Tuple ywzx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, Z, X>(value)); }
	Tuple ywzy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, Z, Y>(value)); }
	Tuple ywzz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, Z, Z>(value)); }
	Tuple ywzw() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, Z, W>(value)); }
	Tuple ywwx() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, W, X>(value)); }
	Tuple ywwy() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, W, Y>(value)); }
	Tuple ywwz() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, W, Z>(value)); }
	Tuple ywww() const noexcept requires (N == 4) { return Tuple(swizzle<Y, W, W, W>(value)); }
	Tuple zxxx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, X, X>(value)); }
	Tuple zxxy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, X, Y>(value)); }
	Tuple zxxz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, X, Z>(value)); }
	Tuple zxxw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, X, W>(value)); }
	Tuple zxyx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Y, X>(value)); }
	Tuple zxyy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Y, Y>(value)); }
	Tuple zxyz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Y, Z>(value)); }
	Tuple zxyw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Y, W>(value)); }
	Tuple zxzx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Z, X>(value)); }
	Tuple zxzy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Z, Y>(value)); }
	Tuple zxzz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Z, Z>(value)); }
	Tuple zxzw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, Z, W>(value)); }
	Tuple zxwx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, W, X>(value)); }
	Tuple zxwy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, W, Y>(value)); }
	Tuple zxwz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, W, Z>(value)); }
	Tuple zxww() const noexcept requires (N == 4) { return Tuple(swizzle<Z, X, W, W>(value)); }
	Tuple zyxx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, X, X>(value)); }
	Tuple zyxy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, X, Y>(value)); }
	Tuple zyxz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, X, Z>(value)); }
	Tuple zyxw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, X, W>(value)); }
	Tuple zyyx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Y, X>(value)); }
	Tuple zyyy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Y, Y>(value)); }
	Tuple zyyz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Y, Z>(value)); }
	Tuple zyyw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Y, W>(value)); }
	Tuple zyzx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Z, X>(value)); }
	Tuple zyzy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Z, Y>(value)); }
	Tuple zyzz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Z, Z>(value)); }
	Tuple zyzw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, Z, W>(value)); }
	Tuple zywx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, W, X>(value)); }
	Tuple zywy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, W, Y>(value)); }
	Tuple zywz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, W, Z>(value)); }
	Tuple zyww() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Y, W, W>(value)); }
	Tuple zzxx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, X, X>(value)); }
	Tuple zzxy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, X, Y>(value)); }
	Tuple zzxz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, X, Z>(value)); }
	Tuple zzxw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, X, W>(value)); }
	Tuple zzyx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, Y, X>(value)); }
	Tuple zzyy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, Y, Y>(value)); }
	Tuple zzyz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, Y, Z>(value)); }
	Tuple zzyw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, Y, W>(value)); }
	Tuple zzzx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, Z, X>(value)); }
	Tuple zzzy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, Z, Y>(value)); }
	Tuple zzzz() const noexcept requires (N == 4) { return Tuple(broadcast<Z>(value)); }
	Tuple zzzw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, Z, W>(value)); }
	Tuple zzwx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, W, X>(value)); }
	Tuple zzwy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, W, Y>(value)); }
	Tuple zzwz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, W, Z>(value)); }
	Tuple zzww() const noexcept requires (N == 4) { return Tuple(swizzle<Z, Z, W, W>(value)); }
	Tuple zwxx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, X, X>(value)); }
	Tuple zwxy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, X, Y>(value)); }
	Tuple zwxz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, X, Z>(value)); }
	Tuple zwxw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, X, W>(value)); }
	Tuple zwyx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Y, X>(value)); }
	Tuple zwyy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Y, Y>(value)); }
	Tuple zwyz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Y, Z>(value)); }
	Tuple zwyw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Y, W>(value)); }
	Tuple zwzx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Z, X>(value)); }
	Tuple zwzy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Z, Y>(value)); }
	Tuple zwzz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Z, Z>(value)); }
	Tuple zwzw() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, Z, W>(value)); }
	Tuple zwwx() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, W, X>(value)); }
	Tuple zwwy() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, W, Y>(value)); }
	Tuple zwwz() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, W, Z>(value)); }
	Tuple zwww() const noexcept requires (N == 4) { return Tuple(swizzle<Z, W, W, W>(value)); }
	Tuple wxxx() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, X, X>(value)); }
	Tuple wxxy() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, X, Y>(value)); }
	Tuple wxxz() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, X, Z>(value)); }
	Tuple wxxw() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, X, W>(value)); }
	Tuple wxyx() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, Y, X>(value)); }
	Tuple wxyy() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, Y, Y>(value)); }
	Tuple wxyz() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, Y, Z>(value)); }
	Tuple wxyw() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, Y, W>(value)); }
	Tuple wxzx() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, Z, X>(value)); }
	Tuple wxzy() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, Z, Y>(value)); }
	Tuple wxzz() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, Z, Z>(value)); }
	Tuple wxzw() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, Z, W>(value)); }
	Tuple wxwx() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, W, X>(value)); }
	Tuple wxwy() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, W, Y>(value)); }
	Tuple wxwz() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, W, Z>(value)); }
	Tuple wxww() const noexcept requires (N == 4) { return Tuple(swizzle<W, X, W, W>(value)); }
	Tuple wyxx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, X, X>(value)); }
	Tuple wyxy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, X, Y>(value)); }
	Tuple wyxz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, X, Z>(value)); }
	Tuple wyxw() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, X, W>(value)); }
	Tuple wyyx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Y, X>(value)); }
	Tuple wyyy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Y, Y>(value)); }
	Tuple wyyz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Y, Z>(value)); }
	Tuple wyyw() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Y, W>(value)); }
	Tuple wyzx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Z, X>(value)); }
	Tuple wyzy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Z, Y>(value)); }
	Tuple wyzz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Z, Z>(value)); }
	Tuple wyzw() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, Z, W>(value)); }
	Tuple wywx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, W, X>(value)); }
	Tuple wywy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, W, Y>(value)); }
	Tuple wywz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, W, Z>(value)); }
	Tuple wyww() const noexcept requires (N == 4) { return Tuple(swizzle<W, Y, W, W>(value)); }
	Tuple wzxx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, X, X>(value)); }
	Tuple wzxy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, X, Y>(value)); }
	Tuple wzxz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, X, Z>(value)); }
	Tuple wzxw() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, X, W>(value)); }
	Tuple wzyx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Y, X>(value)); }
	Tuple wzyy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Y, Y>(value)); }
	Tuple wzyz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Y, Z>(value)); }
	Tuple wzyw() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Y, W>(value)); }
	Tuple wzzx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Z, X>(value)); }
	Tuple wzzy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Z, Y>(value)); }
	Tuple wzzz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Z, Z>(value)); }
	Tuple wzzw() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, Z, W>(value)); }
	Tuple wzwx() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, W, X>(value)); }
	Tuple wzwy() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, W, Y>(value)); }
	Tuple wzwz() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, W, Z>(value)); }
	Tuple wzww() const noexcept requires (N == 4) { return Tuple(swizzle<W, Z, W, W>(value)); }
	Tuple wwxx() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, X, X>(value)); }
	Tuple wwxy() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, X, Y>(value)); }
	Tuple wwxz() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, X, Z>(value)); }
	Tuple wwxw() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, X, W>(value)); }
	Tuple wwyx() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, Y, X>(value)); }
	Tuple wwyy() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, Y, Y>(value)); }
	Tuple wwyz() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, Y, Z>(value)); }
	Tuple wwyw() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, Y, W>(value)); }
	Tuple wwzx() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, Z, X>(value)); }
	Tuple wwzy() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, Z, Y>(value)); }
	Tuple wwzz() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, Z, Z>(value)); }
	Tuple wwzw() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, Z, W>(value)); }
	Tuple wwwx() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, W, X>(value)); }
	Tuple wwwy() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, W, Y>(value)); }
	Tuple wwwz() const noexcept requires (N == 4) { return Tuple(swizzle<W, W, W, Z>(value)); }
	Tuple wwww() const noexcept requires (N == 4) { return Tuple(broadcast<W>(value)); }

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
