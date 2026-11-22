/*
 *	Name: Intrinsics
 *	Author: Pawel Mrochen
 */

#pragma once

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#define SIMD_CALL __vectorcall
#else
#define SIMD_CALL
#endif

#include <concepts>
#include "SseIntrinsics.hpp"
#include "AvxIntrinsics.hpp"
#include "NeonIntrinsics.hpp"
#include "WasmIntrinsics.hpp"

namespace simd {

// #if defined(SIMD_WASM) // #TODO Uncomment after full implementation

// using namespace wasm;
// #define SIMD_HAS_INT4 0
// #define SIMD_HAS_FLOAT4 SIMD_WASM_HAS_FLOAT4
// #define SIMD_HAS_DOUBLE2 0
// #define SIMD_HAS_DOUBLE4 0

// #elif defined(SIMD_NEON) // #TODO Uncomment after full implementation

// using namespace neon;
// #define SIMD_HAS_INT4 0
// #define SIMD_HAS_FLOAT4 SIMD_NEON_HAS_FLOAT4
// #define SIMD_HAS_DOUBLE2 0
// #define SIMD_HAS_DOUBLE4 0

// #else

#ifdef SIMD_SSE
using namespace sse;
#define SIMD_HAS_INT4 SIMD_SSE_HAS_INT4
#define SIMD_HAS_FLOAT4 SIMD_SSE_HAS_FLOAT4
#define SIMD_HAS_DOUBLE2 SIMD_SSE_HAS_DOUBLE2
#else
#define SIMD_HAS_INT4 0
#define SIMD_HAS_FLOAT4 0
#define SIMD_HAS_DOUBLE2 0
#endif

#ifdef SIMD_AVX
using namespace avx;
#define SIMD_HAS_DOUBLE4 SIMD_AVX_HAS_DOUBLE4
#else
#define SIMD_HAS_DOUBLE4 0
#endif

//#endif

template<typename T>
concept FloatingPoint = (
#if SIMD_HAS_FLOAT4
	std::same_as<T, float4> || 
#endif
#if SIMD_HAS_DOUBLE2
	std::same_as<T, double2> || 
#endif
#if SIMD_HAS_DOUBLE4
	std::same_as<T, double4> ||
#endif
	false
);

template<typename T>
concept Integral = ( 
#if SIMD_HAS_INT4
	std::same_as<T, int4> ||
#endif
	false
);

template<typename T>
concept Arithmetic = (FloatingPoint<T> || Integral<T>);

namespace scalar {

template<typename T>
concept Arithmetic = (std::floating_point<T> || std::integral<T>);

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

} // namespace scalar

#if SIMD_HAS_DOUBLE2
template<typename T> inline decltype(broadcast<X>(T())) xx(T v) { return broadcast<X>(v); }
template<typename T> inline decltype(broadcast<Y>(T())) yy(T v) { return broadcast<Y>(v); }
template<typename T> inline decltype(swizzle<X, Y>(T())) xy(T v) { return swizzle<X, Y>(v); }
template<typename T> inline decltype(swizzle<Y, X>(T())) yx(T v) { return swizzle<Y, X>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(broadcast<X>(T())) xxxx(T v) { return broadcast<X>(v); }
template<typename T> inline decltype(swizzle<X, X, X, Y>(T())) xxxy(T v) { return swizzle<X, X, X, Y>(v); }
template<typename T> inline decltype(swizzle<X, X, X, Z>(T())) xxxz(T v) { return swizzle<X, X, X, Z>(v); }
template<typename T> inline decltype(swizzle<X, X, X, W>(T())) xxxw(T v) { return swizzle<X, X, X, W>(v); }
template<typename T> inline decltype(swizzle<X, X, Y, X>(T())) xxyx(T v) { return swizzle<X, X, Y, X>(v); }
template<typename T> inline decltype(swizzle<X, X, Y, Y>(T())) xxyy(T v) { return swizzle<X, X, Y, Y>(v); }
template<typename T> inline decltype(swizzle<X, X, Y, Z>(T())) xxyz(T v) { return swizzle<X, X, Y, Z>(v); }
template<typename T> inline decltype(swizzle<X, X, Y, W>(T())) xxyw(T v) { return swizzle<X, X, Y, W>(v); }
template<typename T> inline decltype(swizzle<X, X, Z, X>(T())) xxzx(T v) { return swizzle<X, X, Z, X>(v); }
template<typename T> inline decltype(swizzle<X, X, Z, Y>(T())) xxzy(T v) { return swizzle<X, X, Z, Y>(v); }
template<typename T> inline decltype(swizzle<X, X, Z, Z>(T())) xxzz(T v) { return swizzle<X, X, Z, Z>(v); }
template<typename T> inline decltype(swizzle<X, X, Z, W>(T())) xxzw(T v) { return swizzle<X, X, Z, W>(v); }
template<typename T> inline decltype(swizzle<X, X, W, X>(T())) xxwx(T v) { return swizzle<X, X, W, X>(v); }
template<typename T> inline decltype(swizzle<X, X, W, Y>(T())) xxwy(T v) { return swizzle<X, X, W, Y>(v); }
template<typename T> inline decltype(swizzle<X, X, W, Z>(T())) xxwz(T v) { return swizzle<X, X, W, Z>(v); }
template<typename T> inline decltype(swizzle<X, X, W, W>(T())) xxww(T v) { return swizzle<X, X, W, W>(v); }
template<typename T> inline decltype(swizzle<X, Y, X, X>(T())) xyxx(T v) { return swizzle<X, Y, X, X>(v); }
template<typename T> inline decltype(swizzle<X, Y, X, Y>(T())) xyxy(T v) { return swizzle<X, Y, X, Y>(v); }
template<typename T> inline decltype(swizzle<X, Y, X, Z>(T())) xyxz(T v) { return swizzle<X, Y, X, Z>(v); }
template<typename T> inline decltype(swizzle<X, Y, X, W>(T())) xyxw(T v) { return swizzle<X, Y, X, W>(v); }
template<typename T> inline decltype(swizzle<X, Y, Y, X>(T())) xyyx(T v) { return swizzle<X, Y, Y, X>(v); }
template<typename T> inline decltype(swizzle<X, Y, Y, Y>(T())) xyyy(T v) { return swizzle<X, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle<X, Y, Y, Z>(T())) xyyz(T v) { return swizzle<X, Y, Y, Z>(v); }
template<typename T> inline decltype(swizzle<X, Y, Y, W>(T())) xyyw(T v) { return swizzle<X, Y, Y, W>(v); }
template<typename T> inline decltype(swizzle<X, Y, Z, X>(T())) xyzx(T v) { return swizzle<X, Y, Z, X>(v); }
template<typename T> inline decltype(swizzle<X, Y, Z, Y>(T())) xyzy(T v) { return swizzle<X, Y, Z, Y>(v); }
template<typename T> inline decltype(swizzle<X, Y, Z, Z>(T())) xyzz(T v) { return swizzle<X, Y, Z, Z>(v); }
template<typename T> inline decltype(swizzle<X, Y, Z, W>(T())) xyzw(T v) { return swizzle<X, Y, Z, W>(v); }
template<typename T> inline decltype(swizzle<X, Y, W, X>(T())) xywx(T v) { return swizzle<X, Y, W, X>(v); }
template<typename T> inline decltype(swizzle<X, Y, W, Y>(T())) xywy(T v) { return swizzle<X, Y, W, Y>(v); }
template<typename T> inline decltype(swizzle<X, Y, W, Z>(T())) xywz(T v) { return swizzle<X, Y, W, Z>(v); }
template<typename T> inline decltype(swizzle<X, Y, W, W>(T())) xyww(T v) { return swizzle<X, Y, W, W>(v); }
template<typename T> inline decltype(swizzle<X, Z, X, X>(T())) xzxx(T v) { return swizzle<X, Z, X, X>(v); }
template<typename T> inline decltype(swizzle<X, Z, X, Y>(T())) xzxy(T v) { return swizzle<X, Z, X, Y>(v); }
template<typename T> inline decltype(swizzle<X, Z, X, Z>(T())) xzxz(T v) { return swizzle<X, Z, X, Z>(v); }
template<typename T> inline decltype(swizzle<X, Z, X, W>(T())) xzxw(T v) { return swizzle<X, Z, X, W>(v); }
template<typename T> inline decltype(swizzle<X, Z, Y, X>(T())) xzyx(T v) { return swizzle<X, Z, Y, X>(v); }
template<typename T> inline decltype(swizzle<X, Z, Y, Y>(T())) xzyy(T v) { return swizzle<X, Z, Y, Y>(v); }
template<typename T> inline decltype(swizzle<X, Z, Y, Z>(T())) xzyz(T v) { return swizzle<X, Z, Y, Z>(v); }
template<typename T> inline decltype(swizzle<X, Z, Y, W>(T())) xzyw(T v) { return swizzle<X, Z, Y, W>(v); }
template<typename T> inline decltype(swizzle<X, Z, Z, X>(T())) xzzx(T v) { return swizzle<X, Z, Z, X>(v); }
template<typename T> inline decltype(swizzle<X, Z, Z, Y>(T())) xzzy(T v) { return swizzle<X, Z, Z, Y>(v); }
template<typename T> inline decltype(swizzle<X, Z, Z, Z>(T())) xzzz(T v) { return swizzle<X, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle<X, Z, Z, W>(T())) xzzw(T v) { return swizzle<X, Z, Z, W>(v); }
template<typename T> inline decltype(swizzle<X, Z, W, X>(T())) xzwx(T v) { return swizzle<X, Z, W, X>(v); }
template<typename T> inline decltype(swizzle<X, Z, W, Y>(T())) xzwy(T v) { return swizzle<X, Z, W, Y>(v); }
template<typename T> inline decltype(swizzle<X, Z, W, Z>(T())) xzwz(T v) { return swizzle<X, Z, W, Z>(v); }
template<typename T> inline decltype(swizzle<X, Z, W, W>(T())) xzww(T v) { return swizzle<X, Z, W, W>(v); }
template<typename T> inline decltype(swizzle<X, W, X, X>(T())) xwxx(T v) { return swizzle<X, W, X, X>(v); }
template<typename T> inline decltype(swizzle<X, W, X, Y>(T())) xwxy(T v) { return swizzle<X, W, X, Y>(v); }
template<typename T> inline decltype(swizzle<X, W, X, Z>(T())) xwxz(T v) { return swizzle<X, W, X, Z>(v); }
template<typename T> inline decltype(swizzle<X, W, X, W>(T())) xwxw(T v) { return swizzle<X, W, X, W>(v); }
template<typename T> inline decltype(swizzle<X, W, Y, X>(T())) xwyx(T v) { return swizzle<X, W, Y, X>(v); }
template<typename T> inline decltype(swizzle<X, W, Y, Y>(T())) xwyy(T v) { return swizzle<X, W, Y, Y>(v); }
template<typename T> inline decltype(swizzle<X, W, Y, Z>(T())) xwyz(T v) { return swizzle<X, W, Y, Z>(v); }
template<typename T> inline decltype(swizzle<X, W, Y, W>(T())) xwyw(T v) { return swizzle<X, W, Y, W>(v); }
template<typename T> inline decltype(swizzle<X, W, Z, X>(T())) xwzx(T v) { return swizzle<X, W, Z, X>(v); }
template<typename T> inline decltype(swizzle<X, W, Z, Y>(T())) xwzy(T v) { return swizzle<X, W, Z, Y>(v); }
template<typename T> inline decltype(swizzle<X, W, Z, Z>(T())) xwzz(T v) { return swizzle<X, W, Z, Z>(v); }
template<typename T> inline decltype(swizzle<X, W, Z, W>(T())) xwzw(T v) { return swizzle<X, W, Z, W>(v); }
template<typename T> inline decltype(swizzle<X, W, W, X>(T())) xwwx(T v) { return swizzle<X, W, W, X>(v); }
template<typename T> inline decltype(swizzle<X, W, W, Y>(T())) xwwy(T v) { return swizzle<X, W, W, Y>(v); }
template<typename T> inline decltype(swizzle<X, W, W, Z>(T())) xwwz(T v) { return swizzle<X, W, W, Z>(v); }
template<typename T> inline decltype(swizzle<X, W, W, W>(T())) xwww(T v) { return swizzle<X, W, W, W>(v); }
template<typename T> inline decltype(swizzle<Y, X, X, X>(T())) yxxx(T v) { return swizzle<Y, X, X, X>(v); }
template<typename T> inline decltype(swizzle<Y, X, X, Y>(T())) yxxy(T v) { return swizzle<Y, X, X, Y>(v); }
template<typename T> inline decltype(swizzle<Y, X, X, Z>(T())) yxxz(T v) { return swizzle<Y, X, X, Z>(v); }
template<typename T> inline decltype(swizzle<Y, X, X, W>(T())) yxxw(T v) { return swizzle<Y, X, X, W>(v); }
template<typename T> inline decltype(swizzle<Y, X, Y, X>(T())) yxyx(T v) { return swizzle<Y, X, Y, X>(v); }
template<typename T> inline decltype(swizzle<Y, X, Y, Y>(T())) yxyy(T v) { return swizzle<Y, X, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Y, X, Y, Z>(T())) yxyz(T v) { return swizzle<Y, X, Y, Z>(v); }
template<typename T> inline decltype(swizzle<Y, X, Y, W>(T())) yxyw(T v) { return swizzle<Y, X, Y, W>(v); }
template<typename T> inline decltype(swizzle<Y, X, Z, X>(T())) yxzx(T v) { return swizzle<Y, X, Z, X>(v); }
template<typename T> inline decltype(swizzle<Y, X, Z, Y>(T())) yxzy(T v) { return swizzle<Y, X, Z, Y>(v); }
template<typename T> inline decltype(swizzle<Y, X, Z, Z>(T())) yxzz(T v) { return swizzle<Y, X, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Y, X, Z, W>(T())) yxzw(T v) { return swizzle<Y, X, Z, W>(v); }
template<typename T> inline decltype(swizzle<Y, X, W, X>(T())) yxwx(T v) { return swizzle<Y, X, W, X>(v); }
template<typename T> inline decltype(swizzle<Y, X, W, Y>(T())) yxwy(T v) { return swizzle<Y, X, W, Y>(v); }
template<typename T> inline decltype(swizzle<Y, X, W, Z>(T())) yxwz(T v) { return swizzle<Y, X, W, Z>(v); }
template<typename T> inline decltype(swizzle<Y, X, W, W>(T())) yxww(T v) { return swizzle<Y, X, W, W>(v); }
template<typename T> inline decltype(swizzle<Y, Y, X, X>(T())) yyxx(T v) { return swizzle<Y, Y, X, X>(v); }
template<typename T> inline decltype(swizzle<Y, Y, X, Y>(T())) yyxy(T v) { return swizzle<Y, Y, X, Y>(v); }
template<typename T> inline decltype(swizzle<Y, Y, X, Z>(T())) yyxz(T v) { return swizzle<Y, Y, X, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Y, X, W>(T())) yyxw(T v) { return swizzle<Y, Y, X, W>(v); }
template<typename T> inline decltype(swizzle<Y, Y, Y, X>(T())) yyyx(T v) { return swizzle<Y, Y, Y, X>(v); }
template<typename T> inline decltype(broadcast<Y>(T())) yyyy(T v) { return broadcast<Y>(v); }
template<typename T> inline decltype(swizzle<Y, Y, Y, Z>(T())) yyyz(T v) { return swizzle<Y, Y, Y, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Y, Y, W>(T())) yyyw(T v) { return swizzle<Y, Y, Y, W>(v); }
template<typename T> inline decltype(swizzle<Y, Y, Z, X>(T())) yyzx(T v) { return swizzle<Y, Y, Z, X>(v); }
template<typename T> inline decltype(swizzle<Y, Y, Z, Y>(T())) yyzy(T v) { return swizzle<Y, Y, Z, Y>(v); }
template<typename T> inline decltype(swizzle<Y, Y, Z, Z>(T())) yyzz(T v) { return swizzle<Y, Y, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Y, Z, W>(T())) yyzw(T v) { return swizzle<Y, Y, Z, W>(v); }
template<typename T> inline decltype(swizzle<Y, Y, W, X>(T())) yywx(T v) { return swizzle<Y, Y, W, X>(v); }
template<typename T> inline decltype(swizzle<Y, Y, W, Y>(T())) yywy(T v) { return swizzle<Y, Y, W, Y>(v); }
template<typename T> inline decltype(swizzle<Y, Y, W, Z>(T())) yywz(T v) { return swizzle<Y, Y, W, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Y, W, W>(T())) yyww(T v) { return swizzle<Y, Y, W, W>(v); }
template<typename T> inline decltype(swizzle<Y, Z, X, X>(T())) yzxx(T v) { return swizzle<Y, Z, X, X>(v); }
template<typename T> inline decltype(swizzle<Y, Z, X, Y>(T())) yzxy(T v) { return swizzle<Y, Z, X, Y>(v); }
template<typename T> inline decltype(swizzle<Y, Z, X, Z>(T())) yzxz(T v) { return swizzle<Y, Z, X, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Z, X, W>(T())) yzxw(T v) { return swizzle<Y, Z, X, W>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Y, X>(T())) yzyx(T v) { return swizzle<Y, Z, Y, X>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Y, Y>(T())) yzyy(T v) { return swizzle<Y, Z, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Y, Z>(T())) yzyz(T v) { return swizzle<Y, Z, Y, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Y, W>(T())) yzyw(T v) { return swizzle<Y, Z, Y, W>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Z, X>(T())) yzzx(T v) { return swizzle<Y, Z, Z, X>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Z, Y>(T())) yzzy(T v) { return swizzle<Y, Z, Z, Y>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Z, Z>(T())) yzzz(T v) { return swizzle<Y, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Z, W>(T())) yzzw(T v) { return swizzle<Y, Z, Z, W>(v); }
template<typename T> inline decltype(swizzle<Y, Z, W, X>(T())) yzwx(T v) { return swizzle<Y, Z, W, X>(v); }
template<typename T> inline decltype(swizzle<Y, Z, W, Y>(T())) yzwy(T v) { return swizzle<Y, Z, W, Y>(v); }
template<typename T> inline decltype(swizzle<Y, Z, W, Z>(T())) yzwz(T v) { return swizzle<Y, Z, W, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Z, W, W>(T())) yzww(T v) { return swizzle<Y, Z, W, W>(v); }
template<typename T> inline decltype(swizzle<Y, W, X, X>(T())) ywxx(T v) { return swizzle<Y, W, X, X>(v); }
template<typename T> inline decltype(swizzle<Y, W, X, Y>(T())) ywxy(T v) { return swizzle<Y, W, X, Y>(v); }
template<typename T> inline decltype(swizzle<Y, W, X, Z>(T())) ywxz(T v) { return swizzle<Y, W, X, Z>(v); }
template<typename T> inline decltype(swizzle<Y, W, X, W>(T())) ywxw(T v) { return swizzle<Y, W, X, W>(v); }
template<typename T> inline decltype(swizzle<Y, W, Y, X>(T())) ywyx(T v) { return swizzle<Y, W, Y, X>(v); }
template<typename T> inline decltype(swizzle<Y, W, Y, Y>(T())) ywyy(T v) { return swizzle<Y, W, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Y, W, Y, Z>(T())) ywyz(T v) { return swizzle<Y, W, Y, Z>(v); }
template<typename T> inline decltype(swizzle<Y, W, Y, W>(T())) ywyw(T v) { return swizzle<Y, W, Y, W>(v); }
template<typename T> inline decltype(swizzle<Y, W, Z, X>(T())) ywzx(T v) { return swizzle<Y, W, Z, X>(v); }
template<typename T> inline decltype(swizzle<Y, W, Z, Y>(T())) ywzy(T v) { return swizzle<Y, W, Z, Y>(v); }
template<typename T> inline decltype(swizzle<Y, W, Z, Z>(T())) ywzz(T v) { return swizzle<Y, W, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Y, W, Z, W>(T())) ywzw(T v) { return swizzle<Y, W, Z, W>(v); }
template<typename T> inline decltype(swizzle<Y, W, W, X>(T())) ywwx(T v) { return swizzle<Y, W, W, X>(v); }
template<typename T> inline decltype(swizzle<Y, W, W, Y>(T())) ywwy(T v) { return swizzle<Y, W, W, Y>(v); }
template<typename T> inline decltype(swizzle<Y, W, W, Z>(T())) ywwz(T v) { return swizzle<Y, W, W, Z>(v); }
template<typename T> inline decltype(swizzle<Y, W, W, W>(T())) ywww(T v) { return swizzle<Y, W, W, W>(v); }
template<typename T> inline decltype(swizzle<Z, X, X, X>(T())) zxxx(T v) { return swizzle<Z, X, X, X>(v); }
template<typename T> inline decltype(swizzle<Z, X, X, Y>(T())) zxxy(T v) { return swizzle<Z, X, X, Y>(v); }
template<typename T> inline decltype(swizzle<Z, X, X, Z>(T())) zxxz(T v) { return swizzle<Z, X, X, Z>(v); }
template<typename T> inline decltype(swizzle<Z, X, X, W>(T())) zxxw(T v) { return swizzle<Z, X, X, W>(v); }
template<typename T> inline decltype(swizzle<Z, X, Y, X>(T())) zxyx(T v) { return swizzle<Z, X, Y, X>(v); }
template<typename T> inline decltype(swizzle<Z, X, Y, Y>(T())) zxyy(T v) { return swizzle<Z, X, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Z, X, Y, Z>(T())) zxyz(T v) { return swizzle<Z, X, Y, Z>(v); }
template<typename T> inline decltype(swizzle<Z, X, Y, W>(T())) zxyw(T v) { return swizzle<Z, X, Y, W>(v); }
template<typename T> inline decltype(swizzle<Z, X, Z, X>(T())) zxzx(T v) { return swizzle<Z, X, Z, X>(v); }
template<typename T> inline decltype(swizzle<Z, X, Z, Y>(T())) zxzy(T v) { return swizzle<Z, X, Z, Y>(v); }
template<typename T> inline decltype(swizzle<Z, X, Z, Z>(T())) zxzz(T v) { return swizzle<Z, X, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Z, X, Z, W>(T())) zxzw(T v) { return swizzle<Z, X, Z, W>(v); }
template<typename T> inline decltype(swizzle<Z, X, W, X>(T())) zxwx(T v) { return swizzle<Z, X, W, X>(v); }
template<typename T> inline decltype(swizzle<Z, X, W, Y>(T())) zxwy(T v) { return swizzle<Z, X, W, Y>(v); }
template<typename T> inline decltype(swizzle<Z, X, W, Z>(T())) zxwz(T v) { return swizzle<Z, X, W, Z>(v); }
template<typename T> inline decltype(swizzle<Z, X, W, W>(T())) zxww(T v) { return swizzle<Z, X, W, W>(v); }
template<typename T> inline decltype(swizzle<Z, Y, X, X>(T())) zyxx(T v) { return swizzle<Z, Y, X, X>(v); }
template<typename T> inline decltype(swizzle<Z, Y, X, Y>(T())) zyxy(T v) { return swizzle<Z, Y, X, Y>(v); }
template<typename T> inline decltype(swizzle<Z, Y, X, Z>(T())) zyxz(T v) { return swizzle<Z, Y, X, Z>(v); }
template<typename T> inline decltype(swizzle<Z, Y, X, W>(T())) zyxw(T v) { return swizzle<Z, Y, X, W>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Y, X>(T())) zyyx(T v) { return swizzle<Z, Y, Y, X>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Y, Y>(T())) zyyy(T v) { return swizzle<Z, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Y, Z>(T())) zyyz(T v) { return swizzle<Z, Y, Y, Z>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Y, W>(T())) zyyw(T v) { return swizzle<Z, Y, Y, W>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Z, X>(T())) zyzx(T v) { return swizzle<Z, Y, Z, X>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Z, Y>(T())) zyzy(T v) { return swizzle<Z, Y, Z, Y>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Z, Z>(T())) zyzz(T v) { return swizzle<Z, Y, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Z, W>(T())) zyzw(T v) { return swizzle<Z, Y, Z, W>(v); }
template<typename T> inline decltype(swizzle<Z, Y, W, X>(T())) zywx(T v) { return swizzle<Z, Y, W, X>(v); }
template<typename T> inline decltype(swizzle<Z, Y, W, Y>(T())) zywy(T v) { return swizzle<Z, Y, W, Y>(v); }
template<typename T> inline decltype(swizzle<Z, Y, W, Z>(T())) zywz(T v) { return swizzle<Z, Y, W, Z>(v); }
template<typename T> inline decltype(swizzle<Z, Y, W, W>(T())) zyww(T v) { return swizzle<Z, Y, W, W>(v); }
template<typename T> inline decltype(swizzle<Z, Z, X, X>(T())) zzxx(T v) { return swizzle<Z, Z, X, X>(v); }
template<typename T> inline decltype(swizzle<Z, Z, X, Y>(T())) zzxy(T v) { return swizzle<Z, Z, X, Y>(v); }
template<typename T> inline decltype(swizzle<Z, Z, X, Z>(T())) zzxz(T v) { return swizzle<Z, Z, X, Z>(v); }
template<typename T> inline decltype(swizzle<Z, Z, X, W>(T())) zzxw(T v) { return swizzle<Z, Z, X, W>(v); }
template<typename T> inline decltype(swizzle<Z, Z, Y, X>(T())) zzyx(T v) { return swizzle<Z, Z, Y, X>(v); }
template<typename T> inline decltype(swizzle<Z, Z, Y, Y>(T())) zzyy(T v) { return swizzle<Z, Z, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Z, Z, Y, Z>(T())) zzyz(T v) { return swizzle<Z, Z, Y, Z>(v); }
template<typename T> inline decltype(swizzle<Z, Z, Y, W>(T())) zzyw(T v) { return swizzle<Z, Z, Y, W>(v); }
template<typename T> inline decltype(swizzle<Z, Z, Z, X>(T())) zzzx(T v) { return swizzle<Z, Z, Z, X>(v); }
template<typename T> inline decltype(swizzle<Z, Z, Z, Y>(T())) zzzy(T v) { return swizzle<Z, Z, Z, Y>(v); }
template<typename T> inline decltype(broadcast<Z>(T())) zzzz(T v) { return broadcast<Z>(v); }
template<typename T> inline decltype(swizzle<Z, Z, Z, W>(T())) zzzw(T v) { return swizzle<Z, Z, Z, W>(v); }
template<typename T> inline decltype(swizzle<Z, Z, W, X>(T())) zzwx(T v) { return swizzle<Z, Z, W, X>(v); }
template<typename T> inline decltype(swizzle<Z, Z, W, Y>(T())) zzwy(T v) { return swizzle<Z, Z, W, Y>(v); }
template<typename T> inline decltype(swizzle<Z, Z, W, Z>(T())) zzwz(T v) { return swizzle<Z, Z, W, Z>(v); }
template<typename T> inline decltype(swizzle<Z, Z, W, W>(T())) zzww(T v) { return swizzle<Z, Z, W, W>(v); }
template<typename T> inline decltype(swizzle<Z, W, X, X>(T())) zwxx(T v) { return swizzle<Z, W, X, X>(v); }
template<typename T> inline decltype(swizzle<Z, W, X, Y>(T())) zwxy(T v) { return swizzle<Z, W, X, Y>(v); }
template<typename T> inline decltype(swizzle<Z, W, X, Z>(T())) zwxz(T v) { return swizzle<Z, W, X, Z>(v); }
template<typename T> inline decltype(swizzle<Z, W, X, W>(T())) zwxw(T v) { return swizzle<Z, W, X, W>(v); }
template<typename T> inline decltype(swizzle<Z, W, Y, X>(T())) zwyx(T v) { return swizzle<Z, W, Y, X>(v); }
template<typename T> inline decltype(swizzle<Z, W, Y, Y>(T())) zwyy(T v) { return swizzle<Z, W, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Z, W, Y, Z>(T())) zwyz(T v) { return swizzle<Z, W, Y, Z>(v); }
template<typename T> inline decltype(swizzle<Z, W, Y, W>(T())) zwyw(T v) { return swizzle<Z, W, Y, W>(v); }
template<typename T> inline decltype(swizzle<Z, W, Z, X>(T())) zwzx(T v) { return swizzle<Z, W, Z, X>(v); }
template<typename T> inline decltype(swizzle<Z, W, Z, Y>(T())) zwzy(T v) { return swizzle<Z, W, Z, Y>(v); }
template<typename T> inline decltype(swizzle<Z, W, Z, Z>(T())) zwzz(T v) { return swizzle<Z, W, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Z, W, Z, W>(T())) zwzw(T v) { return swizzle<Z, W, Z, W>(v); }
template<typename T> inline decltype(swizzle<Z, W, W, X>(T())) zwwx(T v) { return swizzle<Z, W, W, X>(v); }
template<typename T> inline decltype(swizzle<Z, W, W, Y>(T())) zwwy(T v) { return swizzle<Z, W, W, Y>(v); }
template<typename T> inline decltype(swizzle<Z, W, W, Z>(T())) zwwz(T v) { return swizzle<Z, W, W, Z>(v); }
template<typename T> inline decltype(swizzle<Z, W, W, W>(T())) zwww(T v) { return swizzle<Z, W, W, W>(v); }
template<typename T> inline decltype(swizzle<W, X, X, X>(T())) wxxx(T v) { return swizzle<W, X, X, X>(v); }
template<typename T> inline decltype(swizzle<W, X, X, Y>(T())) wxxy(T v) { return swizzle<W, X, X, Y>(v); }
template<typename T> inline decltype(swizzle<W, X, X, Z>(T())) wxxz(T v) { return swizzle<W, X, X, Z>(v); }
template<typename T> inline decltype(swizzle<W, X, X, W>(T())) wxxw(T v) { return swizzle<W, X, X, W>(v); }
template<typename T> inline decltype(swizzle<W, X, Y, X>(T())) wxyx(T v) { return swizzle<W, X, Y, X>(v); }
template<typename T> inline decltype(swizzle<W, X, Y, Y>(T())) wxyy(T v) { return swizzle<W, X, Y, Y>(v); }
template<typename T> inline decltype(swizzle<W, X, Y, Z>(T())) wxyz(T v) { return swizzle<W, X, Y, Z>(v); }
template<typename T> inline decltype(swizzle<W, X, Y, W>(T())) wxyw(T v) { return swizzle<W, X, Y, W>(v); }
template<typename T> inline decltype(swizzle<W, X, Z, X>(T())) wxzx(T v) { return swizzle<W, X, Z, X>(v); }
template<typename T> inline decltype(swizzle<W, X, Z, Y>(T())) wxzy(T v) { return swizzle<W, X, Z, Y>(v); }
template<typename T> inline decltype(swizzle<W, X, Z, Z>(T())) wxzz(T v) { return swizzle<W, X, Z, Z>(v); }
template<typename T> inline decltype(swizzle<W, X, Z, W>(T())) wxzw(T v) { return swizzle<W, X, Z, W>(v); }
template<typename T> inline decltype(swizzle<W, X, W, X>(T())) wxwx(T v) { return swizzle<W, X, W, X>(v); }
template<typename T> inline decltype(swizzle<W, X, W, Y>(T())) wxwy(T v) { return swizzle<W, X, W, Y>(v); }
template<typename T> inline decltype(swizzle<W, X, W, Z>(T())) wxwz(T v) { return swizzle<W, X, W, Z>(v); }
template<typename T> inline decltype(swizzle<W, X, W, W>(T())) wxww(T v) { return swizzle<W, X, W, W>(v); }
template<typename T> inline decltype(swizzle<W, Y, X, X>(T())) wyxx(T v) { return swizzle<W, Y, X, X>(v); }
template<typename T> inline decltype(swizzle<W, Y, X, Y>(T())) wyxy(T v) { return swizzle<W, Y, X, Y>(v); }
template<typename T> inline decltype(swizzle<W, Y, X, Z>(T())) wyxz(T v) { return swizzle<W, Y, X, Z>(v); }
template<typename T> inline decltype(swizzle<W, Y, X, W>(T())) wyxw(T v) { return swizzle<W, Y, X, W>(v); }
template<typename T> inline decltype(swizzle<W, Y, Y, X>(T())) wyyx(T v) { return swizzle<W, Y, Y, X>(v); }
template<typename T> inline decltype(swizzle<W, Y, Y, Y>(T())) wyyy(T v) { return swizzle<W, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle<W, Y, Y, Z>(T())) wyyz(T v) { return swizzle<W, Y, Y, Z>(v); }
template<typename T> inline decltype(swizzle<W, Y, Y, W>(T())) wyyw(T v) { return swizzle<W, Y, Y, W>(v); }
template<typename T> inline decltype(swizzle<W, Y, Z, X>(T())) wyzx(T v) { return swizzle<W, Y, Z, X>(v); }
template<typename T> inline decltype(swizzle<W, Y, Z, Y>(T())) wyzy(T v) { return swizzle<W, Y, Z, Y>(v); }
template<typename T> inline decltype(swizzle<W, Y, Z, Z>(T())) wyzz(T v) { return swizzle<W, Y, Z, Z>(v); }
template<typename T> inline decltype(swizzle<W, Y, Z, W>(T())) wyzw(T v) { return swizzle<W, Y, Z, W>(v); }
template<typename T> inline decltype(swizzle<W, Y, W, X>(T())) wywx(T v) { return swizzle<W, Y, W, X>(v); }
template<typename T> inline decltype(swizzle<W, Y, W, Y>(T())) wywy(T v) { return swizzle<W, Y, W, Y>(v); }
template<typename T> inline decltype(swizzle<W, Y, W, Z>(T())) wywz(T v) { return swizzle<W, Y, W, Z>(v); }
template<typename T> inline decltype(swizzle<W, Y, W, W>(T())) wyww(T v) { return swizzle<W, Y, W, W>(v); }
template<typename T> inline decltype(swizzle<W, Z, X, X>(T())) wzxx(T v) { return swizzle<W, Z, X, X>(v); }
template<typename T> inline decltype(swizzle<W, Z, X, Y>(T())) wzxy(T v) { return swizzle<W, Z, X, Y>(v); }
template<typename T> inline decltype(swizzle<W, Z, X, Z>(T())) wzxz(T v) { return swizzle<W, Z, X, Z>(v); }
template<typename T> inline decltype(swizzle<W, Z, X, W>(T())) wzxw(T v) { return swizzle<W, Z, X, W>(v); }
template<typename T> inline decltype(swizzle<W, Z, Y, X>(T())) wzyx(T v) { return swizzle<W, Z, Y, X>(v); }
template<typename T> inline decltype(swizzle<W, Z, Y, Y>(T())) wzyy(T v) { return swizzle<W, Z, Y, Y>(v); }
template<typename T> inline decltype(swizzle<W, Z, Y, Z>(T())) wzyz(T v) { return swizzle<W, Z, Y, Z>(v); }
template<typename T> inline decltype(swizzle<W, Z, Y, W>(T())) wzyw(T v) { return swizzle<W, Z, Y, W>(v); }
template<typename T> inline decltype(swizzle<W, Z, Z, X>(T())) wzzx(T v) { return swizzle<W, Z, Z, X>(v); }
template<typename T> inline decltype(swizzle<W, Z, Z, Y>(T())) wzzy(T v) { return swizzle<W, Z, Z, Y>(v); }
template<typename T> inline decltype(swizzle<W, Z, Z, Z>(T())) wzzz(T v) { return swizzle<W, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle<W, Z, Z, W>(T())) wzzw(T v) { return swizzle<W, Z, Z, W>(v); }
template<typename T> inline decltype(swizzle<W, Z, W, X>(T())) wzwx(T v) { return swizzle<W, Z, W, X>(v); }
template<typename T> inline decltype(swizzle<W, Z, W, Y>(T())) wzwy(T v) { return swizzle<W, Z, W, Y>(v); }
template<typename T> inline decltype(swizzle<W, Z, W, Z>(T())) wzwz(T v) { return swizzle<W, Z, W, Z>(v); }
template<typename T> inline decltype(swizzle<W, Z, W, W>(T())) wzww(T v) { return swizzle<W, Z, W, W>(v); }
template<typename T> inline decltype(swizzle<W, W, X, X>(T())) wwxx(T v) { return swizzle<W, W, X, X>(v); }
template<typename T> inline decltype(swizzle<W, W, X, Y>(T())) wwxy(T v) { return swizzle<W, W, X, Y>(v); }
template<typename T> inline decltype(swizzle<W, W, X, Z>(T())) wwxz(T v) { return swizzle<W, W, X, Z>(v); }
template<typename T> inline decltype(swizzle<W, W, X, W>(T())) wwxw(T v) { return swizzle<W, W, X, W>(v); }
template<typename T> inline decltype(swizzle<W, W, Y, X>(T())) wwyx(T v) { return swizzle<W, W, Y, X>(v); }
template<typename T> inline decltype(swizzle<W, W, Y, Y>(T())) wwyy(T v) { return swizzle<W, W, Y, Y>(v); }
template<typename T> inline decltype(swizzle<W, W, Y, Z>(T())) wwyz(T v) { return swizzle<W, W, Y, Z>(v); }
template<typename T> inline decltype(swizzle<W, W, Y, W>(T())) wwyw(T v) { return swizzle<W, W, Y, W>(v); }
template<typename T> inline decltype(swizzle<W, W, Z, X>(T())) wwzx(T v) { return swizzle<W, W, Z, X>(v); }
template<typename T> inline decltype(swizzle<W, W, Z, Y>(T())) wwzy(T v) { return swizzle<W, W, Z, Y>(v); }
template<typename T> inline decltype(swizzle<W, W, Z, Z>(T())) wwzz(T v) { return swizzle<W, W, Z, Z>(v); }
template<typename T> inline decltype(swizzle<W, W, Z, W>(T())) wwzw(T v) { return swizzle<W, W, Z, W>(v); }
template<typename T> inline decltype(swizzle<W, W, W, X>(T())) wwwx(T v) { return swizzle<W, W, W, X>(v); }
template<typename T> inline decltype(swizzle<W, W, W, Y>(T())) wwwy(T v) { return swizzle<W, W, W, Y>(v); }
template<typename T> inline decltype(swizzle<W, W, W, Z>(T())) wwwz(T v) { return swizzle<W, W, W, Z>(v); }
template<typename T> inline decltype(broadcast<W>(T())) wwww(T v) { return broadcast<W>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> requires scalar::Arithmetic<T> inline decltype(zero<typename scalar::Storage<T, 4>::Type>()) zero4() { return zero<typename scalar::Storage<T, 4>::Type>(); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(set<typename scalar::Storage<T, 4>::Type, 1>(T())) set1(T s) { return set<typename scalar::Storage<T, 4>::Type, 1>(s); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(load<typename scalar::Storage<T, 4>::Type, 1>((const T*)nullptr)) load1(const T* v) { return load<typename scalar::Storage<T, 4>::Type, 1>(v); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(set<typename scalar::Storage<T, 4>::Type, 2>(T())) set2(T s) { return set<typename scalar::Storage<T, 4>::Type, 2>(s); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(set<typename scalar::Storage<T, 4>::Type>(T(), T())) set2(T x, T y) { return set<typename scalar::Storage<T, 4>::Type>(x, y); }
template<typename T> requires (!scalar::Arithmetic<T>) inline decltype(set<T>(T(), T())) set2(T x, T y) { return set<T>(x, y); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(load<typename scalar::Storage<T, 4>::Type, 2>((const T*)nullptr)) load2(const T* v) { return load<typename scalar::Storage<T, 4>::Type, 2>(v); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(set<typename scalar::Storage<T, 4>::Type, 3>(T())) set3(T s) { return set<typename scalar::Storage<T, 4>::Type, 3>(s); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(set<typename scalar::Storage<T, 4>::Type>(T(), T(), T())) set3(T x, T y, T z) { return set<typename scalar::Storage<T, 4>::Type>(x, y, z); }
template<typename T> requires (!scalar::Arithmetic<T>) inline decltype(set<T>(T(), T(), T())) set3(T x, T y, T z) { return set<T>(x, y, z); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(load<typename scalar::Storage<T, 4>::Type, 3>((const T*)nullptr)) load3(const T* v) { return load<typename scalar::Storage<T, 4>::Type, 3>(v); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(set<typename scalar::Storage<T, 4>::Type, 4>(T())) set4(T s) { return set<typename scalar::Storage<T, 4>::Type, 4>(s); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(set<typename scalar::Storage<T, 4>::Type>(T(), T(), T(), T())) set4(T x, T y, T z, T w) { return set<typename scalar::Storage<T, 4>::Type>(x, y, z, w); }
template<typename T> requires (!scalar::Arithmetic<T>) inline decltype(set<T>(T(), T(), T(), T())) set4(T x, T y, T z, T w) { return set<T>(x, y, z, w); }
template<typename T> requires scalar::Arithmetic<T> inline decltype(load<typename scalar::Storage<T, 4>::Type, 4>((const T*)nullptr)) load4(const T* v) { return load<typename scalar::Storage<T, 4>::Type, 4>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(cutoff<1>(T())) cutoff1(T v) { return cutoff<1>(v); }
template<int I, typename T, typename U> inline decltype(insert<I, 1>(T(), U())) insert1(T u, U v) { return insert<I, 1>(u, v); }
template<typename T, typename U> inline void store2(T u, U* v) { store<2>(u, v); }
template<typename T> inline decltype(cutoff<2>(T())) cutoff2(T v) { return cutoff<2>(v); }
template<typename T, typename U> inline decltype(insert<0, 2>(T(), U())) insert2(T u, U v) { return insert<0, 2>(u, v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T, typename U> inline void store3(T u, U* v) { store<3>(u, v); }
template<typename T> inline decltype(cutoff<3>(T())) cutoff3(T v) { return cutoff<3>(v); }
template<typename T, typename U> inline decltype(insert<0, 3>(T(), U())) insert3(T u, U v) { return insert<0, 3>(u, v); }
template<typename T, typename U> inline void store4(T u, U* v) { store<4>(u, v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T, typename U> inline decltype(pack<2>(T(), U())) pack2(T row0, U row1) { return pack<2>(row0, row1); }
template<typename T> inline decltype(unpack<2>(T())) unpack2(T m) { return unpack<2>(m); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(all<2>(T())) all2(T v) { return all<2>(v); }
template<typename T> inline decltype(any<2>(T())) any2(T v) { return any<2>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(all<3>(T())) all3(T v) { return all<3>(v); }
template<typename T> inline decltype(any<3>(T())) any3(T v) { return any<3>(v); }
template<typename T> inline decltype(all<4>(T())) all4(T v) { return all<4>(v); }
template<typename T> inline decltype(any<4>(T())) any4(T v) { return any<4>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T, typename U, typename V> inline decltype(min(max(T(), U()), V())) clamp(T u, U v, V w) { return min(max(u, v), w); }
template<typename T> inline decltype(neg<1>(T())) neg1(T v) { return neg<1>(v); }
template<typename T> inline decltype(neg<2>(T())) neg2(T v) { return neg<2>(v); }
template<typename T, typename U> inline decltype(div<2>(T(), U())) div2(T u, U v) { return div<2>(u, v); }
template<typename T, typename U, typename V> inline decltype(add(mul(T(), U()), V())) mulAdd(T u, U v, V w) { return add(mul(u, v), w); }
template<typename T, typename U, typename V> inline decltype(sub(mul(T(), U()), V())) mulSub(T u, U v, V w) { return sub(mul(u, v), w); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(neg<3>(T())) neg3(T v) { return neg<3>(v); }
template<typename T, typename U> inline decltype(div<3>(T(), U())) div3(T u, U v) { return div<3>(u, v); }
template<typename T> inline decltype(neg<4>(T())) neg4(T v) { return neg<4>(v); }
template<typename T, typename U> inline decltype(div<4>(T(), U())) div4(T u, U v) { return div<4>(u, v); }
#endif

#if SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(sqrt<1>(T())) sqrt1(T v) { return sqrt<1>(v); }
template<typename T> inline decltype(rcpSqrtApprox<1>(T())) rcpSqrtApprox1(T v) { return rcpSqrtApprox<1>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(hMin<2>(T())) hMin2(T v) { return hMin<2>(v); }
template<typename T> inline decltype(hMax<2>(T())) hMax2(T v) { return hMax<2>(v); }
template<typename T> inline decltype(hAdd<2>(T())) hAdd2(T v) { return hAdd<2>(v); }
template<typename T, typename U> inline decltype(dot<2>(T(), U())) dot2(T u, U v) { return dot<2>(u, v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(hMin<3>(T())) hMin3(T v) { return hMin<3>(v); }
template<typename T> inline decltype(hMax<3>(T())) hMax3(T v) { return hMax<3>(v); }
template<typename T> inline decltype(hAdd<3>(T())) hAdd3(T v) { return hAdd<3>(v); }
template<typename T, typename U> inline decltype(dot<3>(T(), U())) dot3(T u, U v) { return dot<3>(u, v); }
template<typename T> inline decltype(hMin<4>(T())) hMin4(T v) { return hMin<4>(v); }
template<typename T> inline decltype(hMax<4>(T())) hMax4(T v) { return hMax<4>(v); }
template<typename T> inline decltype(hAdd<4>(T())) hAdd4(T v) { return hAdd<4>(v); }
template<typename T, typename U> inline decltype(dot<4>(T(), U())) dot4(T u, U v) { return dot<4>(u, v); }
#endif

} // namespace simd
