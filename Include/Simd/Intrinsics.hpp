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

#if SIMD_HAS_DOUBLE2
template<typename T> inline decltype(broadcast(T())) xx(T v) { return broadcast<X>(v); }
template<typename T> inline decltype(broadcast(T())) yy(T v) { return broadcast<Y>(v); }
template<typename T> inline decltype(swizzle(T())) xy(T v) { return swizzle<X, Y>(v); }
template<typename T> inline decltype(swizzle(T())) yx(T v) { return swizzle<Y, X>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(broadcast(T())) xxxx(T v) { return broadcast<X>(v); }
template<typename T> inline decltype(broadcast(T())) yyyy(T v) { return broadcast<Y>(v); }
template<typename T> inline decltype(broadcast(T())) zzzz(T v) { return broadcast<Z>(v); }
template<typename T> inline decltype(broadcast(T())) wwww(T v) { return broadcast<W>(v); }
template<typename T> inline decltype(swizzle(T())) xyyy(T v) { return swizzle<X, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle(T())) xyxy(T v) { return swizzle<X, Y, X, Y>(v); }
template<typename T> inline decltype(swizzle(T())) xxyy(T v) { return swizzle<X, X, Y, Y>(v); }
template<typename T> inline decltype(swizzle(T())) xzzz(T v) { return swizzle<X, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle(T())) xwww(T v) { return swizzle<X, W, W, W>(v); }
template<typename T> inline decltype(swizzle(T())) xyzz(T v) { return swizzle<X, Y, Z, Z>(v); }
template<typename T> inline decltype(swizzle(T())) xyww(T v) { return swizzle<X, Y, W, W>(v); }
template<typename T> inline decltype(swizzle(T())) xzww(T v) { return swizzle<X, Z, W, W>(v); }
template<typename T> inline decltype(swizzle(T())) yxxx(T v) { return swizzle<Y, X, X, X>(v); }
template<typename T> inline decltype(swizzle(T())) yzzz(T v) { return swizzle<Y, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle(T())) yzxx(T v) { return swizzle<Y, Z, X, X>(v); }
template<typename T> inline decltype(swizzle(T())) yzxw(T v) { return swizzle<Y, Z, X, W>(v); }
template<typename T> inline decltype(swizzle(T())) yzww(T v) { return swizzle<Y, Z, W, W>(v); }
template<typename T> inline decltype(swizzle(T())) ywww(T v) { return swizzle<Y, W, W, W>(v); }
template<typename T> inline decltype(swizzle(T())) yxzz(T v) { return swizzle<Y, X, Z, Z>(v); }
template<typename T> inline decltype(swizzle(T())) yxzw(T v) { return swizzle<Y, X, Z, W>(v); }
template<typename T> inline decltype(swizzle(T())) yxwz(T v) { return swizzle<Y, X, W, Z>(v); }
template<typename T> inline decltype(swizzle(T())) zxxx(T v) { return swizzle<Z, X, X, X>(v); }
template<typename T> inline decltype(swizzle(T())) zxyy(T v) { return swizzle<Z, X, Y, Y>(v); }
template<typename T> inline decltype(swizzle(T())) zxyw(T v) { return swizzle<Z, X, Y, W>(v); }
template<typename T> inline decltype(swizzle(T())) zyyy(T v) { return swizzle<Z, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle(T())) zyww(T v) { return swizzle<Z, Y, W, W>(v); }
template<typename T> inline decltype(swizzle(T())) zzww(T v) { return swizzle<Z, Z, W, W>(v); }
template<typename T> inline decltype(swizzle(T())) zwww(T v) { return swizzle<Z, W, W, W>(v); }
template<typename T> inline decltype(swizzle(T())) zwxy(T v) { return swizzle<Z, W, X, Y>(v); }
template<typename T> inline decltype(swizzle(T())) zwzw(T v) { return swizzle<Z, W, Z, W>(v); }
template<typename T> inline decltype(swizzle(T())) wxxx(T v) { return swizzle<W, X, X, X>(v); }
template<typename T> inline decltype(swizzle(T())) wyyy(T v) { return swizzle<W, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle(T())) wzzz(T v) { return swizzle<W, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle(T())) wzyx(T v) { return swizzle<W, Z, Y, X>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T, typename U> inline decltype(set<T>(U())) set1(U s) { return set<T, 1>(s); }
template<typename T> inline decltype(cutoff(T())) cutoff1(T v) { return cutoff<1>(v); }
template<int I, typename T, typename U> inline decltype(insert(T(), U())) insert1(T u, U v) { return insert<I, 1>(u, v); }
template<typename T, typename U> inline decltype(set<T>(U())) set2(U s) { return set<T, 2>(s); }
template<typename T, typename U> inline decltype(set<T>(U(), U())) set2(U x, U y) { return set<T>(x, y); }
template<typename T, typename U> inline decltype(load<T>((const U*)nullptr)) load2(const U* v) { return load<T, 2>(v); }
template<typename T, typename U> inline void store2(T u, U* v) { store<2>(u, v); }
template<typename T> inline decltype(cutoff(T())) cutoff2(T v) { return cutoff<2>(v); }
template<typename T, typename U> inline decltype(insert(T(), U())) insert2(T u, U v) { return insert<0, 2>(u, v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T, typename U> inline decltype(set<T>(U())) set3(U s) { return set<T, 3>(s); }
template<typename T, typename U> inline decltype(set<T>(U(), U(), U())) set3(U x, U y, U z) { return set<T>(x, y, z); }
template<typename T, typename U> inline decltype(load<T>((const U*)nullptr)) load3(const U* v) { return load<T, 3>(v); }
template<typename T, typename U> inline void store3(T u, U* v) { store<3>(u, v); }
template<typename T> inline decltype(cutoff(T())) cutoff3(T v) { return cutoff<3>(v); }
template<typename T, typename U> inline decltype(insert(T(), U())) insert3(T u, U v) { return insert<0, 3>(u, v); }
template<typename T, typename U> inline decltype(set<T>(U())) set4(U s) { return set<T, 4>(s); }
template<typename T, typename U> inline decltype(set<T>(U(), U(), U(), U())) set4(U x, U y, U z, U w) { return set<T>(x, y, z, w); }
template<typename T, typename U> inline decltype(load<T>((const U*)nullptr)) load4(const U* v) { return load<T, 4>(v); }
template<typename T, typename U> inline void store4(T u, U* v) { store<4>(u, v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T, typename U> inline decltype(pack(T(), U())) pack2(T row0, U row1) { return pack<2>(row0, row1); }
template<typename T> inline decltype(unpack(T())) unpack2(T m) { return unpack<2>(m); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(all(T())) all2(T v) { return all<2>(v); }
template<typename T> inline decltype(any(T())) any2(T v) { return any<2>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(all(T())) all3(T v) { return all<3>(v); }
template<typename T> inline decltype(any(T())) any3(T v) { return any<3>(v); }
template<typename T> inline decltype(all(T())) all4(T v) { return all<4>(v); }
template<typename T> inline decltype(any(T())) any4(T v) { return any<4>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T, typename U, typename V> inline decltype(min(max(T(), U()), V())) clamp(T u, U v, V w) { return min(max(u, v), w); }
template<typename T> inline decltype(neg(T())) neg1(T v) { return neg<1>(v); }
template<typename T> inline decltype(neg(T())) neg2(T v) { return neg<2>(v); }
template<typename T, typename U> inline decltype(div(T(), U())) div2(T u, U v) { return div<2>(u, v); }
template<typename T, typename U, typename V> inline decltype(add(mul(T(), U()), V())) mulAdd(T u, U v, V w) { return add(mul(u, v), w); }
template<typename T, typename U, typename V> inline decltype(sub(mul(T(), U()), V())) mulSub(T u, U v, V w) { return sub(mul(u, v), w); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(neg(T())) neg3(T v) { return neg<3>(v); }
template<typename T, typename U> inline decltype(div(T(), U())) div3(T u, U v) { return div<3>(u, v); }
template<typename T> inline decltype(neg(T())) neg4(T v) { return neg<4>(v); }
template<typename T, typename U> inline decltype(div(T(), U())) div4(T u, U v) { return div<4>(u, v); }
#endif

#if SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(sqrt(T())) sqrt1(T v) { return sqrt<1>(v); }
template<typename T> inline decltype(rcpSqrtApprox(T())) rcpSqrtApprox1(T v) { return rcpSqrtApprox<1>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE2 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(hMin(T())) hMin2(T v) { return hMin<2>(v); }
template<typename T> inline decltype(hMax(T())) hMax2(T v) { return hMax<2>(v); }
template<typename T> inline decltype(hAdd(T())) hAdd2(T v) { return hAdd<2>(v); }
template<typename T, typename U> inline decltype(dot(T(), U())) dot2(T u, U v) { return dot<2>(u, v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(hMin(T())) hMin3(T v) { return hMin<3>(v); }
template<typename T> inline decltype(hMax(T())) hMax3(T v) { return hMax<3>(v); }
template<typename T> inline decltype(hAdd(T())) hAdd3(T v) { return hAdd<3>(v); }
template<typename T, typename U> inline decltype(dot(T(), U())) dot3(T u, U v) { return dot<3>(u, v); }
template<typename T> inline decltype(hMin(T())) hMin4(T v) { return hMin<4>(v); }
template<typename T> inline decltype(hMax(T())) hMax4(T v) { return hMax<4>(v); }
template<typename T> inline decltype(hAdd(T())) hAdd4(T v) { return hAdd<4>(v); }
template<typename T, typename U> inline decltype(dot(T(), U())) dot4(T u, U v) { return dot<4>(u, v); }
#endif

} // namespace simd
