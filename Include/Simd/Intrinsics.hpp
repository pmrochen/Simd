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
concept Arithmetic = (std::floating_point<T> || std::integral<T>);

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

} // namespace templates

#if SIMD_HAS_DOUBLE2
template<typename T> inline decltype(broadcast<X>(T())) xx(T v) { return broadcast<X>(v); }
template<typename T> inline decltype(broadcast<Y>(T())) yy(T v) { return broadcast<Y>(v); }
template<typename T> inline decltype(swizzle<X, Y>(T())) xy(T v) { return swizzle<X, Y>(v); }
template<typename T> inline decltype(swizzle<Y, X>(T())) yx(T v) { return swizzle<Y, X>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> inline decltype(broadcast<X>(T())) xxxx(T v) { return broadcast<X>(v); }
template<typename T> inline decltype(broadcast<Y>(T())) yyyy(T v) { return broadcast<Y>(v); }
template<typename T> inline decltype(broadcast<Z>(T())) zzzz(T v) { return broadcast<Z>(v); }
template<typename T> inline decltype(broadcast<W>(T())) wwww(T v) { return broadcast<W>(v); }
template<typename T> inline decltype(swizzle<X, Y, Y, Y>(T())) xyyy(T v) { return swizzle<X, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle<X, Y, X, Y>(T())) xyxy(T v) { return swizzle<X, Y, X, Y>(v); }
template<typename T> inline decltype(swizzle<X, X, Y, Y>(T())) xxyy(T v) { return swizzle<X, X, Y, Y>(v); }
template<typename T> inline decltype(swizzle<X, Z, Z, Z>(T())) xzzz(T v) { return swizzle<X, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle<X, W, W, W>(T())) xwww(T v) { return swizzle<X, W, W, W>(v); }
template<typename T> inline decltype(swizzle<X, Y, Z, Z>(T())) xyzz(T v) { return swizzle<X, Y, Z, Z>(v); }
template<typename T> inline decltype(swizzle<X, Y, W, W>(T())) xyww(T v) { return swizzle<X, Y, W, W>(v); }
template<typename T> inline decltype(swizzle<X, Z, W, W>(T())) xzww(T v) { return swizzle<X, Z, W, W>(v); }
template<typename T> inline decltype(swizzle<Y, X, X, X>(T())) yxxx(T v) { return swizzle<Y, X, X, X>(v); }
template<typename T> inline decltype(swizzle<Y, Z, Z, Z>(T())) yzzz(T v) { return swizzle<Y, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Y, Z, X, X>(T())) yzxx(T v) { return swizzle<Y, Z, X, X>(v); }
template<typename T> inline decltype(swizzle<Y, Z, X, W>(T())) yzxw(T v) { return swizzle<Y, Z, X, W>(v); }
template<typename T> inline decltype(swizzle<Y, Z, W, W>(T())) yzww(T v) { return swizzle<Y, Z, W, W>(v); }
template<typename T> inline decltype(swizzle<Y, W, W, W>(T())) ywww(T v) { return swizzle<Y, W, W, W>(v); }
template<typename T> inline decltype(swizzle<Y, X, Z, Z>(T())) yxzz(T v) { return swizzle<Y, X, Z, Z>(v); }
template<typename T> inline decltype(swizzle<Y, X, Z, W>(T())) yxzw(T v) { return swizzle<Y, X, Z, W>(v); }
template<typename T> inline decltype(swizzle<Y, X, W, Z>(T())) yxwz(T v) { return swizzle<Y, X, W, Z>(v); }
template<typename T> inline decltype(swizzle<Z, X, X, X>(T())) zxxx(T v) { return swizzle<Z, X, X, X>(v); }
template<typename T> inline decltype(swizzle<Z, X, Y, Y>(T())) zxyy(T v) { return swizzle<Z, X, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Z, X, Y, W>(T())) zxyw(T v) { return swizzle<Z, X, Y, W>(v); }
template<typename T> inline decltype(swizzle<Z, Y, Y, Y>(T())) zyyy(T v) { return swizzle<Z, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle<Z, Y, W, W>(T())) zyww(T v) { return swizzle<Z, Y, W, W>(v); }
template<typename T> inline decltype(swizzle<Z, Z, W, W>(T())) zzww(T v) { return swizzle<Z, Z, W, W>(v); }
template<typename T> inline decltype(swizzle<Z, W, W, W>(T())) zwww(T v) { return swizzle<Z, W, W, W>(v); }
template<typename T> inline decltype(swizzle<Z, W, X, Y>(T())) zwxy(T v) { return swizzle<Z, W, X, Y>(v); }
template<typename T> inline decltype(swizzle<Z, W, Z, W>(T())) zwzw(T v) { return swizzle<Z, W, Z, W>(v); }
template<typename T> inline decltype(swizzle<W, X, X, X>(T())) wxxx(T v) { return swizzle<W, X, X, X>(v); }
template<typename T> inline decltype(swizzle<W, Y, Y, Y>(T())) wyyy(T v) { return swizzle<W, Y, Y, Y>(v); }
template<typename T> inline decltype(swizzle<W, Z, Z, Z>(T())) wzzz(T v) { return swizzle<W, Z, Z, Z>(v); }
template<typename T> inline decltype(swizzle<W, Z, Y, X>(T())) wzyx(T v) { return swizzle<W, Z, Y, X>(v); }
#endif

#if SIMD_HAS_INT4 || SIMD_HAS_FLOAT4 || SIMD_HAS_DOUBLE4
template<typename T> requires Arithmetic<T> inline decltype(zero<typename templates::Storage<T, 4>::Type>()) zero4() { return zero<typename templates::Storage<T, 4>::Type>(); }
template<typename T> requires Arithmetic<T> inline decltype(set<typename templates::Storage<T, 4>::Type, 1>(T())) set1(T s) { return set<typename templates::Storage<T, 4>::Type, 1>(s); }
template<typename T> requires Arithmetic<T> inline decltype(load<typename templates::Storage<T, 4>::Type, 1>((const T*)nullptr)) load1(const T* v) { return load<typename templates::Storage<T, 4>::Type, 1>(v); }
template<typename T> requires Arithmetic<T> inline decltype(set<typename templates::Storage<T, 4>::Type, 2>(T())) set2(T s) { return set<typename templates::Storage<T, 4>::Type, 2>(s); }
template<typename T> requires Arithmetic<T> inline decltype(set<typename templates::Storage<T, 4>::Type>(T(), T())) set2(T x, T y) { return set<typename templates::Storage<T, 4>::Type>(x, y); }
template<typename T> requires (!Arithmetic<T>) inline decltype(set<T>(T(), T())) set2(T x, T y) { return set<T>(x, y); }
template<typename T> requires Arithmetic<T> inline decltype(load<typename templates::Storage<T, 4>::Type, 2>((const T*)nullptr)) load2(const T* v) { return load<typename templates::Storage<T, 4>::Type, 2>(v); }
template<typename T> requires Arithmetic<T> inline decltype(set<typename templates::Storage<T, 4>::Type, 3>(T())) set3(T s) { return set<typename templates::Storage<T, 4>::Type, 3>(s); }
template<typename T> requires Arithmetic<T> inline decltype(set<typename templates::Storage<T, 4>::Type>(T(), T(), T())) set3(T x, T y, T z) { return set<typename templates::Storage<T, 4>::Type>(x, y, z); }
template<typename T> requires (!Arithmetic<T>) inline decltype(set<T>(T(), T(), T())) set3(T x, T y, T z) { return set<T>(x, y, z); }
template<typename T> requires Arithmetic<T> inline decltype(load<typename templates::Storage<T, 4>::Type, 3>((const T*)nullptr)) load3(const T* v) { return load<typename templates::Storage<T, 4>::Type, 3>(v); }
template<typename T> requires Arithmetic<T> inline decltype(set<typename templates::Storage<T, 4>::Type, 4>(T())) set4(T s) { return set<typename templates::Storage<T, 4>::Type, 4>(s); }
template<typename T> requires Arithmetic<T> inline decltype(set<typename templates::Storage<T, 4>::Type>(T(), T(), T(), T())) set4(T x, T y, T z, T w) { return set<typename templates::Storage<T, 4>::Type>(x, y, z, w); }
template<typename T> requires (!Arithmetic<T>) inline decltype(set<T>(T(), T(), T(), T())) set4(T x, T y, T z, T w) { return set<T>(x, y, z, w); }
template<typename T> requires Arithmetic<T> inline decltype(load<typename templates::Storage<T, 4>::Type, 4>((const T*)nullptr)) load4(const T* v) { return load<typename templates::Storage<T, 4>::Type, 4>(v); }
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
