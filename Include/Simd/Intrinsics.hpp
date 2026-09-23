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

#ifndef SIMD_HAS_FLOAT4
#define SIMD_HAS_FLOAT4 0
#endif

#ifndef SIMD_HAS_INT4
#define SIMD_HAS_INT4 0
#endif

namespace simd {

#ifdef SIMD_SSE
using namespace sse;
#endif

#ifdef SIMD_AVX
using namespace avx;
#endif

#if defined(SIMD_SSE) /*|| defined(SIMD_AVX)*/

template<typename T> inline decltype(set(T)) set1(T s) { return set<1>(s); }
template<typename T> inline decltype(set(T)) set2(T s) { return set<2>(s); }
template<typename T> inline decltype(set(T, T)) set2(T x, T y) { return set(x, y); }
template<typename T> inline decltype(set(T)) set3(T s) { return set<3>(s); }
template<typename T> inline decltype(set(T, T, T)) set3(T x, T y, T z) { return set(x, y, z); }
template<typename T> inline decltype(set(T)) set4(T s) { return set<4>(s); }
template<typename T> inline decltype(set(T, T, T, T)) set4(T x, T y, T z, T w) { return set(x, y, z, w); }
template<typename T> inline decltype(load(const T*)) load2(const T* v) { return load<2>(v); }
template<typename T> inline decltype(load(const T*)) load3(const T* v) { return load<3>(v); }
template<typename T> inline decltype(load(const T*)) load4(const T* v) { return load<4>(v); }
template<typename T, typename U> inline void store2(T u, U* v) { store<2>(u, v); }
template<typename T, typename U> inline void store3(T u, U* v) { store<3>(u, v); }
template<typename T, typename U> inline void store4(T u, U* v) { store<4>(u, v); }
template<typename T> inline T cutoff1(T v) { return cutoff<1>(v); }
template<typename T> inline T cutoff2(T v) { return cutoff<2>(v); }
template<typename T> inline T cutoff3(T v) { return cutoff<3>(v); }
template<int I, typename T> inline T insert1(T u, T v) { return insert<I, 1>(u, v); }
template<typename T> inline T insert2(T u, T v) { return insert<0, 2>(u, v); }
template<typename T> inline T insert3(T u, T v) { return insert<0, 3>(u, v); }
template<typename T> inline T xxxx(T v) { return broadcast<X>(v); }
template<typename T> inline T yyyy(T v) { return broadcast<Y>(v); }
template<typename T> inline T zzzz(T v) { return broadcast<Z>(v); }
template<typename T> inline T wwww(T v) { return broadcast<W>(v); }
template<typename T> inline T xyyy(T v) { return swizzle<X, Y, Y, Y>(v); }
template<typename T> inline T xyxy(T v) { return swizzle<X, Y, X, Y>(v); }
template<typename T> inline T xxyy(T v) { return swizzle<X, X, Y, Y>(v); }
template<typename T> inline T xzzz(T v) { return swizzle<X, Z, Z, Z>(v); }
template<typename T> inline T xwww(T v) { return swizzle<X, W, W, W>(v); }
template<typename T> inline T xyzz(T v) { return swizzle<X, Y, Z, Z>(v); }
template<typename T> inline T xyww(T v) { return swizzle<X, Y, W, W>(v); }
template<typename T> inline T xzww(T v) { return swizzle<X, Z, W, W>(v); }
template<typename T> inline T yxxx(T v) { return swizzle<Y, X, X, X>(v); }
template<typename T> inline T yzzz(T v) { return swizzle<Y, Z, Z, Z>(v); }
template<typename T> inline T yzxx(T v) { return swizzle<Y, Z, X, X>(v); }
template<typename T> inline T yzxw(T v) { return swizzle<Y, Z, X, W>(v); }
template<typename T> inline T yzww(T v) { return swizzle<Y, Z, W, W>(v); }
template<typename T> inline T ywww(T v) { return swizzle<Y, W, W, W>(v); }
template<typename T> inline T yxzz(T v) { return swizzle<Y, X, Z, Z>(v); }
template<typename T> inline T yxzw(T v) { return swizzle<Y, X, Z, W>(v); }
template<typename T> inline T yxwz(T v) { return swizzle<Y, X, W, Z>(v); }
template<typename T> inline T zxxx(T v) { return swizzle<Z, X, X, X>(v); }
template<typename T> inline T zxyy(T v) { return swizzle<Z, X, Y, Y>(v); }
template<typename T> inline T zxyw(T v) { return swizzle<Z, X, Y, W>(v); }
template<typename T> inline T zyyy(T v) { return swizzle<Z, Y, Y, Y>(v); }
template<typename T> inline T zyww(T v) { return swizzle<Z, Y, W, W>(v); }
template<typename T> inline T zzww(T v) { return swizzle<Z, Z, W, W>(v); }
template<typename T> inline T zwww(T v) { return swizzle<Z, W, W, W>(v); }
template<typename T> inline T zwxy(T v) { return swizzle<Z, W, X, Y>(v); }
template<typename T> inline T zwzw(T v) { return swizzle<Z, W, Z, W>(v); }
template<typename T> inline T wxxx(T v) { return swizzle<W, X, X, X>(v); }
template<typename T> inline T wyyy(T v) { return swizzle<W, Y, Y, Y>(v); }
template<typename T> inline T wzzz(T v) { return swizzle<W, Z, Z, Z>(v); }
template<typename T> inline T wzyx(T v) { return swizzle<W, Z, Y, X>(v); }
template<typename T> inline T not4(T v) { return logicalNot(v); }
template<typename T> inline T and4(T v) { return logicalAnd(v); }
template<typename T> inline T or4(T v) { return logicalOr(v); }
template<typename T> inline bool all2(T v) { return all<2>(v); }
template<typename T> inline bool all3(T v) { return all<3>(v); }
template<typename T> inline bool all4(T v) { return all<4>(v); }
template<typename T> inline bool any2(T v) { return any<2>(v); }
template<typename T> inline bool any3(T v) { return any<3>(v); }
template<typename T> inline bool any4(T v) { return any<4>(v); }
template<typename T> inline T min4(T u, T v) { return min(u, v); }
template<typename T> inline T max4(T u, T v) { return max(u, v); }
template<typename T> inline T neg1(T v) { return negate<1>(v); }
template<typename T> inline T neg2(T v) { return negate<2/*XY*/>(v); }
template<typename T> inline T neg3(T v) { return negate<3/*XYZ*/>(v); }
template<typename T> inline T neg4(T v) { return negate<4/*XYZW*/>(v); }
template<typename T> inline T abs4(T v) { return abs(v); }
template<typename T> inline T add4(T u, T v) { return add(u, v); }
template<typename T> inline T sub4(T u, T v) { return subtract(u, v); }
template<typename T> inline T subAdd4(T u, T v) { return subAdd(u, v); }
template<typename T> inline T mul4(T u, T v) { return multiply(u, v); }
template<typename T> inline T div2(T u, T v) { return divide<2>(u, v); }
template<typename T> inline T div3(T u, T v) { return divide<3>(u, v); }
template<typename T> inline T div4(T u, T v) { return divide<4>(u, v); }
template<typename T> inline T mulAdd4(T u, T v, T w) { return multiplyAdd(u, v, w); }
template<typename T> inline T mulSub4(T u, T v, T w) { return multiplySub(u, v, w); }
template<typename T> inline T sqrt1(T v) { return sqrt<1>(v); }
template<typename T> inline T rcpSqrtApprox1(T v) { return rcpSqrtApprox<1>(v); }
template<typename T> inline T hMin2(T v) { return hMin<2>(v); }
template<typename T> inline T hMin3(T v) { return hMin<3>(v); }
template<typename T> inline T hMin4(T v) { return hMin<4>(v); }
template<typename T> inline T hMax2(T v) { return hMax<2>(v); }
template<typename T> inline T hMax3(T v) { return hMax<3>(v); }
template<typename T> inline T hMax4(T v) { return hMax<4>(v); }
template<typename T> inline T hAdd2(T v) { return hAdd<2>(v); }
template<typename T> inline T hAdd3(T v) { return hAdd<3>(v); }
template<typename T> inline T hAdd4(T v) { return hAdd<4>(v); }
template<typename T> inline T dot2(T u, T v) { return dot<2>(u, v); }
template<typename T> inline T dot3(T u, T v) { return dot<3>(u, v); }
template<typename T> inline T dot4(T u, T v) { return dot<4>(u, v); }

#endif

} // namespace simd
