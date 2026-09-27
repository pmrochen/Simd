/*
 *	Name: WasmIntrinsics
 *	Author: Pawel Mrochen
 */

#pragma once

#ifdef __wasm_simd128__
#define SIMD_WASM 1
#endif

#ifdef SIMD_WASM

#include <wasm_simd128.h>
#include <type_traits>
#include <tuple>

#define SIMD_WASM_HAS_FLOAT4 0/*1*/ // #TODO Change to 1 after full implementation

namespace simd::wasm {

using float4 = v128_t;

constexpr int X = 0;
constexpr int Y = 1;
constexpr int Z = 2;
constexpr int W = 3;

// #TODO

template<typename T>
	requires std::is_same_v<T, v128_t>
inline T set(float x, float y, float z, float w)
{ 
	return wasm_f32x4_make(x, y, z, w); 
}

template<int I = 0>
	requires ((I & ~3) == 0)
inline float extract(v128_t v)
{
	return wasm_f32x4_extract_lane(v, I);
}

inline v128_t add(v128_t v1, v128_t v2)
{
	return wasm_f32x4_add(v1, v2);
}

inline v128_t sub(v128_t v1, v128_t v2)
{
	return wasm_f32x4_sub(v1, v2);
}

// #TODO

} // namespace simd::wasm

#endif /* SIMD_WASM */
