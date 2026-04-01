cmake_minimum_required(VERSION 3.25)

project(Simd LANGUAGES CXX)

set(SIMD_LIB_NAME "Simd")
add_library(${SIMD_LIB_NAME} INTERFACE)

target_include_directories(${SIMD_LIB_NAME} 
	INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:.>
)

target_compile_features(${SIMD_LIB_NAME} 
    INTERFACE 
        cxx_std_20
)
