# Simd

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CACHE{SIMD_LIB_VERSION})
	string(TIMESTAMP VER_YEAR "%y")
	string(TIMESTAMP VER_MONTH "%m")
	math(EXPR VER_MONTH "${VER_MONTH} + 0")
    set(SIMD_LIB_VERSION "${VER_YEAR}.${VER_MONTH}" CACHE STRING "Project version (auto-set at first configure)")
endif()

project(Simd
	VERSION ${SIMD_LIB_VERSION}
	LANGUAGES CXX
)

set(SIMD_LIB_NAME "Simd")

add_library(${SIMD_LIB_NAME} INTERFACE)
add_library(Core::${SIMD_LIB_NAME} ALIAS ${SIMD_LIB_NAME})

set(SIMD_HEADERS
	Include/Simd/AvxIntrinsics.hpp
	Include/Simd/Functional.hpp
	Include/Simd/Intrinsics.hpp
	Include/Simd/Simd.hpp
	Include/Simd/SseIntrinsics.hpp
	Include/Simd/Tuple.hpp
)

target_sources(${SIMD_LIB_NAME}
	INTERFACE
    	FILE_SET HEADERS
		BASE_DIRS Include
    	FILES
			${SIMD_HEADERS}
)

#source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/Include" FILES ${SIMD_HEADERS})

target_compile_features(${SIMD_LIB_NAME} 
    INTERFACE 
        cxx_std_20
)
