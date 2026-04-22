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

target_include_directories(${SIMD_LIB_NAME} 	# to remove after moving header files to Include folder
	INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

set(SIMD_HEADERS
	AvxIntrinsics.hpp
	Float4.hpp
	Functional.hpp
	Intrinsics.hpp
	Simd.hpp
	SseIntrinsics.hpp
)

target_sources(${SIMD_LIB_NAME}
	INTERFACE
    	FILE_SET HEADERS
		#BASE_DIRS Include
    	FILES
			${SIMD_HEADERS}
)

#source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/Include" FILES ${SIMD_HEADERS})

target_compile_features(${SIMD_LIB_NAME} 
    INTERFACE 
        cxx_std_20
)
