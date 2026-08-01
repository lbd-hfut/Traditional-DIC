include(FetchContent)

find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

if(MSVC)
    find_package(yaml-cpp QUIET)
endif()
if(NOT yaml-cpp_FOUND)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0
    )
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

if(MSVC)
    find_package(OpenCV QUIET)
endif()

# pybind11: search conda site-packages first, then pip site-packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Lib/site-packages/pybind11/share/cmake/pybind11")
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_pip_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(_pybind11_pip_cmake_dir)
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_pip_cmake_dir}")
endif()
find_package(pybind11 QUIET)

if(MSVC)
	    find_package(GTest QUIET)
	endif()
	if(NOT GTest_FOUND)
	    FetchContent_Declare(
	        googletest
	        GIT_REPOSITORY https://github.com/google/googletest.git
	        GIT_TAG v1.14.0
	    )
	    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
	    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	    FetchContent_MakeAvailable(googletest)
	    set(GTest_FOUND TRUE)
	endif()
