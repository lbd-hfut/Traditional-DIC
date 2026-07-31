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

find_package(yaml-cpp QUIET)
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

find_package(OpenCV QUIET)

find_package(Ceres QUIET)
if(NOT Ceres_FOUND AND DEFINED CONDA_PREFIX)
    find_path(CERES_INCLUDE_DIR ceres/ceres.h PATHS "$ENV{CONDA_PREFIX}/Library/include" NO_DEFAULT_PATH)
    find_library(CERES_LIBRARY NAMES ceres PATHS "$ENV{CONDA_PREFIX}/Library/lib" NO_DEFAULT_PATH)
    if(CERES_INCLUDE_DIR AND CERES_LIBRARY)
        set(Ceres_FOUND TRUE)
        set(CERES_LIBRARIES "${CERES_LIBRARY}")
        set(CERES_INCLUDE_DIRS "${CERES_INCLUDE_DIR}")
        message(STATUS "Using manual Ceres from $ENV{CONDA_PREFIX}/Library")
    endif()
endif()

# pybind11 from conda site-packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 QUIET)

find_package(GTest QUIET)
