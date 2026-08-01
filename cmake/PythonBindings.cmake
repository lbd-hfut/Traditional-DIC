if(pybind11_FOUND)
    pybind11_add_module(_traditional_dic bindings/python/module.cpp bindings/python/bind_subset.cpp bindings/python/bind_mesh.cpp bindings/python/bind_calibration.cpp bindings/python/bind_geometry.cpp bindings/python/bind_postprocess.cpp)
    target_link_libraries(_traditional_dic PRIVATE traditional_dic_core)
    # Output .pyd directly into the Python package directory so it is
    # importable as traditional_dic._traditional_dic without extra PYTHONPATH.
    set_target_properties(_traditional_dic PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/traditional_dic"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/traditional_dic"
    )
else()
    message(STATUS "pybind11 not found; Python module skipped.")
endif()
