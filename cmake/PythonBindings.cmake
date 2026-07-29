if(pybind11_FOUND)
    pybind11_add_module(_traditional_dic
        bindings/python/module.cpp
        bindings/python/bind_core.cpp
        bindings/python/bind_io.cpp
        bindings/python/bind_subset.cpp
        bindings/python/bind_mesh.cpp
        bindings/python/bind_calibration.cpp
        bindings/python/bind_geometry.cpp
        bindings/python/bind_postprocess.cpp)
    target_link_libraries(_traditional_dic PRIVATE traditional_dic_core)
    add_custom_command(TARGET _traditional_dic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:_traditional_dic>
            "${CMAKE_CURRENT_SOURCE_DIR}/python/traditional_dic/$<TARGET_FILE_NAME:_traditional_dic>")
else()
    message(STATUS "pybind11 not found; Python module skipped.")
endif()
