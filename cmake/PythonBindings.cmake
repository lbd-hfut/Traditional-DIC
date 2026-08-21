if(pybind11_FOUND)
    pybind11_add_module(_traditional_dic NO_EXTRAS
        bindings/python/module.cpp
        bindings/python/bind_core.cpp
        bindings/python/bind_io.cpp
        bindings/python/bind_subset.cpp
        bindings/python/bind_mesh.cpp
        bindings/python/bind_calibration.cpp
        bindings/python/bind_geometry.cpp
        bindings/python/bind_reconstruction.cpp
        bindings/python/bind_postprocess.cpp
        bindings/python/bind_surface_outlier_cleaning.cpp
        bindings/python/bind_visualization.cpp)
    target_link_libraries(_traditional_dic PRIVATE traditional_dic_core)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND MINGW)
        # Keep the extension independent of the MSYS2 libstdc++ runtime used by OpenCV.
        # Linux only: static libstdc++ duplicates the locale facet-id registry inside the
        # extension while OpenCV still pulls the shared libstdc++, so iostream formatting
        # (e.g. io.save_displacement_csv) crashes with a wrong facet (segfault). Link the
        # shared runtime there like every other library in the process.
        target_link_options(_traditional_dic PRIVATE -static-libstdc++ -static-libgcc)
    endif()
    # Keep the existing source-tree development workflow, but let
    # scikit-build-core own the install-tree artifact used for wheels.
    if(NOT SKBUILD)
        add_custom_command(TARGET _traditional_dic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:_traditional_dic>
                "${CMAKE_CURRENT_SOURCE_DIR}/python/traditional_dic/$<TARGET_FILE_NAME:_traditional_dic>")
    endif()
    install(TARGETS _traditional_dic LIBRARY DESTINATION traditional_dic)
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/config/"
        DESTINATION traditional_dic/resources/config
        FILES_MATCHING PATTERN "*.yaml")
else()
    message(STATUS "pybind11 not found; Python module skipped.")
endif()
