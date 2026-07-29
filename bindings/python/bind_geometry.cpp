/**
 * @file bind_geometry.cpp
 * @brief Minimal implementation placeholder for geometry Python bindings.
 *
 * Responsibilities:
 * - Provide linkable definitions matching the public header.
 * - Keep complex DIC mathematics marked as TODO for later implementation.
 *
 * Inputs:
 * - Values supplied through the corresponding API.
 *
 * Outputs:
 * - Placeholder values or explicit not-implemented exceptions.
 *
 * Dependencies:
 * - Corresponding public header plus Eigen/OpenCV-ready module boundaries.
 *
 * TODO:
 * - Replace placeholders with validated Traditional-DIC algorithms.
 * - Add numerical tests and performance benchmarks.
 */

#include <dic/geometry/projection.hpp>
#include <dic/geometry/triangulation.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_geometry(pybind11::module_& m) {
    auto sub = m.def_submodule("geometry");

    py::class_<dic::TriangulationOptions>(sub, "TriangulationOptions")
        .def(py::init<>())
        .def_readwrite("max_reprojection_error", &dic::TriangulationOptions::max_reprojection_error)
        .def_readwrite("require_positive_depth", &dic::TriangulationOptions::require_positive_depth);

    py::class_<dic::TriangulationResult>(sub, "TriangulationResult")
        .def(py::init<>())
        .def_readwrite("point", &dic::TriangulationResult::point)
        .def_readwrite("mean_reprojection_error", &dic::TriangulationResult::mean_reprojection_error)
        .def_readwrite("max_reprojection_error", &dic::TriangulationResult::max_reprojection_error)
        .def_readwrite("observations_used", &dic::TriangulationResult::observations_used)
        .def_readwrite("valid", &dic::TriangulationResult::valid);

    sub.def("world_to_camera", &dic::world_to_camera, py::arg("point"), py::arg("camera"));
    sub.def("distort_normalized_point", &dic::distort_normalized_point, py::arg("normalized"), py::arg("distortion"));
    sub.def("project_point", &dic::project_point, py::arg("point"), py::arg("camera"));
    sub.def("reprojection_error", &dic::reprojection_error, py::arg("point"), py::arg("observation"), py::arg("camera"));
    sub.def("camera_depth", &dic::camera_depth, py::arg("point"), py::arg("camera"));
    sub.def("triangulate_points", &dic::triangulate_points, py::arg("observations"), py::arg("cameras"));
    sub.def("triangulate_points_checked",
            &dic::triangulate_points_checked,
            py::arg("observations"),
            py::arg("cameras"),
            py::arg("options") = dic::TriangulationOptions{});
}
