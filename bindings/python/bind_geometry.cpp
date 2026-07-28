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

#include <dic/geometry/multiview_triangulation.hpp>
#include <dic/geometry/projection.hpp>
#include <dic/geometry/stereo_triangulation.hpp>

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

    py::class_<dic::StereoReconstructionResult>(sub, "StereoReconstructionResult")
        .def(py::init<>())
        .def_readwrite("points", &dic::StereoReconstructionResult::points)
        .def_readwrite("mean_reprojection_errors", &dic::StereoReconstructionResult::mean_reprojection_errors)
        .def_readwrite("valid_mask", &dic::StereoReconstructionResult::valid_mask)
        .def_readwrite("mean_reprojection_error", &dic::StereoReconstructionResult::mean_reprojection_error);

    sub.def("world_to_camera", &dic::world_to_camera, py::arg("point"), py::arg("camera"));
    sub.def("distort_normalized_point", &dic::distort_normalized_point, py::arg("normalized"), py::arg("distortion"));
    sub.def("project_point", &dic::project_point, py::arg("point"), py::arg("camera"));
    sub.def("reprojection_error", &dic::reprojection_error, py::arg("point"), py::arg("observation"), py::arg("camera"));
    sub.def("camera_depth", &dic::camera_depth, py::arg("point"), py::arg("camera"));
    sub.def("triangulate_multiview", &dic::triangulate_multiview, py::arg("observations"), py::arg("cameras"));
    sub.def("triangulate_multiview_checked",
            &dic::triangulate_multiview_checked,
            py::arg("observations"),
            py::arg("cameras"),
            py::arg("options") = dic::TriangulationOptions{});
    sub.def("triangulate_stereo", &dic::triangulate_stereo, py::arg("point_left"), py::arg("point_right"), py::arg("left"), py::arg("right"));
    sub.def("triangulate_stereo_checked",
            &dic::triangulate_stereo_checked,
            py::arg("point_left"),
            py::arg("point_right"),
            py::arg("left"),
            py::arg("right"),
            py::arg("options") = dic::TriangulationOptions{});
    sub.def("reconstruct_stereo_points",
            &dic::reconstruct_stereo_points,
            py::arg("left_points"),
            py::arg("right_points"),
            py::arg("left"),
            py::arg("right"),
            py::arg("options") = dic::TriangulationOptions{});
}
