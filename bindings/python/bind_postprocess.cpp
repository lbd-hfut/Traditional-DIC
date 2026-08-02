/**
 * @file bind_postprocess.cpp
 * @brief Minimal implementation placeholder for postprocess Python bindings.
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

#include <dic/postprocess/strain_3d.hpp>
#include <dic/postprocess/least_squares_strain.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void bind_postprocess(pybind11::module_& m) {
    auto sub = m.def_submodule("postprocess");

    pybind11::class_<dic::LeastSquaresStrain2D>(sub, "LeastSquaresStrain2D")
        .def_readonly("du_dx", &dic::LeastSquaresStrain2D::du_dx)
        .def_readonly("du_dy", &dic::LeastSquaresStrain2D::du_dy)
        .def_readonly("dv_dx", &dic::LeastSquaresStrain2D::dv_dx)
        .def_readonly("dv_dy", &dic::LeastSquaresStrain2D::dv_dy)
        .def_readonly("exx", &dic::LeastSquaresStrain2D::exx)
        .def_readonly("eyy", &dic::LeastSquaresStrain2D::eyy)
        .def_readonly("exy", &dic::LeastSquaresStrain2D::exy)
        .def_readonly("sample_count", &dic::LeastSquaresStrain2D::sample_count)
        .def_readonly("valid", &dic::LeastSquaresStrain2D::valid);

    sub.def("compute_least_squares_strain_2d", &dic::compute_least_squares_strain_2d,
            pybind11::arg("points"), pybind11::arg("displacement"), pybind11::arg("radius"),
            pybind11::arg("min_samples") = 6, pybind11::arg("green_lagrange") = true);
    sub.def("compute_mesh_least_squares_strain_2d", &dic::compute_mesh_least_squares_strain_2d,
            pybind11::arg("nodes"), pybind11::arg("displacement"), pybind11::arg("elements"),
            pybind11::arg("min_samples") = 3, pybind11::arg("green_lagrange") = true);

    pybind11::class_<dic::SurfaceStrain3D>(sub, "SurfaceStrain3D")
        .def(pybind11::init<>())
        .def_readwrite("F", &dic::SurfaceStrain3D::F)
        .def_readwrite("C", &dic::SurfaceStrain3D::C)
        .def_readwrite("J", &dic::SurfaceStrain3D::J)
        .def_readwrite("E", &dic::SurfaceStrain3D::E)
        .def_readwrite("e", &dic::SurfaceStrain3D::e)
        .def_readwrite("Emgn", &dic::SurfaceStrain3D::Emgn)
        .def_readwrite("emgn", &dic::SurfaceStrain3D::emgn)
        .def_readwrite("Epc1", &dic::SurfaceStrain3D::Epc1)
        .def_readwrite("Epc2", &dic::SurfaceStrain3D::Epc2)
        .def_readwrite("epc1", &dic::SurfaceStrain3D::epc1)
        .def_readwrite("epc2", &dic::SurfaceStrain3D::epc2)
        .def_readwrite("EShearMax", &dic::SurfaceStrain3D::EShearMax)
        .def_readwrite("eShearMax", &dic::SurfaceStrain3D::eShearMax)
        .def_readwrite("Eeq", &dic::SurfaceStrain3D::Eeq)
        .def_readwrite("eeq", &dic::SurfaceStrain3D::eeq)
        .def_readwrite("area", &dic::SurfaceStrain3D::area)
        .def_readwrite("d3", &dic::SurfaceStrain3D::d3)
        .def_readwrite("Lambda1", &dic::SurfaceStrain3D::Lambda1)
        .def_readwrite("Lambda2", &dic::SurfaceStrain3D::Lambda2)
        .def_readwrite("valid", &dic::SurfaceStrain3D::valid);

    sub.def("compute_surface_strain",
            &dic::compute_surface_strain,
            pybind11::arg("faces"),
            pybind11::arg("points_ref"),
            pybind11::arg("points_def"),
            pybind11::arg("valid_faces"));
}
