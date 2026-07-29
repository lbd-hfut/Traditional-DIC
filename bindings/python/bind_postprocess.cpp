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

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void bind_postprocess(pybind11::module_& m) {
    auto sub = m.def_submodule("postprocess");

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
