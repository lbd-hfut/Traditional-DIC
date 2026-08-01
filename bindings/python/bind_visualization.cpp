#include <dic/visualization/surface_field.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<Eigen::Vector3d> points_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> points)
{
    const auto buf = points.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::invalid_argument("points must have shape (n, 3)");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    std::vector<Eigen::Vector3d> out(static_cast<std::size_t>(buf.shape[0]));
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = Eigen::Vector3d(ptr[3 * i], ptr[3 * i + 1], ptr[3 * i + 2]);
    }
    return out;
}

std::vector<std::array<std::int64_t, 3>> faces_from_array(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> faces)
{
    const auto buf = faces.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::invalid_argument("faces must have shape (m, 3)");
    }
    const auto* ptr = static_cast<const std::int64_t*>(buf.ptr);
    std::vector<std::array<std::int64_t, 3>> out(static_cast<std::size_t>(buf.shape[0]));
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = {ptr[3 * i], ptr[3 * i + 1], ptr[3 * i + 2]};
    }
    return out;
}

std::vector<double> values_from_array(
    py::array_t<double, py::array::c_style | py::array::forcecast> values)
{
    const auto buf = values.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("point_values must have shape (n,)");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + static_cast<std::size_t>(buf.shape[0]));
}

} // namespace

void bind_visualization(py::module_& m)
{
    auto sub = m.def_submodule("visualization");
    sub.def(
        "prepare_surface_field",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> faces,
           py::array_t<double, py::array::c_style | py::array::forcecast> point_values) {
            auto point_vec = points_from_array(points);
            auto face_vec = faces_from_array(faces);
            auto value_vec = values_from_array(point_values);

            py::gil_scoped_release release;
            const auto prepared = dic::visualization::prepare_surface_field(point_vec, face_vec, value_vec);
            py::gil_scoped_acquire acquire;

            const auto face_count = static_cast<py::ssize_t>(prepared.faces.size());
            py::array_t<std::int64_t> out_faces({face_count, py::ssize_t{3}});
            py::array_t<double> centers({face_count, py::ssize_t{3}});
            py::array_t<double> values({face_count});
            py::array_t<std::uint8_t> valid({face_count});

            auto* face_ptr = static_cast<std::int64_t*>(out_faces.request().ptr);
            auto* center_ptr = static_cast<double*>(centers.request().ptr);
            auto* value_ptr = static_cast<double*>(values.request().ptr);
            auto* valid_ptr = static_cast<std::uint8_t*>(valid.request().ptr);
            for (py::ssize_t i = 0; i < face_count; ++i) {
                const auto idx = static_cast<std::size_t>(i);
                for (py::ssize_t j = 0; j < 3; ++j) {
                    face_ptr[3 * idx + static_cast<std::size_t>(j)] = prepared.faces[idx][static_cast<std::size_t>(j)];
                    center_ptr[3 * idx + static_cast<std::size_t>(j)] = prepared.face_centers[idx](j);
                }
                value_ptr[idx] = prepared.face_values[idx];
                valid_ptr[idx] = prepared.valid_faces[idx];
            }

            py::dict out;
            out["faces"] = out_faces;
            out["face_centers"] = centers;
            out["face_values"] = values;
            out["valid_faces"] = valid;
            return out;
        },
        py::arg("points"),
        py::arg("faces"),
        py::arg("point_values"));
}
