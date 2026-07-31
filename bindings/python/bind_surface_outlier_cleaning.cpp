#include <dic/reconstruction/surface_outlier_cleaning.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

void bind_surface_outlier_cleaning(py::module_& m)
{
    m.def(
        "clean_surface_outliers_cpp",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> reference,
           py::array_t<double, py::array::c_style | py::array::forcecast> deformed,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> faces,
           py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> initial_valid,
           int neighbor_count, double distance_sigma, double displacement_sigma, double face_edge_scale) {
            auto rb = reference.request();
            auto db = deformed.request();
            auto fb = faces.request();
            auto vb = initial_valid.request();
            if (rb.ndim != 2 || rb.shape[1] != 3 || db.ndim != 2 || db.shape[1] != 3 ||
                db.shape[0] != rb.shape[0] || fb.ndim != 2 || fb.shape[1] != 3 ||
                vb.ndim != 1 || vb.shape[0] != rb.shape[0]) {
                throw std::invalid_argument("invalid surface cleaning array shapes");
            }
            const auto n = static_cast<std::size_t>(rb.shape[0]);
            std::vector<Eigen::Vector3d> ref(n), def(n);
            const auto* rp = static_cast<const double*>(rb.ptr);
            const auto* dp = static_cast<const double*>(db.ptr);
            for (std::size_t i = 0; i < n; ++i) {
                ref[i] = Eigen::Vector3d(rp[3 * i], rp[3 * i + 1], rp[3 * i + 2]);
                def[i] = Eigen::Vector3d(dp[3 * i], dp[3 * i + 1], dp[3 * i + 2]);
            }
            const auto* fp = static_cast<const std::int64_t*>(fb.ptr);
            std::vector<std::array<std::int64_t, 3>> triangles(static_cast<std::size_t>(fb.shape[0]));
            for (std::size_t i = 0; i < triangles.size(); ++i) {
                triangles[i] = {fp[3 * i], fp[3 * i + 1], fp[3 * i + 2]};
            }
            const auto* vp = static_cast<const std::uint8_t*>(vb.ptr);
            std::vector<std::uint8_t> valid(vp, vp + n);
            dic::SurfaceOutlierCleaningOptions options;
            options.neighbor_count = neighbor_count;
            options.distance_sigma = distance_sigma;
            options.displacement_sigma = displacement_sigma;
            options.face_edge_scale = face_edge_scale;
            py::gil_scoped_release release;
            const auto cleaned = dic::clean_surface_outliers(ref, def, triangles, valid, options);
            py::gil_scoped_acquire acquire;
            py::array_t<std::uint8_t> point_mask(cleaned.valid_points.size());
            py::array_t<std::uint8_t> face_mask(cleaned.valid_faces.size());
            std::copy(cleaned.valid_points.begin(), cleaned.valid_points.end(), static_cast<std::uint8_t*>(point_mask.request().ptr));
            std::copy(cleaned.valid_faces.begin(), cleaned.valid_faces.end(), static_cast<std::uint8_t*>(face_mask.request().ptr));
            py::dict out;
            out["valid_points"] = point_mask;
            out["valid_faces"] = face_mask;
            out["removed_points"] = cleaned.removed_points;
            out["removed_faces"] = cleaned.removed_faces;
            return out;
        },
        py::arg("reference"), py::arg("deformed"), py::arg("faces"), py::arg("initial_valid"),
        py::arg("neighbor_count") = 8, py::arg("distance_sigma") = 6.0,
        py::arg("displacement_sigma") = 6.0, py::arg("face_edge_scale") = 4.0);
}
