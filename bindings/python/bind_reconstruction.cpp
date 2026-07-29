#include <dic/reconstruction/displacement_3d.hpp>
#include <dic/reconstruction/shape_reconstruction.hpp>
#include <dic/reconstruction/stereo_dic.hpp>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using UInt8Array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

void require_nx2(const DoubleArray& array, const char* name)
{
    if (array.ndim() != 2 || array.shape(1) != 2) {
        throw std::invalid_argument(std::string(name) + " must have shape (n, 2).");
    }
}

void require_n(const py::array& array, py::ssize_t n, const char* name)
{
    if (array.ndim() != 1 || array.shape(0) != n) {
        throw std::invalid_argument(std::string(name) + " must have shape (n,).");
    }
}

template <typename Rows>
bool finite_row2(const Rows& rows, py::ssize_t i)
{
    return std::isfinite(rows(i, 0)) && std::isfinite(rows(i, 1));
}

dic::StereoDICResult reconstruct_stereo_fields_batch(
    const DoubleArray& xy,
    const DoubleArray& reference_disparity,
    const DoubleArray& left_temporal,
    const DoubleArray& deformed_disparity,
    const UInt8Array& valid,
    const DoubleArray& corr_ref,
    const DoubleArray& corr_left_temporal,
    const DoubleArray& corr_deformed_disparity,
    const dic::CameraModel& left_camera,
    const dic::CameraModel& right_camera,
    const dic::StereoDIC::Options& options)
{
    require_nx2(xy, "xy");
    require_nx2(reference_disparity, "reference_disparity");
    require_nx2(left_temporal, "left_temporal");
    require_nx2(deformed_disparity, "deformed_disparity");
    const py::ssize_t n = xy.shape(0);
    if (reference_disparity.shape(0) != n || left_temporal.shape(0) != n ||
        deformed_disparity.shape(0) != n) {
        throw std::invalid_argument("All field arrays must have the same row count.");
    }
    require_n(valid, n, "valid");
    require_n(corr_ref, n, "corr_ref");
    require_n(corr_left_temporal, n, "corr_left_temporal");
    require_n(corr_deformed_disparity, n, "corr_deformed_disparity");

    const auto xy_rows = xy.unchecked<2>();
    const auto ref_rows = reference_disparity.unchecked<2>();
    const auto left_rows = left_temporal.unchecked<2>();
    const auto def_rows = deformed_disparity.unchecked<2>();
    const auto valid_rows = valid.unchecked<1>();
    const auto corr_ref_rows = corr_ref.unchecked<1>();
    const auto corr_left_rows = corr_left_temporal.unchecked<1>();
    const auto corr_def_rows = corr_deformed_disparity.unchecked<1>();

    std::vector<dic::PointObservation> left_obs;
    std::vector<dic::PointObservation> right_obs;
    left_obs.reserve(static_cast<std::size_t>(n));
    right_obs.reserve(static_cast<std::size_t>(n));

    for (py::ssize_t i = 0; i < n; ++i) {
        if (valid_rows(i) == 0 || !finite_row2(xy_rows, i) || !finite_row2(ref_rows, i) ||
            !finite_row2(left_rows, i) || !finite_row2(def_rows, i) ||
            !std::isfinite(corr_ref_rows(i)) || !std::isfinite(corr_left_rows(i)) ||
            !std::isfinite(corr_def_rows(i))) {
            continue;
        }

        const double x = xy_rows(i, 0);
        const double y = xy_rows(i, 1);
        const Eigen::Vector2d p_r0(x + ref_rows(i, 0), y + ref_rows(i, 1));
        const Eigen::Vector2d p_l1(x + left_rows(i, 0), y + left_rows(i, 1));
        const Eigen::Vector2d p_r1(x + def_rows(i, 0), y + def_rows(i, 1));

        dic::PointObservation left;
        left.camera_index = 0;
        left.uv_ref = Eigen::Vector2d(x, y);
        left.uv_def = p_l1;
        left.u_displacement = p_l1.x() - x;
        left.v_displacement = p_l1.y() - y;
        left.correlation = std::min(corr_left_rows(i), corr_ref_rows(i));
        left.dic_valid = true;

        dic::PointObservation right;
        right.camera_index = 1;
        right.uv_ref = p_r0;
        right.uv_def = p_r1;
        right.u_displacement = p_r1.x() - p_r0.x();
        right.v_displacement = p_r1.y() - p_r0.y();
        right.correlation = std::min(corr_def_rows(i), corr_ref_rows(i));
        right.dic_valid = true;

        left_obs.push_back(left);
        right_obs.push_back(right);
    }

    py::gil_scoped_release release;
    return dic::StereoDIC(options).reconstruct(left_obs, right_obs, left_camera, right_camera);
}

py::dict stereo_result_arrays(const dic::StereoDICResult& result)
{
    const py::ssize_t n = static_cast<py::ssize_t>(result.points.size());
    py::array_t<double> point_ref_world({n, py::ssize_t{3}});
    py::array_t<double> point_def_world({n, py::ssize_t{3}});
    py::array_t<double> displacement_world({n, py::ssize_t{3}});
    py::array_t<double> displacement_norm_world({n});
    py::array_t<double> reprojection_error_ref({n});
    py::array_t<double> reprojection_error_def({n});
    py::array_t<double> combined_correlation({n});
    py::array_t<std::uint8_t> valid({n});

    auto ref = point_ref_world.mutable_unchecked<2>();
    auto def = point_def_world.mutable_unchecked<2>();
    auto disp = displacement_world.mutable_unchecked<2>();
    auto mag = displacement_norm_world.mutable_unchecked<1>();
    auto err_ref = reprojection_error_ref.mutable_unchecked<1>();
    auto err_def = reprojection_error_def.mutable_unchecked<1>();
    auto corr = combined_correlation.mutable_unchecked<1>();
    auto valid_rows = valid.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < n; ++i) {
        const auto& point = result.points[static_cast<std::size_t>(i)];
        for (py::ssize_t j = 0; j < 3; ++j) {
            ref(i, j) = point.point_ref_world(static_cast<Eigen::Index>(j));
            def(i, j) = point.point_def_world(static_cast<Eigen::Index>(j));
            disp(i, j) = point.displacement_world(static_cast<Eigen::Index>(j));
        }
        mag(i) = point.displacement_norm_world;
        err_ref(i) = point.reprojection_error_ref;
        err_def(i) = point.reprojection_error_def;
        corr(i) = point.combined_correlation;
        valid_rows(i) = point.valid ? 1 : 0;
    }

    py::dict arrays;
    arrays["point_ref_world"] = point_ref_world;
    arrays["point_def_world"] = point_def_world;
    arrays["displacement_world"] = displacement_world;
    arrays["displacement_norm_world"] = displacement_norm_world;
    arrays["reprojection_error_ref"] = reprojection_error_ref;
    arrays["reprojection_error_def"] = reprojection_error_def;
    arrays["combined_correlation"] = combined_correlation;
    arrays["valid"] = valid;
    return arrays;
}

} // namespace

void bind_reconstruction(py::module_& m)
{
    auto sub = m.def_submodule("reconstruction");

    py::class_<dic::PointObservation>(sub, "PointObservation")
        .def(py::init<>())
        .def_readwrite("camera_index", &dic::PointObservation::camera_index)
        .def_readwrite("uv_ref", &dic::PointObservation::uv_ref)
        .def_readwrite("uv_def", &dic::PointObservation::uv_def)
        .def_readwrite("u_displacement", &dic::PointObservation::u_displacement)
        .def_readwrite("v_displacement", &dic::PointObservation::v_displacement)
        .def_readwrite("correlation", &dic::PointObservation::correlation)
        .def_readwrite("dic_valid", &dic::PointObservation::dic_valid);

    py::class_<dic::ReconstructedPoint>(sub, "ReconstructedPoint")
        .def(py::init<>())
        .def_readwrite("track_id", &dic::ReconstructedPoint::track_id)
        .def_readwrite("point_ref", &dic::ReconstructedPoint::point_ref)
        .def_readwrite("point_def", &dic::ReconstructedPoint::point_def)
        .def_readwrite("point_ref_world", &dic::ReconstructedPoint::point_ref_world)
        .def_readwrite("point_def_world", &dic::ReconstructedPoint::point_def_world)
        .def_readwrite("displacement", &dic::ReconstructedPoint::displacement)
        .def_readwrite("displacement_world", &dic::ReconstructedPoint::displacement_world)
        .def_readwrite("displacement_norm_world", &dic::ReconstructedPoint::displacement_norm_world)
        .def_readwrite("num_views", &dic::ReconstructedPoint::num_views)
        .def_readwrite("mean_correlation", &dic::ReconstructedPoint::mean_correlation)
        .def_readwrite("reprojection_error_ref", &dic::ReconstructedPoint::reprojection_error_ref)
        .def_readwrite("reprojection_error_def", &dic::ReconstructedPoint::reprojection_error_def)
        .def_readwrite("valid", &dic::ReconstructedPoint::valid);

    py::class_<dic::ShapeReconstructionOptions>(sub, "ShapeReconstructionOptions")
        .def(py::init<>())
        .def_readwrite("min_views", &dic::ShapeReconstructionOptions::min_views)
        .def_readwrite("min_correlation", &dic::ShapeReconstructionOptions::min_correlation)
        .def_readwrite("max_reprojection_error_px", &dic::ShapeReconstructionOptions::max_reprojection_error_px)
        .def_readwrite("world_scale", &dic::ShapeReconstructionOptions::world_scale)
        .def_readwrite("remove_rigid_body_motion", &dic::ShapeReconstructionOptions::remove_rigid_body_motion);

    py::class_<dic::ShapeReconstructionResult>(sub, "ShapeReconstructionResult")
        .def(py::init<>())
        .def_readwrite("points", &dic::ShapeReconstructionResult::points)
        .def_readwrite("world_scale", &dic::ShapeReconstructionResult::world_scale)
        .def_readwrite("total_tracks", &dic::ShapeReconstructionResult::total_tracks)
        .def_readwrite("valid_tracks", &dic::ShapeReconstructionResult::valid_tracks);

    py::class_<dic::ShapeReconstruction>(sub, "ShapeReconstruction")
        .def(py::init<>())
        .def(py::init<const dic::ShapeReconstructionOptions&>())
        .def("reconstruct", &dic::ShapeReconstruction::reconstruct)
        .def("reconstruct_pair", &dic::ShapeReconstruction::reconstruct_pair)
        .def_static("build_projection", &dic::ShapeReconstruction::build_projection)
        .def_static("rigid_body_rotation", &dic::ShapeReconstruction::rigid_body_rotation);

    py::class_<dic::StereoPointResult>(sub, "StereoPointResult")
        .def(py::init<>())
        .def_readwrite("point_ref", &dic::StereoPointResult::point_ref)
        .def_readwrite("point_def", &dic::StereoPointResult::point_def)
        .def_readwrite("point_ref_world", &dic::StereoPointResult::point_ref_world)
        .def_readwrite("point_def_world", &dic::StereoPointResult::point_def_world)
        .def_readwrite("displacement", &dic::StereoPointResult::displacement)
        .def_readwrite("displacement_world", &dic::StereoPointResult::displacement_world)
        .def_readwrite("displacement_norm_world", &dic::StereoPointResult::displacement_norm_world)
        .def_readwrite("uv_ref_left", &dic::StereoPointResult::uv_ref_left)
        .def_readwrite("uv_ref_right", &dic::StereoPointResult::uv_ref_right)
        .def_readwrite("correlation_left", &dic::StereoPointResult::correlation_left)
        .def_readwrite("correlation_right", &dic::StereoPointResult::correlation_right)
        .def_readwrite("combined_correlation", &dic::StereoPointResult::combined_correlation)
        .def_readwrite("reprojection_error_ref", &dic::StereoPointResult::reprojection_error_ref)
        .def_readwrite("reprojection_error_def", &dic::StereoPointResult::reprojection_error_def)
        .def_readwrite("valid", &dic::StereoPointResult::valid);

    py::class_<dic::StereoDICResult>(sub, "StereoDICResult")
        .def(py::init<>())
        .def_readwrite("points", &dic::StereoDICResult::points)
        .def_readwrite("world_scale", &dic::StereoDICResult::world_scale)
        .def_readwrite("total_points", &dic::StereoDICResult::total_points)
        .def_readwrite("valid_points", &dic::StereoDICResult::valid_points)
        .def_readwrite("mean_displacement_norm", &dic::StereoDICResult::mean_displacement_norm);

    py::class_<dic::StereoDIC::Options>(sub, "StereoDICOptions")
        .def(py::init<>())
        .def_readwrite("min_correlation", &dic::StereoDIC::Options::min_correlation)
        .def_readwrite("max_reprojection_error_px", &dic::StereoDIC::Options::max_reprojection_error_px)
        .def_readwrite("world_scale", &dic::StereoDIC::Options::world_scale)
        .def_readwrite("remove_rigid_body_motion", &dic::StereoDIC::Options::remove_rigid_body_motion);

    py::class_<dic::StereoDIC>(sub, "StereoDIC")
        .def(py::init<>())
        .def(py::init<const dic::StereoDIC::Options&>())
        .def("reconstruct", &dic::StereoDIC::reconstruct);

    sub.def("reconstruct_stereo_fields",
            &reconstruct_stereo_fields_batch,
            py::arg("xy"),
            py::arg("reference_disparity"),
            py::arg("left_temporal"),
            py::arg("deformed_disparity"),
            py::arg("valid"),
            py::arg("corr_ref"),
            py::arg("corr_left_temporal"),
            py::arg("corr_deformed_disparity"),
            py::arg("left_camera"),
            py::arg("right_camera"),
            py::arg("options"));
    sub.def("stereo_result_arrays", &stereo_result_arrays, py::arg("result"));

    py::class_<dic::RigidBodyTransform>(sub, "RigidBodyTransform")
        .def(py::init<>())
        .def_readwrite("rotation", &dic::RigidBodyTransform::rotation)
        .def_readwrite("translation", &dic::RigidBodyTransform::translation)
        .def_readwrite("valid", &dic::RigidBodyTransform::valid);

    sub.def("compute_displacement", &dic::compute_displacement);
    sub.def("compute_displacements",
            [](const std::vector<Eigen::Vector3d>& reference,
               const std::vector<Eigen::Vector3d>& deformed) {
                std::vector<Eigen::Vector3d> out;
                dic::compute_displacements(reference, deformed, out);
                return out;
            });
    sub.def("compute_displacement_norms",
            [](const std::vector<Eigen::Vector3d>& displacements) {
                std::vector<double> out;
                dic::compute_displacement_norms(displacements, out);
                return out;
            });
    sub.def("find_rigid_body_transform", &dic::find_rigid_body_transform);
    sub.def("apply_rigid_body_transform", &dic::apply_rigid_body_transform);
}
