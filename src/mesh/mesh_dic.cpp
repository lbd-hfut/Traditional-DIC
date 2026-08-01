#include <dic/mesh/mesh_dic.hpp>
#include <dic/mesh/generation/roi_mesh_generator.hpp>
#include <dic/core/image.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/mesh/postprocess/strain.hpp>
#include <dic/mesh/initialization/fedic_fft_initializer.hpp>
#include <dic/subset/padding.hpp>

#include "coordinate/g2l_internal.hpp"
#include "solver/fem_assembler.hpp"
#include "generation/inform_builder.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef TRADITIONAL_DIC_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace dic {

using mesh::internal::G2LOutput;
using mesh::internal::G2LParams;
using mesh::internal::StiffnessCache;
using mesh::internal::nodes_per_element;
using mesh::internal::assemble_stiffness;
using mesh::internal::global_icgn;
using mesh::internal::compute_global_to_local;

static void mesh_to_flat(const Mesh& mesh,
                         std::vector<double>& nodes_coord,
                         std::vector<int>& elements,
                         mesh::MeshElementType& elem_type)
{
    const auto& nodes = mesh.nodes();
    const auto& elems = mesh.elements();
    nodes_coord.resize(2 * nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        nodes_coord[2 * i]     = nodes[i].coordinate.x();
        nodes_coord[2 * i + 1] = nodes[i].coordinate.y();
    }
    if (elems.empty()) { elements.clear(); elem_type = mesh::MeshElementType::Q4; return; }
    elem_type = elems[0].type;
    int nn = mesh::internal::nodes_per_element(elem_type);
    int stride = (elem_type == mesh::MeshElementType::Q8) ? 9 : nn;
    elements.resize(elems.size() * stride);
    for (size_t e = 0; e < elems.size(); ++e) {
        for (int k = 0; k < nn; ++k)
            elements[e * stride + k] = static_cast<int>(elems[e].node_ids[k]) + 1;
        if (elem_type == mesh::MeshElementType::Q8 && stride > nn)
            elements[e * stride + nn] = 0;
    }
}

static void fill_missing_nodal_initialization(
    const std::vector<double>& nodes_coord,
    const std::vector<unsigned char>& valid,
    Eigen::VectorXd& U)
{
    const int n_nodes = static_cast<int>(valid.size());
    int valid_count = 0;
    for (unsigned char flag : valid) {
        if (flag) {
            ++valid_count;
        }
    }
    if (valid_count == 0 || valid_count == n_nodes) {
        return;
    }

#ifdef TRADITIONAL_DIC_HAS_OPENCV
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_nodes; ++i) {
        if (!valid[static_cast<std::size_t>(i)]) continue;
        min_x = std::min(min_x, nodes_coord[2 * i]);
        max_x = std::max(max_x, nodes_coord[2 * i]);
        min_y = std::min(min_y, nodes_coord[2 * i + 1]);
        max_y = std::max(max_y, nodes_coord[2 * i + 1]);
    }
    const cv::Rect bounds(
        static_cast<int>(std::floor(min_x)) - 2,
        static_cast<int>(std::floor(min_y)) - 2,
        std::max(4, static_cast<int>(std::ceil(max_x - min_x)) + 5),
        std::max(4, static_cast<int>(std::ceil(max_y - min_y)) + 5));
    cv::Subdiv2D triangulation(bounds);
    for (int i = 0; i < n_nodes; ++i) {
        if (valid[static_cast<std::size_t>(i)]) {
            triangulation.insert(cv::Point2f(
                static_cast<float>(nodes_coord[2 * i]),
                static_cast<float>(nodes_coord[2 * i + 1])));
        }
    }
    std::vector<cv::Vec6f> triangles;
    triangulation.getTriangleList(triangles);
    const auto nearest_node = [&](double x, double y) {
        int best = -1;
        double best_distance2 = std::numeric_limits<double>::infinity();
        for (int j = 0; j < n_nodes; ++j) {
            if (!valid[static_cast<std::size_t>(j)]) continue;
            const double dx = x - nodes_coord[2 * j];
            const double dy = y - nodes_coord[2 * j + 1];
            const double distance2 = dx * dx + dy * dy;
            if (distance2 < best_distance2) {
                best_distance2 = distance2;
                best = j;
            }
        }
        // Subdiv2D stores vertices as float, while mesh nodes are double.
        return best_distance2 < 2.5e-3 ? best : -1;
    };
#endif

    for (int i = 0; i < n_nodes; ++i) {
        if (valid[i]) {
            continue;
        }

        const double x = nodes_coord[2 * i];
        const double y = nodes_coord[2 * i + 1];
#ifdef TRADITIONAL_DIC_HAS_OPENCV
        double best_score = -std::numeric_limits<double>::infinity();
        Eigen::Vector3d best_weights = Eigen::Vector3d::Zero();
        Eigen::Vector3i best_nodes = Eigen::Vector3i::Constant(-1);
        for (const auto& triangle : triangles) {
            const Eigen::Vector2d a(triangle[0], triangle[1]);
            const Eigen::Vector2d b(triangle[2], triangle[3]);
            const Eigen::Vector2d c(triangle[4], triangle[5]);
            const double twice_area = (b.x() - a.x()) * (c.y() - a.y()) -
                                      (b.y() - a.y()) * (c.x() - a.x());
            if (std::abs(twice_area) < 1.0e-12) continue;
            const Eigen::Vector2d p(x, y);
            Eigen::Vector3d weights;
            weights(0) = ((b.x() - p.x()) * (c.y() - p.y()) -
                          (b.y() - p.y()) * (c.x() - p.x())) / twice_area;
            weights(1) = ((c.x() - p.x()) * (a.y() - p.y()) -
                          (c.y() - p.y()) * (a.x() - p.x())) / twice_area;
            weights(2) = 1.0 - weights(0) - weights(1);
            const Eigen::Vector3i ids(
                nearest_node(a.x(), a.y()), nearest_node(b.x(), b.y()), nearest_node(c.x(), c.y()));
            if ((ids.array() < 0).any()) continue;
            const double score = weights.minCoeff();
            if (score > best_score) {
                best_score = score;
                best_weights = weights;
                best_nodes = ids;
            }
        }
        if (best_nodes(0) >= 0) {
            U(2 * i) = best_weights(0) * U(2 * best_nodes(0)) +
                       best_weights(1) * U(2 * best_nodes(1)) +
                       best_weights(2) * U(2 * best_nodes(2));
            U(2 * i + 1) = best_weights(0) * U(2 * best_nodes(0) + 1) +
                           best_weights(1) * U(2 * best_nodes(1) + 1) +
                           best_weights(2) * U(2 * best_nodes(2) + 1);
            continue;
        }
#endif
        int best = -1;
        double best_dist2 = std::numeric_limits<double>::infinity();
        for (int j = 0; j < n_nodes; ++j) {
            if (!valid[j]) {
                continue;
            }
            const double dx = x - nodes_coord[2 * j];
            const double dy = y - nodes_coord[2 * j + 1];
            const double dist2 = dx * dx + dy * dy;
            if (dist2 < best_dist2) {
                best_dist2 = dist2;
                best = j;
            }
        }

        if (best >= 0) {
            U(2 * i) = U(2 * best);
            U(2 * i + 1) = U(2 * best + 1);
        }
    }
}

static double median_value(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
    std::nth_element(values.begin(), mid, values.end());
    double med = *mid;
    if (values.size() % 2U == 0U) {
        const auto lo = std::max_element(values.begin(), mid);
        med = 0.5 * (med + *lo);
    }
    return med;
}

static std::vector<int> nearest_valid_nodes(
    const std::vector<double>& nodes_coord,
    const std::vector<unsigned char>& valid,
    int node_id,
    int max_neighbors)
{
    std::vector<std::pair<double, int>> distances;
    const int n_nodes = static_cast<int>(valid.size());
    distances.reserve(static_cast<std::size_t>(n_nodes));
    const double x = nodes_coord[2 * node_id];
    const double y = nodes_coord[2 * node_id + 1];
    for (int j = 0; j < n_nodes; ++j) {
        if (j == node_id || !valid[static_cast<std::size_t>(j)]) {
            continue;
        }
        const double dx = x - nodes_coord[2 * j];
        const double dy = y - nodes_coord[2 * j + 1];
        distances.push_back({dx * dx + dy * dy, j});
    }
    const int keep = std::min(std::max(1, max_neighbors), static_cast<int>(distances.size()));
    if (keep < static_cast<int>(distances.size())) {
        std::nth_element(distances.begin(), distances.begin() + keep, distances.end());
    }
    std::sort(distances.begin(), distances.begin() + keep);
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(keep));
    for (int i = 0; i < keep; ++i) {
        out.push_back(distances[static_cast<std::size_t>(i)].second);
    }
    return out;
}

static void repair_invalid_nodes_by_neighbors(
    const std::vector<double>& nodes_coord,
    std::vector<unsigned char>& valid,
    Eigen::VectorXd& U,
    int interpolation_neighbors)
{
    const int n_nodes = static_cast<int>(valid.size());
    std::vector<unsigned char> repaired = valid;
    for (int i = 0; i < n_nodes; ++i) {
        if (valid[static_cast<std::size_t>(i)]) {
            continue;
        }
        const auto neighbors = nearest_valid_nodes(nodes_coord, valid, i, interpolation_neighbors);
        if (neighbors.empty()) {
            continue;
        }
        double sum_w = 0.0;
        double u = 0.0;
        double v = 0.0;
        const double x = nodes_coord[2 * i];
        const double y = nodes_coord[2 * i + 1];
        for (int j : neighbors) {
            const double dx = x - nodes_coord[2 * j];
            const double dy = y - nodes_coord[2 * j + 1];
            const double w = 1.0 / std::max(std::sqrt(dx * dx + dy * dy), 1.0e-6);
            sum_w += w;
            u += w * U(2 * j);
            v += w * U(2 * j + 1);
        }
        if (sum_w > 0.0) {
            U(2 * i) = u / sum_w;
            U(2 * i + 1) = v / sum_w;
            repaired[static_cast<std::size_t>(i)] = 1;
        }
    }
    valid.swap(repaired);
}

static void apply_mesh_initialization_quality_control(
    const MeshConfig& config,
    const std::vector<double>& nodes_coord,
    const std::vector<double>& init_zncc,
    const std::vector<double>& init_znssd,
    const std::vector<double>& init_pce,
    const std::vector<double>& init_ppe,
    std::vector<unsigned char>& valid,
    Eigen::VectorXd& U)
{
    const auto& qc = config.initialization_quality;
    if (!qc.enabled) {
        return;
    }
    const int n_nodes = static_cast<int>(valid.size());
    for (int i = 0; i < n_nodes; ++i) {
        if (!valid[static_cast<std::size_t>(i)]) {
            continue;
        }
        const double zncc = init_zncc[static_cast<std::size_t>(i)];
        const double znssd = init_znssd[static_cast<std::size_t>(i)];
        if ((std::isfinite(zncc) && std::isfinite(qc.min_zncc) && zncc < qc.min_zncc) ||
            (std::isfinite(znssd) && std::isfinite(qc.max_znssd) && znssd > qc.max_znssd)) {
            valid[static_cast<std::size_t>(i)] = 0;
        }
    }

    if (qc.fedic_qfactor_enabled) {
        const auto filter_qfactor = [&](const std::vector<double>& values) {
            double sum = 0.0;
            double sum_sq = 0.0;
            int count = 0;
            for (int i = 0; i < n_nodes; ++i) {
                const double value = values[static_cast<std::size_t>(i)];
                if (valid[static_cast<std::size_t>(i)] && std::isfinite(value)) {
                    sum += value;
                    sum_sq += value * value;
                    ++count;
                }
            }
            if (count < 4) return;
            const double mean = sum / static_cast<double>(count);
            const double variance = std::max(0.0, sum_sq / static_cast<double>(count) - mean * mean);
            const double threshold = mean - qc.fedic_qfactor_std_factor * std::sqrt(variance);
            for (int i = 0; i < n_nodes; ++i) {
                const double value = values[static_cast<std::size_t>(i)];
                if (valid[static_cast<std::size_t>(i)] && (!std::isfinite(value) || value < threshold)) {
                    valid[static_cast<std::size_t>(i)] = 0;
                }
            }
        };
        filter_qfactor(init_pce);
        filter_qfactor(init_ppe);
    }

    std::vector<unsigned char> keep = valid;
    for (int i = 0; i < n_nodes; ++i) {
        if (!valid[static_cast<std::size_t>(i)]) {
            continue;
        }
        const auto neighbors = nearest_valid_nodes(
            nodes_coord, valid, i, qc.interpolation_neighbors);
        if (neighbors.size() < 3U) {
            continue;
        }
        std::vector<double> us;
        std::vector<double> vs;
        us.reserve(neighbors.size());
        vs.reserve(neighbors.size());
        for (int j : neighbors) {
            us.push_back(U(2 * j));
            vs.push_back(U(2 * j + 1));
        }
        const double med_u = median_value(us);
        const double med_v = median_value(vs);
        std::vector<double> residuals;
        residuals.reserve(neighbors.size());
        for (int j : neighbors) {
            residuals.push_back(std::hypot(U(2 * j) - med_u, U(2 * j + 1) - med_v));
        }
        const double med_r = median_value(residuals);
        for (double& r : residuals) {
            r = std::abs(r - med_r);
        }
        const double mad = median_value(residuals);
        const double robust_sigma = std::max(1.4826 * mad, 1.0e-6);
        const double deviation = std::hypot(U(2 * i) - med_u, U(2 * i + 1) - med_v);
        double threshold = med_r + qc.neighbor_mad_factor * robust_sigma;
        if (qc.max_neighbor_deviation > 0.0) {
            threshold = std::min(threshold, qc.max_neighbor_deviation);
        }
        if (deviation > threshold) {
            keep[static_cast<std::size_t>(i)] = 0;
        }
    }
    valid.swap(keep);
    repair_invalid_nodes_by_neighbors(nodes_coord, valid, U, qc.interpolation_neighbors);
}

static int recommended_mesh_padding(const MeshConfig& config)
{
    const int init_radius = std::max(1, (config.fedic_fft_initialization.window_size + 1) / 2);
    const int bspline_border = std::max(0, config.image_precompute.border);
    const int search_radius = std::max(config.search_radius,
                                       config.fedic_fft_initialization.search_radius);
    return std::max(0, search_radius) + std::max(0, init_radius) + bspline_border;
}

static void solve_global_mesh_displacement(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img,
    int img_h,
    int img_w,
    const int* elements,
    int n_elements,
    Eigen::VectorXd& U,
    const BSplineInterpolator* def_interp,
    double alpha,
    double tol,
    int max_it)
{
    global_icgn(cache, g2l, ref_img, img_h, img_w,
                elements, n_elements, U, def_interp, alpha, tol, max_it,
                0.0);
}

MeshDIC::MeshDIC(MeshConfig config) : config_(config) {}

std::vector<Displacement2D> MeshDIC::compute(
    const Image& reference, const Image& deformed, const Mesh& mesh) const
{
    if (reference.empty() || deformed.empty()) {
        return {};
    }
    if (reference.width() != deformed.width() || reference.height() != deformed.height()) {
        throw std::invalid_argument("MeshDIC requires reference and deformed images with matching dimensions.");
    }

    const int pad = config_.mirror_image_padding ? recommended_mesh_padding(config_) : 0;
    const Image solver_reference = pad > 0 ? mirror_pad_image(reference, pad) : reference;
    const Image solver_deformed = pad > 0 ? mirror_pad_image(deformed, pad) : deformed;
    int img_h = solver_reference.height(), img_w = solver_reference.width();

    // ---- 1. Convert mesh to flat arrays ----
    std::vector<double> nodes_coord;
    std::vector<int> elements_flat;
    mesh::MeshElementType elem_type;
    mesh_to_flat(mesh, nodes_coord, elements_flat, elem_type);
    int n_nodes = static_cast<int>(nodes_coord.size() / 2);
    std::vector<double> solver_nodes_coord = nodes_coord;
    if (pad > 0) {
        for (int i = 0; i < n_nodes; ++i) {
            solver_nodes_coord[2 * i] += static_cast<double>(pad);
            solver_nodes_coord[2 * i + 1] += static_cast<double>(pad);
        }
    }

    // ---- 2. B-spline precompute through the shared interpolation module ----
    BSplineImagePreprocessor preproc(config_.image_precompute);
    BSplinePrecomputedImage ref_precomp = preproc.compute(solver_reference);
    BSplinePrecomputedImage def_precomp = preproc.compute(solver_deformed);
    Eigen::MatrixXd& grad_x = ref_precomp.gradient_x;
    Eigen::MatrixXd& grad_y = ref_precomp.gradient_y;
    if (grad_x.size() == 0 || grad_y.size() == 0) {
        grad_x = Eigen::MatrixXd::Zero(img_h, img_w);
        grad_y = Eigen::MatrixXd::Zero(img_h, img_w);
        for (int y = 0; y < img_h; ++y)
            for (int x = 0; x < img_w; ++x) {
                if (x > 0 && x < img_w - 1)
                    grad_x(y, x) = (solver_reference.at(x+1, y) - solver_reference.at(x-1, y)) * 0.5;
                if (y > 0 && y < img_h - 1)
                    grad_y(y, x) = (solver_reference.at(x, y+1) - solver_reference.at(x, y-1)) * 0.5;
            }
    }

    // ---- 3. Interpolators backed by the same precompute configuration ----
    BSplineInterpolator ref_interp(&ref_precomp);
    BSplineInterpolator def_interp(&def_precomp);

    // ---- 3b. Extract flat image data ----
    std::vector<double> ref_flat(img_h * img_w), def_flat(img_h * img_w);
    std::vector<double> fx_flat(img_h * img_w), fy_flat(img_h * img_w);
    for (int y = 0; y < img_h; ++y)
        for (int x = 0; x < img_w; ++x) {
            int idx = y * img_w + x;
            ref_flat[idx] = static_cast<double>(solver_reference.at(x, y));
            def_flat[idx] = static_cast<double>(solver_deformed.at(x, y));
            fx_flat[idx] = grad_x(y, x);
            fy_flat[idx] = grad_y(y, x);
        }

    // ---- 4-5. Build inform + G2L ----
    int n_elements = static_cast<int>(elements_flat.size()) /
        ((elem_type == mesh::MeshElementType::Q8) ? 9 : nodes_per_element(elem_type));
    const bool fedic_compatible_numerics = elem_type != mesh::MeshElementType::Q8;
    auto inform = fedic_compatible_numerics
        ? mesh::build_fedic_inform(solver_nodes_coord.data(), n_nodes,
            elements_flat.data(), n_elements, elem_type, img_h, img_w)
        : mesh::build_inform(solver_nodes_coord.data(), n_nodes,
            elements_flat.data(), n_elements, elem_type, img_h, img_w);
    int n_pixels = static_cast<int>(inform.size() / 3);

    G2LParams g2l_params; g2l_params.max_iter = 200;
    auto g2l = compute_global_to_local(inform.data(), n_pixels,
        solver_nodes_coord.data(), n_nodes, elements_flat.data(), n_elements,
        img_h, img_w, elem_type, g2l_params);

    // ---- 6. Assemble stiffness ----
    double alpha = config_.regularization_alpha;
    auto cache = assemble_stiffness(g2l, img_h, img_w,
        fx_flat.data(), fy_flat.data(), n_nodes, elements_flat.data(), n_elements, elem_type,
        alpha, 0.0, fedic_compatible_numerics, true);

    // ---- 7. Displacement init ----
    Eigen::VectorXd U = Eigen::VectorXd::Zero(2 * n_nodes);
    std::vector<unsigned char> init_valid(static_cast<std::size_t>(n_nodes), 0);
    std::vector<double> init_zncc(static_cast<std::size_t>(n_nodes), -std::numeric_limits<double>::infinity());
    std::vector<double> init_znssd(static_cast<std::size_t>(n_nodes), std::numeric_limits<double>::infinity());
    std::vector<double> init_pce(static_cast<std::size_t>(n_nodes), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> init_ppe(static_cast<std::size_t>(n_nodes), std::numeric_limits<double>::quiet_NaN());
    for (int i = 0; i < n_nodes; ++i) {
        Eigen::Vector2d pt(nodes_coord[2 * i], nodes_coord[2 * i + 1]);
        if (pt.x() < 0 || pt.x() >= reference.width() || pt.y() < 0 || pt.y() >= reference.height()) continue;
        const auto fedic_initial = mesh::estimate_fedic_fft_initial_displacement(
            reference, deformed, pt,
            config_.fedic_fft_initialization.search_radius,
            config_.fedic_fft_initialization.window_size);
        auto selected_initial = fedic_initial;
        if (!selected_initial.initial.valid &&
            config_.fedic_fft_initialization.mirror_boundary_fallback && pad > 0) {
            const Eigen::Vector2d padded_point(
                solver_nodes_coord[2 * i], solver_nodes_coord[2 * i + 1]);
            selected_initial = mesh::estimate_fedic_fft_initial_displacement(
                solver_reference, solver_deformed, padded_point,
                config_.fedic_fft_initialization.search_radius,
                config_.fedic_fft_initialization.window_size);
        }
        const InitialDisplacement& init = selected_initial.initial;
        if (init.valid) {
            U(2 * i) = init.u;
            U(2 * i + 1) = init.v;
            init_valid[static_cast<std::size_t>(i)] = 1;
            init_zncc[static_cast<std::size_t>(i)] = init.zncc;
            init_znssd[static_cast<std::size_t>(i)] = init.znssd;
            init_pce[static_cast<std::size_t>(i)] = selected_initial.peak_to_correlation_energy;
            init_ppe[static_cast<std::size_t>(i)] = selected_initial.peak_to_entropy;
        }
    }
    apply_mesh_initialization_quality_control(
        config_, nodes_coord, init_zncc, init_znssd, init_pce, init_ppe, init_valid, U);
    fill_missing_nodal_initialization(nodes_coord, init_valid, U);

    // ---- 8. Global solver ----
    double tol = config_.convergence_threshold;
    int max_it = config_.max_iterations;
    solve_global_mesh_displacement(cache, g2l, ref_flat.data(), img_h, img_w,
        elements_flat.data(), n_elements, U, &def_interp, alpha, tol, max_it);

    // ---- 9-10. Write back + convert ----
    std::vector<Displacement2D> results; results.reserve(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        Displacement2D d; d.x = nodes_coord[2*i]; d.y = nodes_coord[2*i+1];
        d.u = U(2*i); d.v = U(2*i+1); d.valid = true;
        results.push_back(d);
    }
    return results;
}

std::vector<Displacement2D> MeshDIC::compute(
    const Image& reference, const Image& deformed, const ROI& roi,
    const mesh::MeshGenerationConfig& mgc) const
{
    mesh::ROIMeshGenerator gen;
    return compute(reference, deformed, gen.generate(roi, mgc));
}

} // namespace dic
