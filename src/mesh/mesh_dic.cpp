#include <dic/mesh/mesh_dic.hpp>
#include <dic/mesh/generation/roi_mesh_generator.hpp>
#include <dic/core/image.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/mesh/postprocess/strain.hpp>
#include <dic/initialization/integer_search.hpp>
#include <dic/initialization/feature_matcher.hpp>
#include <dic/initialization/seed_config.hpp>
#include <dic/initialization/sift_initializer.hpp>
#include <dic/subset/padding.hpp>

#include "coordinate/g2l_internal.hpp"
#include "solver/fem_assembler.hpp"
#include "generation/inform_builder.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dic {

using mesh::internal::G2LOutput;
using mesh::internal::G2LParams;
using mesh::internal::StiffnessCache;
using mesh::internal::nodes_per_element;
using mesh::internal::assemble_stiffness;
using mesh::internal::assemble_residual;
using mesh::internal::compute_objective;
using mesh::internal::global_icgn;
using mesh::internal::global_forward_gn;
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

    for (int i = 0; i < n_nodes; ++i) {
        if (valid[i]) {
            continue;
        }

        const double x = nodes_coord[2 * i];
        const double y = nodes_coord[2 * i + 1];
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

static int initialize_nodes_with_sift_route(
    const Image& reference,
    const Image& deformed,
    const std::vector<double>& nodes_coord,
    const MeshConfig::SIFTNodeInitializationConfig& config,
    Eigen::VectorXd& U,
    std::vector<unsigned char>& valid)
{
    FeatureMatcherConfig matcher_config;
    matcher_config.max_features = config.max_features;
    matcher_config.ratio_threshold = config.ratio_threshold;
    matcher_config.robust_mad_factor = config.robust_mad_factor;
    const FeatureMatcher matcher(matcher_config);
    const auto matches = matcher.match(reference, deformed);
    if (matches.empty()) {
        return 0;
    }

    SIFTInitializerConfig initializer_config;
    initializer_config.matcher = matcher_config;
    initializer_config.interpolation_neighbors = config.interpolation_neighbors;
    initializer_config.interpolation_radius = config.interpolation_radius;
    const SIFTInitializer initializer(initializer_config);

    const int n_nodes = static_cast<int>(nodes_coord.size() / 2U);
    int initialized = 0;
    for (int i = 0; i < n_nodes; ++i) {
        const Eigen::Vector2d point(nodes_coord[2 * i], nodes_coord[2 * i + 1]);
        const auto initial = initializer.estimate_from_matches(matches, point);
        if (!initial.valid) {
            continue;
        }
        U(2 * i) = initial.u;
        U(2 * i + 1) = initial.v;
        valid[static_cast<std::size_t>(i)] = 1;
        ++initialized;
    }
    return initialized;
}

static int recommended_mesh_padding(const MeshConfig& config)
{
    const int init_radius = std::max(config.seed_initialization.integer_search.subset_radius,
                                     config.seed_initialization.subpixel.subset_radius);
    const int bspline_border = std::max(0, config.image_precompute.border);
    const int search_radius = std::max(config.search_radius,
                                       config.seed_initialization.integer_search.search_radius);
    return std::max(0, search_radius) + std::max(0, init_radius) + bspline_border;
}

static void global_icgn_znssd_placeholder(Eigen::VectorXd& U)
{
    // ZNSSD global mesh optimization needs a result/status API before it can
    // report an explicit not-implemented state; keep U at initialization.
    (void)U;
}

static void global_forward_gn_znssd_placeholder(Eigen::VectorXd& U)
{
    // ZNSSD global mesh optimization needs a result/status API before it can
    // report an explicit not-implemented state; keep U at initialization.
    (void)U;
}

static void solve_global_mesh_displacement(
    const MeshConfig& config,
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
    if (config.solver_method == MeshSolverMethod::ForwardGaussNewton) {
        if (config.objective == CorrelationCriterionKind::ZNSSD) {
            global_forward_gn_znssd_placeholder(U);
            return;
        }
        global_forward_gn(cache, g2l, ref_img, img_h, img_w,
                          elements, n_elements, U, def_interp, alpha, tol, max_it);
        return;
    }

    if (config.objective == CorrelationCriterionKind::ZNSSD) {
        global_icgn_znssd_placeholder(U);
        return;
    }
    global_icgn(cache, g2l, ref_img, img_h, img_w,
                elements, n_elements, U, def_interp, alpha, tol, max_it);
}

MeshDIC::MeshDIC(MeshConfig config) : config_(config) {}

std::vector<Displacement2D> MeshDIC::compute(
    const Image& reference, const Image& deformed, Mesh mesh) const
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
    auto inform = mesh::build_inform(solver_nodes_coord.data(), n_nodes,
        elements_flat.data(), n_elements, elem_type, img_h, img_w);
    int n_pixels = static_cast<int>(inform.size() / 3);

    G2LParams g2l_params; g2l_params.max_iter = 200;
    auto g2l = compute_global_to_local(inform.data(), n_pixels,
        solver_nodes_coord.data(), n_nodes, elements_flat.data(), n_elements,
        img_h, img_w, elem_type, g2l_params);

    // ---- 6. Assemble stiffness ----
    double alpha = config_.regularization_alpha;
    auto cache = assemble_stiffness(g2l, img_h, img_w,
        fx_flat.data(), fy_flat.data(), n_nodes, elements_flat.data(), n_elements, elem_type, alpha);

    // ---- 7. Displacement init ----
    Eigen::VectorXd U = Eigen::VectorXd::Zero(2 * n_nodes);
    SeedInitializationConfig seed_cfg = config_.seed_initialization;
    if (seed_cfg.integer_search.search_radius <= 0) {
        seed_cfg.integer_search.search_radius = config_.search_radius;
    }
    if (seed_cfg.integer_search.sift_enabled) {
        seed_cfg.integer_search.pyramid_enabled = false;
    }
    std::vector<unsigned char> init_valid(static_cast<std::size_t>(n_nodes), 0);
    if (seed_cfg.method == SeedInitializationMethod::SIFT) {
        initialize_nodes_with_sift_route(
            reference,
            deformed,
            nodes_coord,
            config_.sift_node_initialization,
            U,
            init_valid);
    }

    std::vector<unsigned char> prior_valid(static_cast<std::size_t>(n_nodes), 0);
    Eigen::VectorXd prior_U = Eigen::VectorXd::Zero(2 * n_nodes);
    if (seed_cfg.method != SeedInitializationMethod::SIFT &&
        seed_cfg.integer_search.sift_enabled) {
        FeatureMatcherConfig matcher_config;
        matcher_config.max_features = config_.sift_node_initialization.max_features;
        matcher_config.ratio_threshold = config_.sift_node_initialization.ratio_threshold;
        matcher_config.robust_mad_factor = config_.sift_node_initialization.robust_mad_factor;
        const FeatureMatcher matcher(matcher_config);
        const auto matches = matcher.match(reference, deformed);

        SIFTInitializerConfig initializer_config;
        initializer_config.matcher = matcher_config;
        initializer_config.interpolation_neighbors = config_.sift_node_initialization.interpolation_neighbors;
        initializer_config.interpolation_radius = config_.sift_node_initialization.interpolation_radius;
        const SIFTInitializer initializer(initializer_config);
        for (int i = 0; i < n_nodes; ++i) {
            const Eigen::Vector2d point(nodes_coord[2 * i], nodes_coord[2 * i + 1]);
            const auto prior = initializer.estimate_from_matches(matches, point);
            if (!prior.valid) {
                continue;
            }
            prior_U(2 * i) = prior.u;
            prior_U(2 * i + 1) = prior.v;
            prior_valid[static_cast<std::size_t>(i)] = 1;
        }
    }

    IntegerSearchInitializer int_search(seed_cfg, config_.image_precompute);
    for (int i = 0; i < n_nodes; ++i) {
        if (init_valid[static_cast<std::size_t>(i)]) {
            continue;
        }
        Eigen::Vector2d pt(solver_nodes_coord[2 * i], solver_nodes_coord[2 * i + 1]);
        if (pt.x() < 0 || pt.x() >= img_w || pt.y() < 0 || pt.y() >= img_h) continue;
        InitialDisplacement init;
        if (prior_valid[static_cast<std::size_t>(i)]) {
            init = int_search.estimate_around_displacement(
                solver_reference,
                solver_deformed,
                pt,
                prior_U(2 * i),
                prior_U(2 * i + 1));
        } else {
            init = int_search.estimate_with_interpolators(
                solver_reference, solver_deformed, pt, ref_interp, def_interp);
        }
        if (init.valid) {
            U(2 * i) = init.u;
            U(2 * i + 1) = init.v;
            init_valid[static_cast<std::size_t>(i)] = 1;
        }
    }
    fill_missing_nodal_initialization(nodes_coord, init_valid, U);

    // ---- 8. Global solver ----
    double tol = config_.convergence_threshold;
    int max_it = config_.max_iterations;
    solve_global_mesh_displacement(config_, cache, g2l, ref_flat.data(), img_h, img_w,
        elements_flat.data(), n_elements, U, &def_interp, alpha, tol, max_it);

    // ---- 9-10. Write back + convert ----
    auto& mesh_nodes = mesh.nodes();
    for (int i = 0; i < n_nodes && i < static_cast<int>(mesh_nodes.size()); ++i) {
        mesh_nodes[i].displacement.x() = U(2 * i);
        mesh_nodes[i].displacement.y() = U(2 * i + 1);
    }
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
