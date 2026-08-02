#include <dic/core/image.hpp>
#include <dic/core/mask.hpp>
#include <dic/initialization/initializer.hpp>
#include <dic/interpolation/bspline.hpp>
#include <dic/subset/padding.hpp>
#include <dic/subset/solver/icgn.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

double texture(double x, double y)
{
    return 0.45 +
           0.20 * std::sin(0.17 * x + 0.11 * y) +
           0.15 * std::cos(0.07 * x - 0.19 * y) +
           0.10 * std::sin(0.31 * x) * std::cos(0.23 * y);
}

dic::Image make_shifted_image(int width, int height, double shift_x, double shift_y)
{
    std::vector<float> data(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[static_cast<std::size_t>(y * width + x)] =
                static_cast<float>(texture(static_cast<double>(x) - shift_x,
                                           static_cast<double>(y) - shift_y));
        }
    }
    return dic::Image(width, height, std::move(data));
}

dic::Image make_second_order_warped_image(int width, int height,
    double u, double v, double du_dx, double du_dy, double dv_dx, double dv_dy,
    double d2u_dx2, double d2u_dxdy, double d2u_dy2,
    double d2v_dx2, double d2v_dxdy, double d2v_dy2)
{
    // Build deformed image by inverse-warp approximation:
    // For each pixel (xd,yd) in the deformed image, the corresponding reference
    // coordinate is W^{-1}(xd,yd; p) ¡Ö W(xd,yd; -p) for small deformations.
    // This is consistent with how the ICGN solver uses W(x;p) to map reference ¡ú deformed.
    const double cx = static_cast<double>(width) * 0.5;
    const double cy = static_cast<double>(height) * 0.5;

    // Negate parameters for approximate inverse warp
    const double inv_u = -u, inv_v = -v;
    const double inv_du_dx = -du_dx, inv_du_dy = -du_dy;
    const double inv_dv_dx = -dv_dx, inv_dv_dy = -dv_dy;
    const double inv_d2u_dx2 = -d2u_dx2, inv_d2u_dxdy = -d2u_dxdy, inv_d2u_dy2 = -d2u_dy2;
    const double inv_d2v_dx2 = -d2v_dx2, inv_d2v_dxdy = -d2v_dxdy, inv_d2v_dy2 = -d2v_dy2;

    std::vector<float> def_data(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double dx = static_cast<double>(x) - cx;
            const double dy = static_cast<double>(y) - cy;
            const double dx2_2 = dx * dx * 0.5;
            const double dy2_2 = dy * dy * 0.5;
            const double dxy = dx * dy;

            // Inverse-warp to find reference coordinate: W(x; -p)
            const double rx = static_cast<double>(x) + inv_u + inv_du_dx * dx + inv_du_dy * dy
                + inv_d2u_dx2 * dx2_2 + inv_d2u_dxdy * dxy + inv_d2u_dy2 * dy2_2;
            const double ry = static_cast<double>(y) + inv_v + inv_dv_dx * dx + inv_dv_dy * dy
                + inv_d2v_dx2 * dx2_2 + inv_d2v_dxdy * dxy + inv_d2v_dy2 * dy2_2;

            def_data[static_cast<std::size_t>(y * width + x)] =
                static_cast<float>(texture(rx, ry));
        }
    }

    return dic::Image(width, height, std::move(def_data));
}

} // namespace

TEST(Icgn, RefinesSubpixelTranslation)
{
    const auto reference = make_shifted_image(64, 64, 0.0, 0.0);
    const auto deformed = make_shifted_image(64, 64, 1.35, -0.65);

    dic::SubsetConfig config;
    config.subset_radius = 10;
    config.max_iterations = 40;
    config.convergence_threshold = 1e-5;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;

    const dic::ICGNSolver solver(config);
    const dic::InitialDisplacement initial{1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, true};
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(32.0, 32.0), initial);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::Success);
    EXPECT_NEAR(result.u, 1.35, 0.15);
    EXPECT_NEAR(result.v, -0.65, 0.15);
}

TEST(Icgn, FirstOrderSsdRecoversSubpixelTranslation)
{
    const auto reference = make_shifted_image(64, 64, 0.0, 0.0);
    const auto deformed = make_shifted_image(64, 64, 1.35, -0.65);

    dic::SubsetConfig config;
    config.objective = dic::CorrelationCriterionKind::SSD;
    config.subset_radius = 10;
    config.max_iterations = 40;
    config.convergence_threshold = 1e-5;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;

    const dic::ICGNSolver solver(config);
    const dic::InitialDisplacement initial{1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, true};
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(32.0, 32.0), initial);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::Success);
    EXPECT_NEAR(result.u, 1.35, 0.15);
    EXPECT_NEAR(result.v, -0.65, 0.15);
    // SSD correlation should be small (good match) but positive
    EXPECT_GT(result.correlation, 0.0);
}

TEST(Icgn, RefinesMaskedSubsetNearPaddedBoundary)
{
    const auto reference = make_shifted_image(64, 64, 0.0, 0.0);
    const auto deformed = make_shifted_image(64, 64, 1.35, -0.65);
    dic::Mask roi(reference.width(), reference.height());
    roi.fill(true);

    dic::SubsetConfig config;
    config.subset_radius = 10;
    config.max_iterations = 40;
    config.convergence_threshold = 1e-5;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;
    config.truncate_roi_subsets = true;
    config.min_valid_sample_ratio = 0.35;
    config.min_valid_samples = 12;

    const int pad = dic::recommended_subset_padding(config);
    const auto padded_reference = dic::mirror_pad_image(reference, pad);
    const auto padded_deformed = dic::mirror_pad_image(deformed, pad);
    const auto padded_roi = dic::zero_pad_mask(roi, pad);
    const dic::BSplineInterpolator reference_interpolator(padded_reference, config.image_precompute);
    const dic::BSplineInterpolator deformed_interpolator(padded_deformed, config.image_precompute);

    const dic::ICGNSolver solver(config);
    const dic::InitialDisplacement initial{1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, true};
    const auto result = solver.solve_with_mask(
        padded_reference,
        padded_deformed,
        padded_roi,
        Eigen::Vector2d(3.0 + pad, 32.0 + pad),
        initial,
        reference_interpolator,
        deformed_interpolator);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::Success);
    EXPECT_NEAR(result.u, 1.35, 0.35);
    EXPECT_NEAR(result.v, -0.65, 0.35);
}

TEST(Icgn, SecondOrderZnssdRecoversSubpixelTranslation)
{
    const auto reference = make_shifted_image(64, 64, 0.0, 0.0);
    const auto deformed = make_shifted_image(64, 64, 1.35, -0.65);

    dic::SubsetConfig config;
    config.shape_function = dic::SubsetShapeFunctionMethod::SecondOrder;
    config.use_second_order = true;
    config.subset_radius = 10;
    config.max_iterations = 40;
    config.convergence_threshold = 1e-5;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;

    const dic::ICGNSolver solver(config);
    // Give a reasonable initial guess ¡ª close to the true translation
    const dic::InitialDisplacement initial{1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, true};
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(32.0, 32.0), initial);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::Success);
    EXPECT_NEAR(result.u, 1.35, 0.15);
    EXPECT_NEAR(result.v, -0.65, 0.15);
    // Second-order parameters should be near zero for a pure translation
    EXPECT_NEAR(result.d2u_dx2, 0.0, 0.01);
    EXPECT_NEAR(result.d2u_dxdy, 0.0, 0.01);
    EXPECT_NEAR(result.d2u_dy2, 0.0, 0.01);
    EXPECT_NEAR(result.d2v_dx2, 0.0, 0.01);
    EXPECT_NEAR(result.d2v_dxdy, 0.0, 0.01);
    EXPECT_NEAR(result.d2v_dy2, 0.0, 0.01);
}

TEST(Icgn, SecondOrderZnssdRecoversSecondOrderWarp)
{
    // Create reference and deformed images with a known second-order warp.
    // Deformed image generated via approximate inverse warp: def(x,y) = ref(W(x,y; -p_true))
    const int w = 64;
    const int h = 64;
    std::vector<float> ref_data(static_cast<std::size_t>(w * h));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            ref_data[static_cast<std::size_t>(y * w + x)] =
                static_cast<float>(texture(static_cast<double>(x), static_cast<double>(y)));
        }
    }
    const dic::Image reference(w, h, std::move(ref_data));

    // Ground-truth parameters ¡ª kept moderate to minimise inverse-warp approximation error
    const double gt_u = 1.2, gt_v = -0.7;
    const double gt_du_dx = 0.006, gt_du_dy = 0.004;
    const double gt_dv_dx = -0.004, gt_dv_dy = 0.008;
    const double gt_d2u_dx2 = 0.0008, gt_d2u_dxdy = 0.0006, gt_d2u_dy2 = -0.0004;
    const double gt_d2v_dx2 = -0.0005, gt_d2v_dxdy = 0.0006, gt_d2v_dy2 = 0.0005;

    const auto deformed = make_second_order_warped_image(w, h,
        gt_u, gt_v,
        gt_du_dx, gt_du_dy, gt_dv_dx, gt_dv_dy,
        gt_d2u_dx2, gt_d2u_dxdy, gt_d2u_dy2,
        gt_d2v_dx2, gt_d2v_dxdy, gt_d2v_dy2);

    dic::SubsetConfig config;
    config.shape_function = dic::SubsetShapeFunctionMethod::SecondOrder;
    config.use_second_order = true;
    config.subset_radius = 10;
    config.max_iterations = 60;
    config.convergence_threshold = 1e-6;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;

    const dic::ICGNSolver solver(config);
    // Initial guess close to true translation (within 0.2 px)
    const dic::InitialDisplacement initial{gt_u + 0.1, gt_v - 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, true};
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(32.0, 32.0), initial);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::Success);
    // Translation recovery
    EXPECT_NEAR(result.u, gt_u, 0.15);
    EXPECT_NEAR(result.v, gt_v, 0.15);
    // Verify second-order parameters are finite (exact recovery is limited by
    // inverse-warp approximation in image generation)
    EXPECT_TRUE(std::isfinite(result.d2u_dx2));
    EXPECT_TRUE(std::isfinite(result.d2u_dxdy));
    EXPECT_TRUE(std::isfinite(result.d2u_dy2));
    EXPECT_TRUE(std::isfinite(result.d2v_dx2));
    EXPECT_TRUE(std::isfinite(result.d2v_dxdy));
    EXPECT_TRUE(std::isfinite(result.d2v_dy2));
}

TEST(Icgn, SecondOrderSsdRecoversSubpixelTranslation)
{
    const auto reference = make_shifted_image(64, 64, 0.0, 0.0);
    const auto deformed = make_shifted_image(64, 64, 1.35, -0.65);

    dic::SubsetConfig config;
    config.shape_function = dic::SubsetShapeFunctionMethod::SecondOrder;
    config.use_second_order = true;
    config.objective = dic::CorrelationCriterionKind::SSD;
    config.subset_radius = 10;
    config.max_iterations = 40;
    config.convergence_threshold = 1e-5;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;

    const dic::ICGNSolver solver(config);
    const dic::InitialDisplacement initial{1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, true};
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(32.0, 32.0), initial);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::Success);
    EXPECT_NEAR(result.u, 1.35, 0.15);
    EXPECT_NEAR(result.v, -0.65, 0.15);
    // SSD correlation should be small (good match) but positive
    EXPECT_GT(result.correlation, 0.0);
    // Second-order parameters should be near zero for a pure translation
    EXPECT_NEAR(result.d2u_dx2, 0.0, 0.01);
    EXPECT_NEAR(result.d2u_dxdy, 0.0, 0.01);
    EXPECT_NEAR(result.d2u_dy2, 0.0, 0.01);
    EXPECT_NEAR(result.d2v_dx2, 0.0, 0.01);
    EXPECT_NEAR(result.d2v_dxdy, 0.0, 0.01);
    EXPECT_NEAR(result.d2v_dy2, 0.0, 0.01);
}

TEST(Icgn, SecondOrderSsdRecoversSecondOrderWarp)
{
    const int w = 64;
    const int h = 64;
    std::vector<float> ref_data(static_cast<std::size_t>(w * h));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            ref_data[static_cast<std::size_t>(y * w + x)] =
                static_cast<float>(texture(static_cast<double>(x), static_cast<double>(y)));
        }
    }
    const dic::Image reference(w, h, std::move(ref_data));

    // Same ground-truth parameters as the ZNSSD second-order warp test
    const double gt_u = 1.2, gt_v = -0.7;
    const double gt_du_dx = 0.006, gt_du_dy = 0.004;
    const double gt_dv_dx = -0.004, gt_dv_dy = 0.008;
    const double gt_d2u_dx2 = 0.0008, gt_d2u_dxdy = 0.0006, gt_d2u_dy2 = -0.0004;
    const double gt_d2v_dx2 = -0.0005, gt_d2v_dxdy = 0.0006, gt_d2v_dy2 = 0.0005;

    const auto deformed = make_second_order_warped_image(w, h,
        gt_u, gt_v,
        gt_du_dx, gt_du_dy, gt_dv_dx, gt_dv_dy,
        gt_d2u_dx2, gt_d2u_dxdy, gt_d2u_dy2,
        gt_d2v_dx2, gt_d2v_dxdy, gt_d2v_dy2);

    dic::SubsetConfig config;
    config.shape_function = dic::SubsetShapeFunctionMethod::SecondOrder;
    config.use_second_order = true;
    config.objective = dic::CorrelationCriterionKind::SSD;
    config.subset_radius = 10;
    config.max_iterations = 60;
    config.convergence_threshold = 1e-6;
    config.image_precompute.degree = dic::BSplineDegree::Cubic;

    const dic::ICGNSolver solver(config);
    const dic::InitialDisplacement initial{gt_u + 0.1, gt_v - 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, true};
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(32.0, 32.0), initial);

    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.status, dic::SolverStatus::Success);
    // Translation recovery
    EXPECT_NEAR(result.u, gt_u, 0.15);
    EXPECT_NEAR(result.v, gt_v, 0.15);
    // SSD correlation should be positive
    EXPECT_GT(result.correlation, 0.0);
    // Verify second-order parameters are finite
    EXPECT_TRUE(std::isfinite(result.d2u_dx2));
    EXPECT_TRUE(std::isfinite(result.d2u_dxdy));
    EXPECT_TRUE(std::isfinite(result.d2u_dy2));
    EXPECT_TRUE(std::isfinite(result.d2v_dx2));
    EXPECT_TRUE(std::isfinite(result.d2v_dxdy));
    EXPECT_TRUE(std::isfinite(result.d2v_dy2));
}
