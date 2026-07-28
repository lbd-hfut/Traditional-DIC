#include <dic/core/image.hpp>
#include <dic/initialization/initializer.hpp>
#include <dic/subset/solver/forward_gauss_newton.hpp>
#include <gtest/gtest.h>

#include <vector>

TEST(ForwardGaussNewtonSolver, PlaceholderReturnsInitialDisplacementAsNotConverged)
{
    const dic::Image reference(3, 3, std::vector<float>(9, 0.0F));
    const dic::Image deformed(3, 3, std::vector<float>(9, 0.0F));
    const dic::InitialDisplacement initial{1.25, -0.5, 0.0, 0.0, 0.0, 0.0, 0.8, true};

    const dic::ForwardGaussNewtonSolver solver;
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(2.0, 1.0), initial);

    EXPECT_DOUBLE_EQ(result.x, 2.0);
    EXPECT_DOUBLE_EQ(result.y, 1.0);
    EXPECT_DOUBLE_EQ(result.u, 1.25);
    EXPECT_DOUBLE_EQ(result.v, -0.5);
    EXPECT_DOUBLE_EQ(result.correlation, 0.8);
    EXPECT_EQ(result.status, dic::SolverStatus::NotConverged);
    EXPECT_FALSE(result.valid);
}

TEST(ForwardGaussNewtonSolver, SupportsSecondOrderPlaceholderSelection)
{
    const dic::Image reference(3, 3, std::vector<float>(9, 0.0F));
    const dic::Image deformed(3, 3, std::vector<float>(9, 0.0F));
    const dic::InitialDisplacement initial{0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.6, true};

    dic::SubsetConfig config;
    config.shape_function = dic::SubsetShapeFunctionMethod::SecondOrder;
    config.use_second_order = true;

    const dic::ForwardGaussNewtonSolver solver(config);
    const auto result = solver.solve(reference, deformed, Eigen::Vector2d(1.0, 1.0), initial);

    EXPECT_DOUBLE_EQ(result.u, 0.5);
    EXPECT_DOUBLE_EQ(result.v, 0.25);
    EXPECT_EQ(result.status, dic::SolverStatus::NotConverged);
    EXPECT_FALSE(result.valid);
}

TEST(ForwardGaussNewtonSolver, DispatchesObjectivePlaceholders)
{
    const dic::Image reference(3, 3, std::vector<float>(9, 0.0F));
    const dic::Image deformed(3, 3, std::vector<float>(9, 0.0F));
    const dic::InitialDisplacement initial{0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.4, true};

    dic::SubsetConfig config;
    for (const auto objective : {dic::CorrelationCriterionKind::SSD, dic::CorrelationCriterionKind::ZNSSD}) {
        config.objective = objective;
        for (const auto shape : {dic::SubsetShapeFunctionMethod::FirstOrder,
                                 dic::SubsetShapeFunctionMethod::SecondOrder}) {
            config.shape_function = shape;
            config.use_second_order = shape == dic::SubsetShapeFunctionMethod::SecondOrder;
            const dic::ForwardGaussNewtonSolver solver(config);
            const auto result = solver.solve(reference, deformed, Eigen::Vector2d(1.0, 1.0), initial);

            EXPECT_DOUBLE_EQ(result.u, 0.2);
            EXPECT_DOUBLE_EQ(result.v, -0.1);
            EXPECT_EQ(result.status, dic::SolverStatus::NotConverged);
            EXPECT_FALSE(result.valid);
        }
    }
}
