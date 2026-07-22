#include <dic/correlation/ssd.hpp>
#include <dic/correlation/zncc.hpp>
#include <dic/correlation/znssd.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(Correlation, SSDEvaluatesSquaredResidual)
{
    dic::SSDCorrelation criterion;

    Eigen::VectorXd reference(3);
    reference << 1.0, 2.0, 4.0;

    Eigen::VectorXd deformed(3);
    deformed << 1.0, 4.0, 1.0;

    EXPECT_DOUBLE_EQ(criterion.evaluate(reference, deformed), 13.0);
}

TEST(Correlation, ZNSSDIsInvariantToLinearIntensityScale)
{
    dic::ZNSSDCorrelation criterion;

    Eigen::VectorXd reference(4);
    reference << 1.0, 2.0, 3.0, 5.0;

    Eigen::VectorXd deformed = 2.0 * reference.array() + 10.0;

    EXPECT_NEAR(criterion.evaluate(reference, deformed), 0.0, 1e-12);
}

TEST(Correlation, ZNSSDIgnoresZeroWeightedNonROISamples)
{
    dic::ZNSSDCorrelation criterion;

    Eigen::VectorXd reference(5);
    reference << 1.0, 2.0, 3.0, 1000.0, -500.0;

    Eigen::VectorXd deformed(5);
    deformed << 3.0, 5.0, 7.0, -9999.0, 8888.0;

    Eigen::VectorXd weights(5);
    weights << 1.0, 1.0, 1.0, 0.0, 0.0;

    EXPECT_NEAR(criterion.evaluate(reference, deformed, weights), 0.0, 1e-12);
}

TEST(Correlation, ZNCCScoresPerfectCorrelation)
{
    dic::ZNCCCorrelation criterion;

    Eigen::VectorXd reference(4);
    reference << 1.0, 2.0, 3.0, 5.0;

    Eigen::VectorXd deformed = 3.0 * reference.array() - 7.0;

    EXPECT_NEAR(criterion.evaluate(reference, deformed), 1.0, 1e-12);
}

TEST(Correlation, ZNCCIgnoresZeroWeightedNonROISamples)
{
    dic::ZNCCCorrelation criterion;

    Eigen::VectorXd reference(5);
    reference << 1.0, 2.0, 3.0, 1000.0, -500.0;

    Eigen::VectorXd deformed(5);
    deformed << 2.0, 4.0, 6.0, -9999.0, 8888.0;

    Eigen::VectorXd weights(5);
    weights << 1.0, 1.0, 1.0, 0.0, 0.0;

    EXPECT_NEAR(criterion.evaluate(reference, deformed, weights), 1.0, 1e-12);
}

TEST(Correlation, ZNCCScoresPerfectInverseCorrelation)
{
    dic::ZNCCCorrelation criterion;

    Eigen::VectorXd reference(4);
    reference << 1.0, 2.0, 3.0, 5.0;

    Eigen::VectorXd deformed = -2.0 * reference.array() + 8.0;

    EXPECT_NEAR(criterion.evaluate(reference, deformed), -1.0, 1e-12);
}

TEST(Correlation, ZNSSDRejectsFlatSubset)
{
    dic::ZNSSDCorrelation criterion;

    const auto flat = Eigen::VectorXd::Ones(4);
    Eigen::VectorXd textured(4);
    textured << 1.0, 2.0, 3.0, 4.0;

    EXPECT_THROW((void)criterion.evaluate(flat, textured), std::invalid_argument);
}

TEST(Correlation, RejectsMismatchedVectorSizes)
{
    dic::SSDCorrelation criterion;

    EXPECT_THROW(
        (void)criterion.evaluate(Eigen::VectorXd::Ones(3), Eigen::VectorXd::Ones(4)),
        std::invalid_argument
    );
}

TEST(Correlation, RejectsInvalidWeights)
{
    dic::SSDCorrelation criterion;

    Eigen::VectorXd reference(3);
    reference << 1.0, 2.0, 3.0;

    Eigen::VectorXd deformed(3);
    deformed << 1.0, 2.0, 3.0;

    Eigen::VectorXd weights(3);
    weights << 1.0, -1.0, 1.0;

    EXPECT_THROW(
        (void)criterion.evaluate(reference, deformed, weights),
        std::invalid_argument
    );
}
