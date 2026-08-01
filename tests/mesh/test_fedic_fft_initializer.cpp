#include <dic/core/image.hpp>
#include <dic/mesh/initialization/fedic_fft_initializer.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

double texture(double x, double y)
{
    return 0.5 + 0.23 * std::sin(0.17 * x + 0.11 * y) +
           0.19 * std::cos(0.23 * x - 0.07 * y);
}

dic::Image make_image(int width, int height, int shift_x, int shift_y)
{
    std::vector<float> data(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[static_cast<std::size_t>(y * width + x)] = static_cast<float>(
                texture(static_cast<double>(x - shift_x), static_cast<double>(y - shift_y)));
        }
    }
    return dic::Image(width, height, std::move(data));
}

} // namespace

#ifdef TRADITIONAL_DIC_HAS_OPENCV
TEST(FEDICFFTInitializer, FindsIntegerTranslationFromNormalizedCorrelationPeak)
{
    const auto reference = make_image(96, 96, 0, 0);
    const auto deformed = make_image(96, 96, 4, -3);
    const auto initial = dic::mesh::estimate_fedic_fft_initial_displacement(
        reference, deformed, Eigen::Vector2d(48.0, 48.0), 8, 21);

    ASSERT_TRUE(initial.initial.valid);
    EXPECT_NEAR(initial.initial.u, 4.0, 0.05);
    EXPECT_NEAR(initial.initial.v, -3.0, 0.05);
    EXPECT_GT(initial.initial.zncc, 0.99);
    EXPECT_GT(initial.peak_to_correlation_energy, 1.0);
    EXPECT_GT(initial.peak_to_entropy, 0.0);
}
#endif
