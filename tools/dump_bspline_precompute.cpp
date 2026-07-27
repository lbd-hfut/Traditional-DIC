#include <dic/core/image.hpp>
#include <dic/interpolation/bspline.hpp>

#include <Eigen/Dense>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_csv(const std::filesystem::path& path, const Eigen::MatrixXd& matrix)
{
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    output << std::setprecision(17);
    for (Eigen::Index y = 0; y < matrix.rows(); ++y) {
        for (Eigen::Index x = 0; x < matrix.cols(); ++x) {
            if (x > 0) {
                output << ',';
            }
            output << matrix(y, x);
        }
        output << '\n';
    }
}

dic::BSplineDegree parse_degree(const std::string& value)
{
    if (value == "1") {
        return dic::BSplineDegree::Linear;
    }
    if (value == "3") {
        return dic::BSplineDegree::Cubic;
    }
    if (value == "5") {
        return dic::BSplineDegree::Quintic;
    }
    throw std::invalid_argument("Degree must be one of 1, 3, or 5.");
}

dic::Image read_image_csv(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Failed to open input CSV: " + path.string());
    }

    std::vector<float> values;
    std::string line;
    int width = -1;
    int height = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream stream(line);
        std::string cell;
        int row_width = 0;
        while (std::getline(stream, cell, ',')) {
            values.push_back(static_cast<float>(std::stod(cell)));
            ++row_width;
        }
        if (width < 0) {
            width = row_width;
        } else if (row_width != width) {
            throw std::runtime_error("Input CSV has inconsistent row widths.");
        }
        ++height;
    }

    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Input CSV is empty.");
    }
    return dic::Image(width, height, std::move(values));
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 5) {
            std::cerr << "Usage: dump_bspline_precompute <image> <output_dir> <degree> <border>\n";
            return EXIT_FAILURE;
        }

        const std::filesystem::path image_path(argv[1]);
        const std::filesystem::path output_dir(argv[2]);
        const auto degree = parse_degree(argv[3]);
        const int border = std::stoi(argv[4]);

        std::filesystem::create_directories(output_dir);

        const dic::Image image = image_path.extension() == ".csv"
            ? read_image_csv(image_path)
            : dic::Image(image_path.string());
        const dic::BSplinePrecomputeConfig config{degree, border, true};
        const dic::BSplineImagePreprocessor preprocessor(config);
        const auto precomputed = preprocessor.compute(image);

        write_csv(output_dir / "cpp_bcoef.csv", precomputed.coefficients);
        write_csv(output_dir / "cpp_gradient_x.csv", precomputed.gradient_x);
        write_csv(output_dir / "cpp_gradient_y.csv", precomputed.gradient_y);

        std::ofstream summary(output_dir / "cpp_summary.txt");
        summary << "image=" << image_path.string() << '\n';
        summary << "width=" << precomputed.width << '\n';
        summary << "height=" << precomputed.height << '\n';
        summary << "degree=" << static_cast<int>(precomputed.config.degree) << '\n';
        summary << "border=" << precomputed.config.border << '\n';
        summary << "use_exact_prefilter=" << (precomputed.config.use_exact_prefilter ? "true" : "false") << '\n';

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "dump_bspline_precompute failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
