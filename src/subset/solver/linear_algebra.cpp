#include <dic/subset/solver/linear_algebra.hpp>

#include <cmath>
#include <stdexcept>

namespace dic {

bool cholesky_in_place(Eigen::MatrixXd& matrix, double lambda)
{
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Cholesky input must be square.");
    }

    const int size = static_cast<int>(matrix.rows());
    for (int i = 0; i < size; ++i) {
        if (i > 0) {
            for (int j = size - 1; j >= i; --j) {
                double sum = 0.0;
                for (int k = 0; k < i; ++k) {
                    sum += matrix(j, k) * matrix(i, k);
                }
                matrix(j, i) -= sum;
            }
        }
        if (matrix(i, i) <= lambda) {
            return false;
        }
        const double diagonal = std::sqrt(matrix(i, i));
        for (int j = i; j < size; ++j) {
            matrix(j, i) /= diagonal;
        }
    }
    return true;
}

void forward_substitution_in_place(Eigen::VectorXd& vector, const Eigen::MatrixXd& lower)
{
    if (lower.rows() != lower.cols() || vector.size() != lower.rows()) {
        throw std::invalid_argument("Forward substitution dimensions do not match.");
    }
    const int size = static_cast<int>(vector.size());
    vector(0) /= lower(0, 0);
    for (int i = 1; i < size; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += lower(i, j) * vector(j);
        }
        vector(i) = (vector(i) - sum) / lower(i, i);
    }
}

void backward_substitution_in_place(Eigen::VectorXd& vector, const Eigen::MatrixXd& lower)
{
    if (lower.rows() != lower.cols() || vector.size() != lower.rows()) {
        throw std::invalid_argument("Backward substitution dimensions do not match.");
    }
    const int size = static_cast<int>(vector.size());
    vector(size - 1) /= lower(size - 1, size - 1);
    for (int i = size - 2; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < size; ++j) {
            sum += lower(j, i) * vector(j);
        }
        vector(i) = (vector(i) - sum) / lower(i, i);
    }
}

} // namespace dic
