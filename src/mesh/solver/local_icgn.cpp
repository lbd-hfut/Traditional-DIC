#include <dic/mesh/solver/local_icgn.hpp>
#include <dic/interpolation/bspline.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef TRADITIONAL_DIC_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace dic {

// ============================================================
// Helper: build df/dp row for 6-DOF affine ICGN
// Warp: x' = x + u + du/dx*X + du/dy*Y
//       y' = y + v + dv/dx*X + dv/dy*Y
// where X = dx[i], Y = dy[i] (local offsets from subset center)
// ============================================================
static Eigen::Matrix<double, 6, 1> build_df_dp_row(
    double X, double Y, double fx, double fy)
{
    Eigen::Matrix<double, 6, 1> row;
    row << fx, fy,
           X * fx, Y * fx,
           X * fy, Y * fy;
    return row;
}

// ============================================================
// Helper: IC inverse-compositional warp update
// Given current warp p and incremental delta, compute p_new
// via the inverse-compositional composition rule.
// ============================================================
static bool inverse_compositional_update_affine(
    const Eigen::Matrix<double, 6, 1>& p_old,
    const Eigen::Matrix<double, 6, 1>& delta,
    Eigen::Matrix<double, 6, 1>& p_new)
{
    double du = p_old(0), dv = p_old(1);
    double dudx = p_old(2), dudy = p_old(3);
    double dvdx = p_old(4), dvdy = p_old(5);

    double dp0 = delta(0), dp1 = delta(1);
    double dp2 = delta(2), dp3 = delta(3);
    double dp4 = delta(4), dp5 = delta(5);

    double denom = dp2 + dp5 + dp2 * dp5 - dp3 * dp4 + 1.0;
    if (!std::isfinite(denom) || std::abs(denom) < 1e-12) {
        return false;
    }

    p_new(0) = du - ((dudx + 1.0) * (dp0 + dp0 * dp5 - dp1 * dp3)) / denom
                 - (dudy * (dp1 - dp0 * dp4 + dp1 * dp2)) / denom;
    p_new(1) = dv - ((dvdy + 1.0) * (dp1 - dp0 * dp4 + dp1 * dp2)) / denom
                 - (dvdx * (dp0 + dp0 * dp5 - dp1 * dp3)) / denom;
    p_new(2) = ((dp5 + 1.0) * (dudx + 1.0)) / denom
                 - (dp4 * dudy) / denom - 1.0;
    p_new(3) = (dudy * (dp2 + 1.0)) / denom
                 - (dp3 * (dudx + 1.0)) / denom;
    p_new(4) = (dvdx * (dp5 + 1.0)) / denom
                 - (dp4 * (dvdy + 1.0)) / denom;
    p_new(5) = ((dp2 + 1.0) * (dvdy + 1.0)) / denom
                 - (dp3 * dvdx) / denom - 1.0;

    return p_new.allFinite();
}

// ============================================================
// solve: 6-DOF affine ICGN subpixel refinement
// ============================================================
LocalICGNResult LocalICGNSolver::solve(
    const double* f_buffer,
    const double* fx_ref,
    const double* fy_ref,
    int n_pixels,
    double xc, double yc,
    const double* dx, const double* dy,
    BSplineInterpolator* bsp,
    double u0, double v0)
{
    LocalICGNResult result;
    int n = n_pixels;

    // ---- Precompute reference statistics (ZNSSD normalisation) ----
    double fm = 0.0;
    for (int i = 0; i < n; ++i) fm += f_buffer[i];
    fm /= n;

    double deltaf_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = f_buffer[i] - fm;
        deltaf_sq += d * d;
    }
    if (deltaf_sq < params_.lambda_reg) {
        return result;
    }
    double deltaf_inv = 1.0 / std::sqrt(deltaf_sq);

    // ---- Build constant df/dp (Jacobian of reference w.r.t. warp params) ----
    Eigen::MatrixXd df_dp(n, 6);
    for (int i = 0; i < n; ++i) {
        df_dp.row(i) = build_df_dp_row(dx[i], dy[i], fx_ref[i], fy_ref[i]);
    }

    // ---- Constant Hessian (ICGN: fixed from reference) ----
    Eigen::Matrix<double, 6, 6> H =
        2.0 * deltaf_inv * deltaf_inv * df_dp.transpose() * df_dp;
    for (int i = 0; i < 6; ++i) H(i, i) += params_.lambda_reg;

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(H);
    if (ldlt.info() != Eigen::Success) {
        return result;
    }

    // ---- Initial warp ----
    Eigen::Matrix<double, 6, 1> p = Eigen::Matrix<double, 6, 1>::Zero();
    p(0) = u0;
    p(1) = v0;

    std::vector<double> g_buf(n);

    // ---- ICGN iteration ----
    for (int iter = 0; iter < params_.max_iter; ++iter) {
        // Warp positions in deformed image
        for (int i = 0; i < n; ++i) {
            double X = dx[i], Y = dy[i];
            double xs = xc + X + p(0) + p(2) * X + p(3) * Y;
            double ys = yc + Y + p(1) + p(4) * X + p(5) * Y;

            // Boundary check: BSplineInterpolator clips internally,
            // but we guard against obviously invalid coordinates.
            if (!std::isfinite(xs) || !std::isfinite(ys)) {
                return result;
            }

            if (bsp) {
                g_buf[i] = bsp->value(xs, ys);
            } else {
                g_buf[i] = 0.0;  // No deformed image → constant
            }
        }

        // ---- Deformed patch statistics ----
        double gm = 0.0;
        for (int i = 0; i < n; ++i) gm += g_buf[i];
        gm /= n;

        double deltag_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = g_buf[i] - gm;
            deltag_sq += d * d;
        }
        if (deltag_sq < params_.lambda_reg) {
            return result;
        }
        double deltag_inv = 1.0 / std::sqrt(deltag_sq);

        // ---- ZNSSD residual ----
        double corr = 0.0;
        Eigen::VectorXd resid(n);
        for (int i = 0; i < n; ++i) {
            double f_norm = (f_buffer[i] - fm) * deltaf_inv;
            double g_norm = (g_buf[i] - gm) * deltag_inv;
            double diff = f_norm - g_norm;
            resid(i) = diff;
            corr += diff * diff;
        }

        // ---- Gradient and update ----
        Eigen::Matrix<double, 6, 1> grad =
            2.0 * deltaf_inv * df_dp.transpose() * resid;

        Eigen::Matrix<double, 6, 1> delta = -ldlt.solve(grad);

        double diffnorm = delta.norm();
        if (!std::isfinite(diffnorm)) {
            return result;
        }

        if (diffnorm < params_.cutoff_diffnorm) {
            result.success = true;
            result.u = p(0);
            result.v = p(1);
            result.corr_coef = corr;
            result.diffnorm = diffnorm;
            result.iterations = iter + 1;
            return result;
        }

        // ---- IC warp update ----
        Eigen::Matrix<double, 6, 1> p_next;
        if (!inverse_compositional_update_affine(p, delta, p_next)) {
            return result;
        }

        p = p_next;
        if (!p.allFinite()) {
            return result;
        }
    }

    // Max iterations reached
    result.success = true;
    result.u = p(0);
    result.v = p(1);
    result.iterations = params_.max_iter;
    return result;
}

// ============================================================
// coarse_search: integer-pixel template matching
// ============================================================
CoarseSearchResult LocalICGNSolver::coarse_search(
    const double* ref_img,
    const double* def_img,
    int img_h, int img_w,
    int cx, int cy,
    int subset_radius,
    int search_radius,
    const uint8_t* mask_pad,
    int mask_pad_h, int mask_pad_w)
{
    CoarseSearchResult result;

    // Extract reference patch
    int y0 = std::max(0, cy - subset_radius);
    int y1 = std::min(img_h, cy + subset_radius + 1);
    int x0 = std::max(0, cx - subset_radius);
    int x1 = std::min(img_w, cx + subset_radius + 1);
    int rh = y1 - y0, rw = x1 - x0;
    if (rh <= 0 || rw <= 0) return result;

    // Search region
    int sy0 = std::max(0, y0 - search_radius);
    int sy1 = std::min(img_h, y1 + search_radius + 1);
    int sx0 = std::max(0, x0 - search_radius);
    int sx1 = std::min(img_w, x1 + search_radius + 1);
    int sh = sy1 - sy0, sw = sx1 - sx0;

#ifdef TRADITIONAL_DIC_HAS_OPENCV
    // ---- OpenCV path: cv::matchTemplate ----
    cv::Mat ref_patch(rh, rw, CV_64FC1);
    for (int i = 0; i < rh; ++i)
        for (int j = 0; j < rw; ++j)
            ref_patch.at<double>(i, j) = ref_img[(y0 + i) * img_w + (x0 + j)];

    cv::Mat search_def(sh, sw, CV_64FC1);
    for (int i = 0; i < sh; ++i)
        for (int j = 0; j < sw; ++j)
            search_def.at<double>(i, j) = def_img[(sy0 + i) * img_w + (sx0 + j)];

    cv::Mat ref_32f, search_32f;
    ref_patch.convertTo(ref_32f, CV_32F);
    search_def.convertTo(search_32f, CV_32F);

    cv::Mat res;
    if (mask_pad && mask_pad_h > 0 && mask_pad_w > 0) {
        cv::Mat mask_patch(rh, rw, CV_8UC1);
        for (int i = 0; i < rh; ++i)
            for (int j = 0; j < rw; ++j)
                mask_patch.at<uint8_t>(i, j) =
                    mask_pad[(y0 + i) * mask_pad_w + (x0 + j)];
        cv::matchTemplate(search_32f, ref_32f, res,
                          cv::TM_CCOEFF_NORMED, mask_patch);
    } else {
        cv::matchTemplate(search_32f, ref_32f, res, cv::TM_CCOEFF_NORMED);
    }

    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(res, nullptr, &max_val, nullptr, &max_loc);

    result.dy = static_cast<double>(sy0 + max_loc.y - y0);
    result.dx = static_cast<double>(sx0 + max_loc.x - x0);
    return result;

#else
    // ---- No OpenCV: manual NCC search ----
    // Compute reference patch mean and norm
    double ref_mean = 0.0;
    int ref_count = 0;
    for (int i = 0; i < rh; ++i) {
        for (int j = 0; j < rw; ++j) {
            if (mask_pad && mask_pad_h > 0 && mask_pad_w > 0) {
                if (!mask_pad[(y0 + i) * mask_pad_w + (x0 + j)]) continue;
            }
            ref_mean += ref_img[(y0 + i) * img_w + (x0 + j)];
            ref_count++;
        }
    }
    if (ref_count == 0) return result;
    ref_mean /= ref_count;

    double ref_norm_sq = 0.0;
    for (int i = 0; i < rh; ++i) {
        for (int j = 0; j < rw; ++j) {
            if (mask_pad && mask_pad_h > 0 && mask_pad_w > 0) {
                if (!mask_pad[(y0 + i) * mask_pad_w + (x0 + j)]) continue;
            }
            double d = ref_img[(y0 + i) * img_w + (x0 + j)] - ref_mean;
            ref_norm_sq += d * d;
        }
    }
    if (ref_norm_sq < 1e-12) return result;

    // Sliding window NCC
    double best_ncc = -2.0;
    int best_dy = 0, best_dx = 0;
    for (int dy = 0; dy <= sh - rh; ++dy) {
        for (int dx_ = 0; dx_ <= sw - rw; ++dx_) {
            double def_mean = 0.0;
            int def_count = 0;
            for (int i = 0; i < rh; ++i) {
                for (int j = 0; j < rw; ++j) {
                    if (mask_pad && mask_pad_h > 0 && mask_pad_w > 0) {
                        if (!mask_pad[(y0 + i) * mask_pad_w + (x0 + j)]) continue;
                    }
                    def_mean += def_img[(sy0 + dy + i) * img_w + (sx0 + dx_ + j)];
                    def_count++;
                }
            }
            if (def_count == 0) continue;
            def_mean /= def_count;

            double def_norm_sq = 0.0, cross = 0.0;
            for (int i = 0; i < rh; ++i) {
                for (int j = 0; j < rw; ++j) {
                    if (mask_pad && mask_pad_h > 0 && mask_pad_w > 0) {
                        if (!mask_pad[(y0 + i) * mask_pad_w + (x0 + j)]) continue;
                    }
                    double d_ref = ref_img[(y0 + i) * img_w + (x0 + j)] - ref_mean;
                    double d_def = def_img[(sy0 + dy + i) * img_w + (sx0 + dx_ + j)] - def_mean;
                    cross += d_ref * d_def;
                    def_norm_sq += d_def * d_def;
                }
            }
            if (def_norm_sq < 1e-12) continue;
            double ncc = cross / std::sqrt(ref_norm_sq * def_norm_sq);
            if (ncc > best_ncc) {
                best_ncc = ncc;
                best_dy = dy;
                best_dx = dx_;
            }
        }
    }

    result.dy = static_cast<double>(sy0 + best_dy - y0);
    result.dx = static_cast<double>(sx0 + best_dx - x0);
    return result;
#endif
}

} // namespace dic
