/**
 * @file strain_3d.cpp
 * @brief 3D surface strain computation from triangle face deformation.
 *
 * Implements the algorithm from:
 *   compute_surface_deformation in Multi-DIC/recon3d_bindings.cpp
 *
 * For each triangular face (X1,X2,X3) -> (x1,x2,x3):
 *   1. Build reference and deformed edge vectors and unit normals.
 *   2. Compute reciprocal basis vectors for the reference face.
 *   3. Assemble deformation gradient F and right Cauchy-Green C.
 *   4. Compute Green-Lagrange (E) and Euler-Almansi (e) strain tensors.
 *   5. Eigen-decompose to extract principal strains, max shear, von Mises, etc.
 */

#include <dic/postprocess/strain_3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace dic {

namespace {

double sqr(double v) { return v * v; }

} // anonymous namespace

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

std::vector<SurfaceStrain3D> compute_surface_strain(
    const std::vector<std::array<int, 3>>& faces,
    const std::vector<Eigen::Vector3d>& points_ref,
    const std::vector<Eigen::Vector3d>& points_def,
    const std::vector<bool>& valid_faces)
{
    int n_faces = static_cast<int>(faces.size());
    std::vector<SurfaceStrain3D> result(n_faces);

    for (int i = 0; i < n_faces; ++i) {
        SurfaceStrain3D& s = result[i];
        s.valid = false;

        if (!valid_faces[i]) continue;

        int a = faces[i][0];
        int b = faces[i][1];
        int c = faces[i][2];

        // Bounds check
        int n_pts = static_cast<int>(points_ref.size());
        if (a < 0 || a >= n_pts || b < 0 || b >= n_pts || c < 0 || c >= n_pts)
            continue;

        const Eigen::Vector3d& X1 = points_ref[a];
        const Eigen::Vector3d& X2 = points_ref[b];
        const Eigen::Vector3d& X3 = points_ref[c];
        const Eigen::Vector3d& x1 = points_def[a];
        const Eigen::Vector3d& x2 = points_def[b];
        const Eigen::Vector3d& x3 = points_def[c];

        if (!X1.allFinite() || !X2.allFinite() || !X3.allFinite() ||
            !x1.allFinite() || !x2.allFinite() || !x3.allFinite())
            continue;

        // Edge vectors
        Eigen::Vector3d D1 = X2 - X1;
        Eigen::Vector3d D2 = X3 - X1;
        Eigen::Vector3d d1 = x2 - x1;
        Eigen::Vector3d d2 = x3 - x1;

        // Cross products and face normals
        Eigen::Vector3d crossD = D1.cross(D2);
        Eigen::Vector3d crossd = d1.cross(d2);
        double normD = crossD.norm();
        double normd = crossd.norm();

        if (normD <= 1.0e-18 || normd <= 1.0e-18) continue;

        Eigen::Vector3d D3 = crossD / normD;   // reference unit normal
        Eigen::Vector3d d3 = crossd / normd;   // deformed unit normal

        // Dnorm = D3^T * crossD = |crossD| (since D3 is unit aligned with crossD)
        double Dnorm = D3.dot(crossD);
        if (std::abs(Dnorm) <= 1.0e-18) continue;

        // Reciprocal basis vectors: Drec1 = (D2 x D3)/|Dnorm|, Drec2 = (D3 x D1)/|Dnorm|
        // These satisfy: Drec_i^T D_j = delta_{ij}
        Eigen::Vector3d Drec1 = D2.cross(D3) / Dnorm;
        Eigen::Vector3d Drec2 = D3.cross(D1) / Dnorm;

        // Deformation gradient: F = d1 * Drec1^T + d2 * Drec2^T + d3 * D3^T
        Eigen::Matrix3d F = d1 * Drec1.transpose() +
                             d2 * Drec2.transpose() +
                             d3 * D3.transpose();

        // Right Cauchy-Green deformation tensor
        Eigen::Matrix3d C = F.transpose() * F;

        // Left Cauchy-Green (Finger) tensor and its inverse
        Eigen::Matrix3d B = F * F.transpose();
        Eigen::Matrix3d B_inv;
        {
            double detB = B.determinant();
            if (std::abs(detB) <= 1.0e-18) continue;
            B_inv = B.inverse();
        }

        // Strain tensors
        Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d E_mat = 0.5 * (C - I3);    // Green-Lagrange
        Eigen::Matrix3d e_mat = 0.5 * (I3 - B_inv); // Euler-Almansi

        // --- Eigen-decomposition of C for stretch ratios ---
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> Ceig(C);
        if (Ceig.info() != Eigen::Success) continue;

        Eigen::Vector3d lambdas_sq(
            std::max(0.0, Ceig.eigenvalues()(0)),
            std::max(0.0, Ceig.eigenvalues()(1)),
            std::max(0.0, Ceig.eigenvalues()(2)));

        Eigen::Vector3d lambdas(
            std::sqrt(lambdas_sq(0)),
            std::sqrt(lambdas_sq(1)),
            std::sqrt(lambdas_sq(2)));

        // Remove the stretch closest to 1.0 (the through-thickness direction)
        int remove_idx = 0;
        double closest = std::abs(lambdas(0) - 1.0);
        for (int k = 1; k < 3; ++k) {
            double diff = std::abs(lambdas(k) - 1.0);
            if (diff < closest) {
                closest = diff;
                remove_idx = k;
            }
        }
        double planar[2];
        int pi = 0;
        for (int k = 0; k < 3; ++k)
            if (k != remove_idx)
                planar[pi++] = lambdas(k);
        if (planar[1] < planar[0]) std::swap(planar[0], planar[1]);

        // --- Eigen-decomposition of E for planar principal strains ---
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> Eeig(E_mat);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eeig(e_mat);
        if (Eeig.info() != Eigen::Success || eeig.info() != Eigen::Success) continue;

        // Identify the eigenvector most aligned with the face normal (D3 for E, d3 for e)
        // and remove that eigenvalue — keeping the in-plane ones.
        auto planar_values = [](const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>& eig,
                                 const Eigen::Vector3d& normal) -> std::array<double, 2> {
            int remove = 0;
            double best = -1.0;
            for (int k = 0; k < 3; ++k) {
                Eigen::Vector3d v = eig.eigenvectors().col(k);
                double alignment = std::abs(v.dot(normal));
                if (alignment > best) {
                    best = alignment;
                    remove = k;
                }
            }
            std::array<double, 2> vals;
            int out = 0;
            for (int k = 0; k < 3; ++k)
                if (k != remove)
                    vals[out++] = eig.eigenvalues()(k);
            if (vals[1] < vals[0]) std::swap(vals[0], vals[1]);
            return vals;
        };

        auto Eplanar = planar_values(Eeig, D3);
        auto eplanar = planar_values(eeig, d3);

        // --- Scalar invariants ---
        double traceE = E_mat.trace();
        double tracee = e_mat.trace();

        // Deviatoric parts
        Eigen::Matrix3d Edev = E_mat - (traceE / 3.0) * I3;
        Eigen::Matrix3d edev = e_mat - (tracee / 3.0) * I3;

        double Emgn = E_mat.norm();
        double emgn = e_mat.norm();
        double Eeq = std::sqrt((2.0 / 3.0) * Edev.squaredNorm());
        double eeq = std::sqrt((2.0 / 3.0) * edev.squaredNorm());

        // Populate result
        s.F       = F;
        s.C       = C;
        s.J       = F.determinant();
        s.E       = E_mat;
        s.e       = e_mat;
        s.Emgn    = Emgn;
        s.emgn    = emgn;
        s.Epc1    = Eplanar[0];
        s.Epc2    = Eplanar[1];
        s.epc1    = eplanar[0];
        s.epc2    = eplanar[1];
        s.EShearMax = 0.5 * (Eplanar[1] - Eplanar[0]);
        s.eShearMax = 0.5 * (eplanar[1] - eplanar[0]);
        s.Eeq     = Eeq;
        s.eeq     = eeq;
        s.area    = 0.5 * Dnorm;
        s.d3      = d3;
        s.Lambda1 = planar[0];
        s.Lambda2 = planar[1];
        s.valid   = true;
    }

    return result;
}

} // namespace dic
