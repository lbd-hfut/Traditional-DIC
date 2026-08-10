"""Validate ComplexCylinderDIC reconstruction against ground truth.

Compares the stitched subset surface / displacement field from a multi-view
3D-DIC run against the theoretical (sinusoidal cylinder) ground truth shipped
in the case directory. Scale is already recovered (1:1 mm), so only a rigid
ICP alignment (Kabsch, no scaling) is needed before computing errors.

Usage (from the repository root):

    python tools/validate_complex_cylinder.py
    python tools/validate_complex_cylinder.py --case case/multi_DIC/ComplexCylinderDIC

Standalone verification tool; no production code depends on it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_stitched(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (reference surface points, displacement vectors) as (N,3) arrays."""
    pts: list[list[float]] = []
    u: list[list[float]] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pts.append([float(row["X0"]), float(row["Y0"]), float(row["Z0"])])
                u.append([float(row["Ux"]), float(row["Uy"]), float(row["Uz"])])
            except (ValueError, KeyError):
                continue
    pts = np.asarray(pts, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(u).all(axis=1)
    return pts[valid], u[valid]


def load_cameras_scaled(json_path: Path) -> dict[str, dict]:
    """Load scaled calibration cameras keyed by label.

    The metric-scaled rig lives under ``scaled_cameras``; the plain ``cameras``
    list keeps the raw SFM units (camera centers ~<4 instead of ~400-900 mm).
    Pairwise triangulation uses ``scaled_cameras``, so those are the ones whose
    frame matches the stitched surface.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for cam in data.get("scaled_cameras") or data.get("cameras") or []:
        out[str(cam.get("label"))] = cam
    return out


# ---------------------------------------------------------------------------
# Rigid alignment
# ---------------------------------------------------------------------------
def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) minimizing ||(R P + t) - Q|| with no scaling."""
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    pn = P - pc
    qn = Q - qc
    h = pn.T @ qn
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    t = qc - r @ pc
    return r, t


def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iter: int = 80,
    robust_pct: float = 95.0,
    tol: float = 1e-8,
    sample: int = 30000,
    init_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Point-to-point ICP on a down-sampled source (fixed seed), then evaluate
    the final transform on the full source. Returns (R, t, final_mean_dist,
    aligned, nearest_dists) evaluated on the full source.

    ``init_t`` pre-applies a rigid translation before the first nearest-neighbor
    match. A centroid-aligned start matters here: the reconstruction sits at
    Z~478 while the ground truth is centered at Z=0, and without it the first
    correspondence is far off and ICP traps in a wrong local minimum.
    """
    n = len(source)
    if n > sample:
        idx = np.random.RandomState(0).choice(n, sample, replace=False)
        s = source[idx]
    else:
        s = source
    tree = cKDTree(target)
    r = np.eye(3)
    t = np.zeros(3) if init_t is None else np.asarray(init_t, dtype=np.float64).copy()
    prev_meand = float("inf")
    for _ in range(max_iter):
        aligned = (r @ s.T).T + t
        d, nidx = tree.query(aligned, k=1)
        meand = float(d.mean())
        if abs(prev_meand - meand) < tol:
            break
        prev_meand = meand
        mask = d < np.percentile(d, robust_pct)
        # Incremental ICP: solve the delta from the *currently aligned* points
        # to their target. Using the raw source here re-solves the full pose
        # every iteration and the composed translation diverges.
        r_new, t_new = kabsch(aligned[mask], target[nidx[mask]])
        # Compose: new pose applied to original source
        r = r_new @ r
        t = r_new @ t + t_new
    aligned = (r @ source.T).T + t
    d, nidx = tree.query(aligned, k=1)
    return r, t, float(d.mean()), aligned, d


def camera_center(cam: dict) -> np.ndarray:
    """Camera center (mm). Prefer the explicit ``camera_center`` field, else
    derive it from R/t as -R^T t (the metric-meta fallback cameras, e.g. the
    plain CylinderDIC case, carry R/t only)."""
    cc = cam.get("camera_center")
    if cc is not None:
        return np.asarray(cc, dtype=np.float64)
    R = np.asarray(cam["R"], dtype=np.float64)
    t = np.asarray(cam["t"], dtype=np.float64)
    return -(R.T @ t)


def stats(d: np.ndarray, name: str) -> None:
    pct = lambda p: float(np.percentile(d, p))
    print(
        f"{name:>18s}  mean={d.mean():9.4f}  rmse={np.sqrt((d**2).mean()):9.4f}  "
        f"p50={pct(50):9.4f}  p95={pct(95):9.4f}  p99={pct(99):9.4f}  max={d.max():9.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=Path,
        default=PROJECT_ROOT / "case" / "multi_DIC" / "ComplexCylinderDIC",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Path to a tagged result root such as case/result_znssd_icgn_2nd; "
        "defaults to <case>/result. Use this when per-method outputs are "
        "stored in tag-suffixed directories.",
    )
    parser.add_argument(
        "--solver",
        choices=("subset", "mesh"),
        default="subset",
        help="Which solver branch the stitched output came from; selects the "
        "reconstruct/stitched/<solver>/ subdirectory (default: subset).",
    )
    parser.add_argument(
        "--element",
        default=None,
        help="Mesh element type (T3/Q4/Q8) to validate; reads "
        "reconstruct/stitched/mesh/<element>/. Ignored for solver=subset.",
    )
    args = parser.parse_args()

    case = args.case
    gt_dir = case / "ground_truth"
    result = args.result_root if args.result_root is not None else case / "result"

    # Ground truth
    gt_ref = np.load(gt_dir / "theoretical_surface_points.npy").astype(np.float64)
    gt_U = np.load(gt_dir / "theoretical_deformation_field_step001.npy").astype(np.float64)
    gt_cam = np.load(gt_dir / "theoretical_camera_parameters.npz")
    gt_K = np.asarray(gt_cam["K"], dtype=np.float64)
    gt_centers = np.asarray(gt_cam["camera_centers"], dtype=np.float64)

    # Reconstruction: mesh output lives under stitched/mesh/<element>/ so each
    # element type is validated independently; subset has no element dimension.
    stitched_dir = result / "reconstruct" / "stitched" / args.solver
    if args.element is not None:
        stitched_dir = stitched_dir / args.element
    rec_pts, rec_U = load_stitched(stitched_dir / "stitched_points.csv")
    rec_cams = load_cameras_scaled(result / "calibration" / "calibration_result_scaled.json")
    print(f"reconstruction points: {len(rec_pts)}   ground-truth points: {len(gt_ref)}")
    print(f"reconstruction |U| mean={np.linalg.norm(rec_U, axis=1).mean():.4f} max={np.linalg.norm(rec_U, axis=1).max():.4f}")
    print(f"ground-truth    |U| mean={np.linalg.norm(gt_U, axis=1).mean():.4f} max={np.linalg.norm(gt_U, axis=1).max():.4f}")
    print()

    # 1. Rigid alignment. Both surfaces are the same geometry (Y-axis cylinder,
    #    radius 80 mm, full 360 deg ring), so a centroid pre-alignment + ICP on
    #    the surface itself is the right anchor. Camera-center alignment is NOT
    #    used: the reconstructed rig layout (cameras scattered at 0-950 mm
    #    radius) differs structurally from the theoretical ring (exactly 480 mm),
    #    so those two point sets share no rigid transform.
    init_t = gt_ref.mean(axis=0) - rec_pts.mean(axis=0)
    R, t, _, aligned, nearest_d = icp(rec_pts, gt_ref, init_t=init_t)
    print("[rigid alignment: centroid pre-align + surface ICP]")
    print(f"  initial centroid shift t0 = {init_t.round(3)}")
    print(f"  final mean nearest distance: {nearest_d.mean():.4f} mm")
    print(f"  rotation R =\n{R.round(4)}")
    print(f"  translation t = {t.round(3)}")
    print()

    # 2. Surface reconstruction error (mm)
    print("[surface reconstruction error / mm]")
    stats(nearest_d, "dist-to-GT")

    # 3. Displacement error: rotate reconstructed displacement into GT frame,
    #    compare with the GT displacement at the nearest reference point.
    U_al = (R @ rec_U.T).T
    tree = cKDTree(gt_ref)
    _, idx = tree.query(aligned, k=1)
    U_gt_nearest = gt_U[idx]
    dU = U_al - U_gt_nearest
    dU_norm = np.linalg.norm(dU, axis=1)

    print("\n[displacement error / mm]")
    stats(dU_norm, "|dU|")

    mag_u = np.linalg.norm(U_gt_nearest, axis=1)
    mag_a = np.linalg.norm(U_al, axis=1)
    dot = np.sum(U_al * U_gt_nearest, axis=1)
    denom = mag_u * mag_a
    valid_ang = denom > 1e-6
    ang = np.degrees(np.arccos(np.clip(dot[valid_ang] / denom[valid_ang], -1.0, 1.0)))
    print(f"displacement direction error (deg): mean={ang.mean():.2f}  median={np.median(ang):.2f}  p95={np.percentile(ang, 95):.2f}")

    # 4. Intrinsics comparison (already the same scale)
    print("\n[intrinsics: reconstructed fx,cx,cy vs theoretical]")
    names = [str(n) for n in gt_cam["cam_names"]]
    fx_err = []
    for name in names:
        rc = rec_cams.get(name)
        if rc is None:
            print(f"  {name}: not found in reconstruction")
            continue
        rK = np.asarray(rc.get("K"), dtype=np.float64)
        gK = gt_K[names.index(name)]
        fx_err.append(abs(rK[0, 0] - gK[0, 0]))
        print(
            f"  {name}: fx rec={rK[0,0]:.3f} gt={gK[0,0]:.3f} (d={abs(rK[0,0]-gK[0,0]):.3f})  "
            f"cx rec={rK[0,2]:.3f} gt={gK[0,2]:.3f}  cy rec={rK[1,2]:.3f} gt={gK[1,2]:.3f}"
        )
    if fx_err:
        print(f"  fx |error| mean={np.mean(fx_err):.3f} px  max={max(fx_err):.3f} px")

    # 5. Camera-center comparison after applying the surface alignment.
    #    Note: the reconstructed rig layout differs structurally from the
    #    theoretical ring (see comment in step 1), so a large residual here
    #    reflects an extrinsic-rig discrepancy, not surface accuracy.
    print("\n[camera centers after rigid alignment]")
    rec_centers = np.stack(
        [camera_center(rec_cams[n]) for n in names if n in rec_cams]
    )
    centers_aligned = (R @ rec_centers.T).T + t
    # match order against GT centers
    rec_names_present = [n for n in names if n in rec_cams]
    gt_centers_matched = np.stack([gt_centers[names.index(n)] for n in rec_names_present])
    center_err = np.linalg.norm(centers_aligned - gt_centers_matched, axis=1)
    stats(center_err, "center err/mm")
    print(f"  (aligned {len(rec_names_present)} camera centers)")


if __name__ == "__main__":
    main()
