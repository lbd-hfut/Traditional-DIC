import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def read_nodes(path: Path):
    nodes = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                nodes.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return nodes


def median(values):
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def robust_filter(matches, mad_factor):
    mutual = [m for m in matches if m["mutual"]]
    if not mutual:
        return
    med_u = median([m["u"] for m in mutual])
    med_v = median([m["v"] for m in mutual])
    sig_u = 1.4826 * median([abs(m["u"] - med_u) for m in mutual])
    sig_v = 1.4826 * median([abs(m["v"] - med_v) for m in mutual])
    gate_u = max(3.0, mad_factor * sig_u)
    gate_v = max(3.0, mad_factor * sig_v)
    for m in matches:
        m["robust_inlier"] = (
            m["mutual"]
            and abs(m["u"] - med_u) <= gate_u
            and abs(m["v"] - med_v) <= gate_v
        )


def compute_sift_matches(ref, deform, max_features, ratio, mad_factor):
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp0, desc0 = sift.detectAndCompute(ref, None)
    kp1, desc1 = sift.detectAndCompute(deform, None)
    if desc0 is None or desc1 is None:
        return kp0, kp1, []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    fwd = matcher.knnMatch(desc0, desc1, k=2)
    rev = matcher.knnMatch(desc1, desc0, k=2)

    reverse_best = {}
    for pair in rev:
        if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance:
            reverse_best[pair[0].queryIdx] = pair[0].trainIdx

    matches = []
    for pair in fwd:
        if len(pair) < 2:
            continue
        best = pair[0]
        if best.distance >= ratio * pair[1].distance:
            continue
        p0 = kp0[best.queryIdx].pt
        p1 = kp1[best.trainIdx].pt
        matches.append(
            {
                "x0": float(p0[0]),
                "y0": float(p0[1]),
                "x1": float(p1[0]),
                "y1": float(p1[1]),
                "u": float(p1[0] - p0[0]),
                "v": float(p1[1] - p0[1]),
                "descriptor_distance": float(best.distance),
                "ratio_passed": True,
                "mutual": reverse_best.get(best.trainIdx, -1) == best.queryIdx,
                "robust_inlier": False,
            }
        )
    robust_filter(matches, mad_factor)
    return kp0, kp1, matches


def interpolate_nodes(nodes, matches, k, radius):
    inliers = [m for m in matches if m["robust_inlier"]]
    out = []
    for node_id, x, y in nodes:
        nearby = []
        for m in inliers:
            d = math.hypot(x - m["x0"], y - m["y0"])
            if d <= radius:
                nearby.append((d, m))
        nearby.sort(key=lambda item: item[0])
        nearby = nearby[:k]
        record = {
            "node": node_id,
            "x": x,
            "y": y,
            "valid": bool(nearby),
            "u": 0.0,
            "v": 0.0,
            "support_count": len(nearby),
            "nearest_match_distance": nearby[0][0] if nearby else 0.0,
        }
        if nearby:
            weights = [1.0 / (d * d + 1.0) for d, _ in nearby]
            sw = sum(weights)
            record["u"] = sum(w * m["u"] for w, (_, m) in zip(weights, nearby)) / sw
            record["v"] = sum(w * m["v"] for w, (_, m) in zip(weights, nearby)) / sw
        out.append(record)
    return out


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, kp0, kp1, matches, nodes, args):
    ratio_count = sum(1 for m in matches if m["ratio_passed"])
    mutual_count = sum(1 for m in matches if m["mutual"])
    inlier_count = sum(1 for m in matches if m["robust_inlier"])
    valid = [n for n in nodes if n["valid"]]
    max_mag = max((math.hypot(n["u"], n["v"]) for n in valid), default=0.0)
    mean_u = sum(n["u"] for n in valid) / len(valid) if valid else 0.0
    mean_v = sum(n["v"] for n in valid) / len(valid) if valid else 0.0
    path.write_text(
        "\n".join(
            [
                f"keypoints_reference={len(kp0)}",
                f"keypoints_deformed={len(kp1)}",
                f"ratio_threshold={args.ratio}",
                f"raw_matches={len(matches)}",
                f"ratio_passed={ratio_count}",
                f"mutual_matches={mutual_count}",
                f"robust_inliers={inlier_count}",
                f"interpolation_k={args.k}",
                f"interpolation_radius={args.radius}",
                f"nodes={len(nodes)}",
                f"node_valid={len(valid)}",
                f"node_invalid={len(nodes) - len(valid)}",
                f"mean_node_u={mean_u}",
                f"mean_node_v={mean_v}",
                f"max_node_magnitude={max_mag}",
                "",
            ]
        )
    )


def write_svg(path, matches, nodes, width, height):
    max_mag = max((math.hypot(n["u"], n["v"]) for n in nodes if n["valid"]), default=0.0)
    scale = 25.0 / max_mag if max_mag > 0 else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L7,3 z" fill="#1f77b4"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" stroke="#cccccc"/>',
        '<g stroke="#999999" stroke-width="0.6" opacity="0.35">',
    ]
    for m in matches:
        if m["robust_inlier"]:
            lines.append(f'<line x1="{m["x0"]}" y1="{m["y0"]}" x2="{m["x1"]}" y2="{m["y1"]}"/>')
    lines.append('</g><g stroke="#1f77b4" stroke-width="1.2" marker-end="url(#arrow)">')
    for n in nodes:
        if n["valid"]:
            lines.append(f'<line x1="{n["x"]}" y1="{n["y"]}" x2="{n["x"] + n["u"] * scale}" y2="{n["y"] + n["v"] * scale}"/>')
    lines.append('</g><g>')
    for n in nodes:
        color = "#2ca02c" if n["valid"] else "#d62728"
        radius = 2.0 if n["valid"] else 3.0
        lines.append(f'<circle cx="{n["x"]}" cy="{n["y"]}" r="{radius}" fill="{color}"/>')
    lines.append('</g><g fill="#ff7f0e" opacity="0.75">')
    for m in matches:
        if m["robust_inlier"]:
            lines.append(f'<circle cx="{m["x0"]}" cy="{m["y0"]}" r="2"/>')
    lines.append("</g></svg>")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("deformed")
    parser.add_argument("nodes")
    parser.add_argument("out_dir")
    parser.add_argument("label")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--radius", type=float, default=180.0)
    parser.add_argument("--mad-factor", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    deform = cv2.imread(args.deformed, cv2.IMREAD_GRAYSCALE)
    if ref is None or deform is None:
        raise RuntimeError("Failed to load input images.")
    if ref.shape != deform.shape:
        raise RuntimeError("Reference and deformed images must have matching shape.")

    node_records = read_nodes(Path(args.nodes))
    kp0, kp1, matches = compute_sift_matches(ref, deform, args.max_features, args.ratio, args.mad_factor)
    node_init = interpolate_nodes(node_records, matches, max(1, args.k), max(1.0, args.radius))

    write_csv(
        out_dir / f"{args.label}_sift_matches.csv",
        matches,
        ["x0", "y0", "x1", "y1", "u", "v", "descriptor_distance", "ratio_passed", "mutual", "robust_inlier"],
    )
    write_csv(
        out_dir / f"{args.label}_sift_node_initialization.csv",
        node_init,
        ["node", "x", "y", "valid", "u", "v", "support_count", "nearest_match_distance"],
    )
    write_summary(out_dir / f"{args.label}_sift_summary.txt", kp0, kp1, matches, node_init, args)
    write_svg(out_dir / f"{args.label}_sift_node_initialization.svg", matches, node_init, ref.shape[1], ref.shape[0])
    print(f"Wrote SIFT node initialization diagnostics to {out_dir}")
    print(f"keypoints_reference={len(kp0)}")
    print(f"keypoints_deformed={len(kp1)}")
    print(f"matches={len(matches)}")


if __name__ == "__main__":
    main()
