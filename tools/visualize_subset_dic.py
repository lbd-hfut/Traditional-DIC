#!/usr/bin/env python
"""Visualize Subset-DIC results in overview style (Reference/Deformed/ROI + U/V/Correlation).

Color style follows overview.png: grayscale images, jet displacement fields
(smooth contour fill), hot correlation map, white background outside ROI.

Usage:
    python visualize_subset_dic.py <displacements.csv> [ref.bmp] [def.bmp] [-o output.png]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from PIL import Image
import sys, os


def load_csv(path):
    """Load a CSV with header row."""
    return np.genfromtxt(path, delimiter=',', names=True)


def masked_triangulation(x, y, edge_factor=3.0):
    """Triangulate points and mask triangles that span gaps (ROI hole/boundary)."""
    triang = mtri.Triangulation(x, y)
    tri_pts_x = x[triang.triangles]
    tri_pts_y = y[triang.triangles]
    edges = np.stack([
        np.hypot(tri_pts_x[:, 0] - tri_pts_x[:, 1], tri_pts_y[:, 0] - tri_pts_y[:, 1]),
        np.hypot(tri_pts_x[:, 1] - tri_pts_x[:, 2], tri_pts_y[:, 1] - tri_pts_y[:, 2]),
        np.hypot(tri_pts_x[:, 2] - tri_pts_x[:, 0], tri_pts_y[:, 2] - tri_pts_y[:, 0]),
    ], axis=1)
    max_edge = edges.max(axis=1)
    triang.set_mask(max_edge > edge_factor * np.median(max_edge))
    return triang


def contour_fill(ax, triang, values, cmap, vmin, vmax, n_levels=64):
    """Smooth contour fill; masked triangles stay white."""
    levels = np.linspace(vmin, vmax, n_levels + 1)
    cf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, extend='both')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_facecolor('white')
    return cf


def main(csv_path, ref_bmp=None, def_bmp=None, out_path=None, ulim=None, vlim=None):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    data = load_csv(csv_path)
    names = data.dtype.names or ()

    # Keep only valid subset points
    if 'valid' in names:
        data = data[data['valid'] != 0]
    if len(data) == 0:
        print("Error: no valid subset points in CSV")
        sys.exit(1)

    x = data['x']; y = data['y']; u = data['u']; v = data['v']
    quality = data['quality'] if 'quality' in names else None

    print(f"Valid subset points: {len(x)}")
    triang = masked_triangulation(x, y)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- 1. Reference image ---
    if ref_bmp and os.path.exists(ref_bmp):
        ref = np.array(Image.open(ref_bmp).convert('L'))
        axes[0, 0].imshow(ref, cmap='gray')
    axes[0, 0].set_title('Reference')
    axes[0, 0].set_xlabel('x [px]'); axes[0, 0].set_ylabel('y [px]')

    # --- 2. Deformed image ---
    if def_bmp and os.path.exists(def_bmp):
        deformed = np.array(Image.open(def_bmp).convert('L'))
        axes[0, 1].imshow(deformed, cmap='gray')
    axes[0, 1].set_title('Deformed')
    axes[0, 1].set_xlabel('x [px]'); axes[0, 1].set_ylabel('y [px]')

    # --- 3. ROI mask (white = covered by valid subsets, black elsewhere) ---
    axes[0, 2].set_facecolor('black')
    axes[0, 2].tripcolor(triang, np.ones_like(x), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_aspect('equal')
    axes[0, 2].invert_yaxis()
    axes[0, 2].set_title('ROI')
    axes[0, 2].set_xlabel('x [px]'); axes[0, 2].set_ylabel('y [px]')

    # --- 4. U displacement (jet, symmetric limits from 99.9th percentile ---
    # to keep a few bad-match outliers from flattening the color scale) ---
    if ulim is None:
        ulim = float(np.percentile(np.abs(u), 99.9)) or 1.0
    cf4 = contour_fill(axes[1, 0], triang, u, 'jet', -ulim, ulim)
    axes[1, 0].set_title('U displacement')
    axes[1, 0].set_xlabel('x [px]'); axes[1, 0].set_ylabel('y [px]')
    plt.colorbar(cf4, ax=axes[1, 0])

    # --- 5. V displacement (jet, symmetric limits from 99.9th percentile) ---
    if vlim is None:
        vlim = float(np.percentile(np.abs(v), 99.9)) or 1.0
    cf5 = contour_fill(axes[1, 1], triang, v, 'jet', -vlim, vlim)
    axes[1, 1].set_title('V displacement')
    axes[1, 1].set_xlabel('x [px]'); axes[1, 1].set_ylabel('y [px]')
    plt.colorbar(cf5, ax=axes[1, 1])

    # --- 6. Correlation quality (hot, full range) ---
    if quality is not None:
        qmin = float(quality.min())
        qmax = float(quality.max())
        if qmax <= qmin:
            qmax = qmin + 1e-6
        cf6 = contour_fill(axes[1, 2], triang, quality, 'hot', qmin, qmax)
        axes[1, 2].set_facecolor('black')
        axes[1, 2].set_title('Correlation')
        axes[1, 2].set_xlabel('x [px]'); axes[1, 2].set_ylabel('y [px]')
        plt.colorbar(cf6, ax=axes[1, 2])
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                                'subset_dic_overview.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # --- 7. Save single-panel U displacement ---
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    _save_single_panel(x, y, u, triang, 'U displacement', 'jet', -ulim, ulim,
                       os.path.join(out_dir, 'subset_dic_u.png'),
                       ref_bmp, def_bmp)

    # --- 8. Save single-panel V displacement ---
    _save_single_panel(x, y, v, triang, 'V displacement', 'jet', -vlim, vlim,
                       os.path.join(out_dir, 'subset_dic_v.png'),
                       ref_bmp, def_bmp)

    # --- 9. Generate per-column CSV files ---
    _save_column_csv(csv_path, out_dir)


def _save_column_csv(csv_path, out_dir):
    """Generate per-column CSV files from the full displacements CSV.

    For each numeric column (u, v, quality, du_dx, du_dy, dv_dx, dv_dy, valid),
    writes a file like '<column>.csv' with header 'x,y,<column>' and all valid rows.
    """
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    names = data.dtype.names or ()
    # Determine which columns to export (skip index, matched_x, matched_y)
    export_cols = [c for c in names if c not in ('index', 'matched_x', 'matched_y')]
    x = data['x']
    y = data['y']
    for col in export_cols:
        out_file = os.path.join(out_dir, f'{col}.csv')
        values = data[col]
        with open(out_file, 'w') as f:
            f.write(f'x,y,{col}\n')
            for i in range(len(x)):
                f.write(f'{x[i]},{y[i]},{values[i]}\n')
        print(f"Saved: {out_file}")


def _save_single_panel(x, y, values, triang, title, cmap, vmin, vmax, out_path,
                       ref_bmp=None, def_bmp=None):
    """Save a single-panel displacement figure with optional reference background."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if ref_bmp and os.path.exists(ref_bmp):
        ref = np.array(Image.open(ref_bmp).convert('L'))
        ax.imshow(ref, cmap='gray', alpha=0.5)
    levels = np.linspace(vmin, vmax, 65)
    cf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, extend='both')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_facecolor('white')
    ax.set_title(title)
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    plt.colorbar(cf, ax=ax)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python visualize_subset_dic.py <displacements.csv> [ref.bmp] [def.bmp] [-o output.png] [--ulim val] [--vlim val]")
        sys.exit(1)
    out = None
    ulim = None
    vlim = None
    if '-o' in args:
        i = args.index('-o')
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    if '--ulim' in args:
        i = args.index('--ulim')
        ulim = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    if '--vlim' in args:
        i = args.index('--vlim')
        vlim = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    main(args[0],
         args[1] if len(args) > 1 else None,
         args[2] if len(args) > 2 else None,
         out, ulim, vlim)
