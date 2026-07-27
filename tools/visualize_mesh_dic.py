#!/usr/bin/env python
"""Visualize Mesh-DIC results: mesh overlay, displacement field, strain maps."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from PIL import Image
import sys, os

def load_csv(path):
    """Load a CSV with header row."""
    data = np.genfromtxt(path, delimiter=',', skip_header=1, names=True)
    return data

def main(result_dir, ref_bmp=None, def_bmp=None):
    U_path   = os.path.join(result_dir, 'U.csv')
    strain_path = os.path.join(result_dir, 'strain.csv')

    if not os.path.exists(U_path):
        print(f"Error: {U_path} not found")
        sys.exit(1)

    U_data = load_csv(U_path)
    x = U_data['x']; y = U_data['y']; u = U_data['u']; v = U_data['v']
    disp_mag = np.sqrt(u**2 + v**2)

    strain_data = load_csv(strain_path) if os.path.exists(strain_path) else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- 1. Reference image ---
    if ref_bmp and os.path.exists(ref_bmp):
        ref = np.array(Image.open(ref_bmp).convert('L'))
        axes[0, 0].imshow(ref, cmap='gray')
        axes[0, 0].set_title('Reference Image')
    axes[0, 0].scatter(x, y, c='red', s=0.5, alpha=0.3)
    axes[0, 0].set_xlabel('x [px]'); axes[0, 0].set_ylabel('y [px]')

    # --- 2. Deformed image with mesh overlay ---
    if def_bmp and os.path.exists(def_bmp):
        deformed = np.array(Image.open(def_bmp).convert('L'))
        axes[0, 1].imshow(deformed, cmap='gray')
    axes[0, 1].scatter(x + u, y + v, c='cyan', s=0.5, alpha=0.3)
    axes[0, 1].set_title('Deformed Nodes')
    axes[0, 1].set_xlabel('x [px]'); axes[0, 1].set_ylabel('y [px]')

    # --- 3. Displacement magnitude ---
    sc3 = axes[0, 2].scatter(x, y, c=disp_mag, s=2, cmap='jet')
    axes[0, 2].set_title(f'|U| [px]  max={disp_mag.max():.2f}')
    axes[0, 2].set_xlabel('x [px]'); axes[0, 2].set_ylabel('y [px]')
    axes[0, 2].invert_yaxis()
    plt.colorbar(sc3, ax=axes[0, 2])

    # --- 4. U displacement ---
    sc4 = axes[1, 0].scatter(x, y, c=u, s=2, cmap='RdBu_r',
                              vmin=-np.abs(u).max(), vmax=np.abs(u).max())
    axes[1, 0].set_title(f'U (x-displacement) [px]')
    axes[1, 0].set_xlabel('x [px]'); axes[1, 0].set_ylabel('y [px]')
    axes[1, 0].invert_yaxis()
    plt.colorbar(sc4, ax=axes[1, 0])

    # --- 5. V displacement ---
    sc5 = axes[1, 1].scatter(x, y, c=v, s=2, cmap='RdBu_r',
                              vmin=-np.abs(v).max(), vmax=np.abs(v).max())
    axes[1, 1].set_title(f'V (y-displacement) [px]')
    axes[1, 1].set_xlabel('x [px]'); axes[1, 1].set_ylabel('y [px]')
    axes[1, 1].invert_yaxis()
    plt.colorbar(sc5, ax=axes[1, 1])

    # --- 6. Strain Exx or displacement quiver ---
    if strain_data is not None:
        sc6 = axes[1, 2].scatter(strain_data['x'], strain_data['y'],
                                  c=strain_data['Exx'], s=2, cmap='RdBu_r')
        axes[1, 2].set_title(f'Exx strain  max={strain_data["Exx"].max():.4f}')
        axes[1, 2].invert_yaxis()
        plt.colorbar(sc6, ax=axes[1, 2])
    else:
        # Quiver plot (subsample for clarity)
        step = max(1, len(x) // 200)
        axes[1, 2].quiver(x[::step], y[::step], u[::step], v[::step],
                           disp_mag[::step], cmap='jet', scale=5, width=0.003)
        axes[1, 2].set_title('Displacement Vectors')
        axes[1, 2].invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(result_dir, 'mesh_dic_result.png')
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_mesh_dic.py <result_dir> [ref.bmp] [def.bmp]")
        sys.exit(1)
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else None)
