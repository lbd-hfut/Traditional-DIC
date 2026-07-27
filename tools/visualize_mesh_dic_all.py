#!/usr/bin/env python
"""Visualize T3/Q4/Q8 full-field displacement for ring case."""

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

def load_mesh_data(mesh_dir, etype):
    """Load nodes, elements, inform for a given element type."""
    import numpy as np
    nodes_file = os.path.join(mesh_dir, f'nodes_{etype}.txt')
    elems_file = os.path.join(mesh_dir, f'elements_{etype}.txt')
    inform_file = os.path.join(mesh_dir, f'Inform_{etype}.npy')
    
    # Nodes
    nodes_data = np.genfromtxt(nodes_file, delimiter=',')
    node_coords = nodes_data[:, 1:3]  # (n, 2)
    
    # Elements
    elems_data = np.genfromtxt(elems_file, delimiter=',', dtype=np.int32)
    elems = elems_data[:, 1:] - 1  # 0-based
    
    # Inform
    inform = np.load(inform_file)  # (n_pix, 3): [x, y, elem_id]
    
    return node_coords, elems, inform

def load_displacement(result_dir):
    """Load U.csv from solver output."""
    import numpy as np
    u_path = os.path.join(result_dir, 'U.csv')
    if not os.path.exists(u_path):
        return None
    data = np.genfromtxt(u_path, delimiter=',', skip_header=1)
    return data[:, 3:5]  # (n_nodes, 2) u, v

def interpolate_full_field(inform, elems, U, h, w, nn):
    """Interpolate nodal displacement to pixel field using element-average."""
    import numpy as np
    u_field = np.full(h * w, np.nan)
    v_field = np.full(h * w, np.nan)
    
    px = inform[:, 0].astype(np.int32)
    py = inform[:, 1].astype(np.int32)
    eid = inform[:, 2].astype(np.int32) - 1
    
    # Simple element-average interpolation
    conn_all = elems[eid][:, :nn]  # (n_pix, nn)
    u_corners = U[conn_all, 0]
    v_corners = U[conn_all, 1]
    u_avg = u_corners.mean(axis=1)
    v_avg = v_corners.mean(axis=1)
    
    idx = np.clip(py, 0, h-1) * w + np.clip(px, 0, w-1)
    u_field[idx] = u_avg
    v_field[idx] = v_avg
    
    return u_field.reshape(h, w), v_field.reshape(h, w)

def make_figure(ref_img, u_field, v_field, etype, out_path):
    """Create 2x3 figure for one element type."""
    import numpy as np
    mag = np.sqrt(u_field**2 + v_field**2)
    h, w = ref_img.shape[:2] if ref_img.ndim == 2 else ref_img.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{etype} Mesh-DIC — Ring Case (001→002)', fontsize=14, fontweight='bold')
    
    # 1. Ref + mesh outline
    axes[0,0].imshow(ref_img, cmap='gray')
    valid = ~np.isnan(mag)
    # Show mesh boundary by thresholding valid pixels
    from scipy import ndimage
    if valid.any():
        edges = ndimage.sobel(valid.astype(float))
        contour = np.abs(edges) > 0.1
        axes[0,0].contour(contour, colors='red', linewidths=0.5, levels=[0.5])
    axes[0,0].set_title('Reference Image + Mesh Boundary')
    
    # 2. Displacement magnitude
    sc2 = axes[0,1].imshow(mag, cmap='hot')
    axes[0,1].set_title(f'|U| [px]  max={np.nanmax(mag):.3f}')
    plt.colorbar(sc2, ax=axes[0,1])
    
    # 3. Displacement vectors
    axes[0,2].imshow(ref_img, cmap='gray')
    step = 30
    yy, xx = np.mgrid[step//2:h:step, step//2:w:step]
    uu = u_field[yy, xx]
    vv = v_field[yy, xx]
    mm = np.sqrt(uu**2 + vv**2)
    valid_v = ~np.isnan(uu)
    if valid_v.any():
        axes[0,2].quiver(xx[valid_v], yy[valid_v], uu[valid_v], vv[valid_v],
                         mm[valid_v], cmap='jet', scale=8, width=0.004)
    axes[0,2].set_title('Displacement Vectors')
    axes[0,2].invert_yaxis()
    
    # 4. U field
    vlim = max(abs(np.nanmin(u_field)), abs(np.nanmax(u_field)), 0.1)
    sc4 = axes[1,0].imshow(u_field, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    axes[1,0].set_title(f'U (x-displacement) [px]')
    plt.colorbar(sc4, ax=axes[1,0])
    
    # 5. V field
    vlim = max(abs(np.nanmin(v_field)), abs(np.nanmax(v_field)), 0.1)
    sc5 = axes[1,1].imshow(v_field, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    axes[1,1].set_title(f'V (y-displacement) [px]')
    plt.colorbar(sc5, ax=axes[1,1])
    
    # 6. Exx (finite difference)
    du_dx = np.gradient(u_field, axis=1)
    dv_dy = np.gradient(v_field, axis=0)
    exx = du_dx
    eyy = dv_dy
    # Use Exx for display
    vlim_e = 0.005
    sc6 = axes[1,2].imshow(exx, cmap='RdBu_r', vmin=-vlim_e, vmax=vlim_e)
    axes[1,2].set_title(f'Exx (dU/dx)')
    plt.colorbar(sc6, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')

def main():
    mesh_base = 'C:/02Project/Research/MeshDIC/case/ring/mesh'
    result_base = 'C:/02Project/Research/Traditional-DIC/case/2D/ring/mesh_dic_result'
    ref_path = 'C:/02Project/Research/MeshDIC/case/ring/001.bmp'
    
    ref = np.array(Image.open(ref_path).convert('L'))
    h, w = ref.shape
    nn_map = {'T3': 3, 'Q4': 4, 'Q8': 8}
    
    for etype in ['T3', 'Q4', 'Q8']:
        print(f'\n{"="*60}')
        print(f'Processing {etype}...')
        
        nodes, elems, inform = load_mesh_data(mesh_base, etype)
        print(f'  Nodes: {len(nodes)}, Elements: {len(elems)}, Inform: {len(inform)}')
        
        U = load_displacement(os.path.join(result_base, etype))
        if U is None:
            print(f'  WARNING: No U.csv for {etype}, skipping')
            continue
        
        print(f'  U range: [{U[:,0].min():.3f}, {U[:,0].max():.3f}] x [{U[:,1].min():.3f}, {U[:,1].max():.3f}]')
        
        nn = nn_map[etype]
        u_field, v_field = interpolate_full_field(inform, elems, U, h, w, nn)
        
        out_path = os.path.join(result_base, etype, f'mesh_dic_{etype}_fullfield.png')
        make_figure(ref, u_field, v_field, etype, out_path)

if __name__ == '__main__':
    main()
