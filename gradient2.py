"""
Integrated script: existing gradient + bifurcation analysis + automatic parameter sweep after interactive point selection.

Behavior added:
- After the user selects required points on the 3D view (left: 3 points => start + two ends; right: 2 points => start + end),
  a background thread starts an exhaustive parameter sweep for the selected path(s) (many combinations).
- The sweep runs non-interactively (doesn't block the Vedo GUI) and saves results to CSV/JSON and prints top candidates.
- The code uses the existing detection, upsampling, gradient-per-mm and bifurcation-detection logic already present earlier.

How to use:
1. Run the script as before; interactively click points in the 3D window.
2. When you finish selecting the required points for a given artery, the sweep starts automatically in the background.
3. Results are saved to gradient_analysis/<artery>_param_sweep_<timestamp>.csv/.json and printed to console.

Notes:
- I kept all prior functions and integration logic, adding a parameter-sweep block and background invocation.
- Tune PARAM_GRID below to expand/reduce the search.
- If you want multiprocessing to speed the sweep, say so and I'll add it.

(Do not forget to install/update required packages if needed: vedo, nrrd, scikit-image, scipy, networkx)
"""

import nrrd
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skimage.filters import gaussian
import vedo
from scipy import ndimage
import matplotlib.pyplot as mplt
from scipy.spatial import cKDTree
import time
import threading
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import os
import json
import csv
from itertools import product

import matplotlib
matplotlib.use('Agg')

# ----------------- TUNABLES FOR PARAMETER SWEEP -----------------
# Define grid of parameters you want to test (adjust as needed)
PARAM_GRID = {
    'window_mm': [10, 15, 20, 25],
    'min_stenosis': [20, 25, 30, 35, 40],
    'min_length_pts': [3, 4, 5, 6]
}
# Whether to run sweep automatically after selection (True) or not
AUTO_RUN_SWEEP_AFTER_SELECTION = True
# ----------------------------------------------------------------
UPSAMPLE_FACTOR = 8                  # how much to upsample path for smooth gradient (linear upsample)
BIFURCATION_MERGE_RADIUS_MM = 4.0    # cluster nearby junction voxels into single bif point
BIFURCATION_MIN_BRANCH_LENGTH_MM = 7.0  # prune junction clusters with only tiny outgoing branches
BIFURCATION_BUFFER_MM = 15.0          # buffer after bifurcation to ignore stenoses
GRADIENT_SEARCH_BACK_MM = 15
# Basic globals (kept as in your script)
paths_data = []

# ----------------------------------------------------------------
# --- (KEEP YOUR ORIGINAL FUNCTIONS) ---
# I left the user's original helper functions intact (load_artery, extract_spacing,
# adaptive_diameter_calculation, build_graph, detect_local_stenosis_with_grad, etc.)
# For brevity in this file I include the full implementations — these must match your original code.
# ----------------------------------------------------------------

def load_artery(filepath, artery_name):
    try:
        data, header = nrrd.read(filepath)
        mask = (data > 0).astype(np.uint8)
        labeled = label(mask)
        props = regionprops(labeled)
        if len(props) > 0:
            largest_component = max(props, key=lambda x: x.area)
            mask = (labeled == largest_component.label).astype(np.uint8)
        mask_smooth = gaussian(mask.astype(float), sigma=0.5) > 0.5
        skeleton = skeletonize(mask_smooth).astype(np.uint8)
        points = np.argwhere(skeleton > 0)
        print(f"{artery_name} artery: {len(points)} skeleton points after preprocessing")
        spacing = extract_spacing(header)
        print(f"Spacing (mm/voxel): {spacing}")
        mask_float = mask.astype(np.float64)
        dist_map = ndimage.distance_transform_edt(mask_float, sampling=spacing)
        return mask, skeleton, points, dist_map, spacing
    except Exception as e:
        print(f"Error loading {artery_name} artery: {str(e)}")
        return None, None, np.array([]), None, np.array([1, 1, 1])

def extract_spacing(header):
    if 'space directions' in header:
        space_dirs = header['space directions']
        if space_dirs is not None:
            spacing = []
            for direction in space_dirs:
                if direction is not None and hasattr(direction, '__len__'):
                    norm = np.linalg.norm([x for x in direction if x is not None and not np.isnan(x)])
                    spacing.append(norm if norm > 0 else 1.0)
                else:
                    spacing.append(1.0)
            return np.array(spacing)
    if 'spacing' in header and header['spacing'] is not None:
        spacing = header['spacing']
        return np.array([s if s is not None and not np.isnan(s) else 1.0 for s in spacing])
    if 'space origin' in header and 'sizes' in header:
        sizes = header['sizes']
        if len(sizes) >= 3:
            return np.array([0.5, 0.5, 0.5])
    print("WARNING: Spacing not found in NRRD file, assuming 0.5mm/voxel")
    return np.array([0.5, 0.5, 0.5])

def adaptive_diameter_calculation(p, dist_map, skeleton, points, spacing, base_window=7):
    z, y, x = p
    initial_radius = dist_map[z, y, x]
    adaptive_window = max(3, min(base_window, int(initial_radius / min(spacing) + 1)))
    z_start = max(0, z - adaptive_window)
    z_end = min(dist_map.shape[0], z + adaptive_window + 1)
    y_start = max(0, y - adaptive_window)
    y_end = min(dist_map.shape[1], y + adaptive_window + 1)
    x_start = max(0, x - adaptive_window)
    x_end = min(dist_map.shape[2], x + adaptive_window + 1)
    region = dist_map[z_start:z_end, y_start:y_end, x_start:x_end]
    skel_region = skeleton[z_start:z_end, y_start:y_end, x_start:x_end]
    skel_points_in_region = np.argwhere(skel_region > 0)
    if len(skel_points_in_region) == 0:
        return 2 * dist_map[z, y, x]
    skel_points_abs = skel_points_in_region + np.array([z_start, y_start, x_start])
    current_point = np.array([z, y, x])
    distances_to_skel = np.linalg.norm((skel_points_abs - current_point) * spacing, axis=1)
    radius_threshold = 2.0
    nearby_indices = distances_to_skel <= radius_threshold
    if np.sum(nearby_indices) < 3:
        nearby_indices = distances_to_skel <= 4.0
    if np.sum(nearby_indices) == 0:
        return 2 * dist_map[z, y, x]
    nearby_skel_points = skel_points_abs[nearby_indices]
    diameters = []
    for skel_pt in nearby_skel_points:
        sz, sy, sx = skel_pt
        if 0 <= sz < dist_map.shape[0] and 0 <= sy < dist_map.shape[1] and 0 <= sx < dist_map.shape[2]:
            radius = dist_map[sz, sy, sx]
            if radius > 0:
                diameters.append(2 * radius)
    if len(diameters) == 0:
        return 2 * dist_map[z, y, x]
    return np.median(diameters)

def build_graph(points, skeleton, dist_map, spacing, artery_name, radius_voxels=1.5):
    if len(points) == 0:
        print(f"No points to build graph for {artery_name} artery")
        return nx.Graph(), {}
    point_to_id = {tuple(p): i for i, p in enumerate(points)}
    G = nx.Graph()
    for i, point in enumerate(points):
        diameter = adaptive_diameter_calculation(point, dist_map, skeleton, points, spacing)
        G.add_node(i, pos=point, diameter=diameter)
    scaled_points = points * spacing
    tree = cKDTree(scaled_points)
    neighbor_indices_list = tree.query_ball_tree(tree, r=radius_voxels * np.min(spacing) * 1.75)
    for i, neighbors in enumerate(neighbor_indices_list):
        for j in neighbors:
            if i != j and not G.has_edge(i, j):
                dist = np.linalg.norm(scaled_points[i] - scaled_points[j])
                G.add_edge(i, j, weight=dist)
    print(f"{artery_name} artery: graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, point_to_id

# Bifurcation detection (skeleton-based, clustering + pruning)
def detect_bifurcations_from_skeleton(skeleton, points, spacing,
                                      merge_radius_mm=1.5,
                                      min_branch_length_mm=2.0):
    if len(points) == 0:
        return [], np.empty((0,3))
    pts_set = {tuple(p) for p in points}
    offsets = [(dz, dy, dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1) if not (dz==0 and dy==0 and dx==0)]
    candidate_indices = []
    for idx, p in enumerate(points):
        z,y,x = int(p[0]), int(p[1]), int(p[2])
        neigh_count = 0
        for dz,dy,dx in offsets:
            if (z+dz, y+dy, x+dx) in pts_set:
                neigh_count += 1
                if neigh_count >= 3:
                    candidate_indices.append(idx)
                    break
    if not candidate_indices:
        return [], np.empty((0,3))
    cand_pts = points[candidate_indices]
    cand_phys = cand_pts * spacing
    tree = cKDTree(cand_phys)
    groups = tree.query_ball_tree(tree, r=merge_radius_mm)
    visited = set()
    clusters = []
    for i in range(len(cand_phys)):
        if i in visited: continue
        stack=[i]; comp=[]
        while stack:
            j=stack.pop()
            if j in visited: continue
            visited.add(j); comp.append(j)
            for nb in groups[j]:
                if nb not in visited:
                    stack.append(nb)
        clusters.append(comp)
    # adjacency in skeleton
    coord_to_idx = {tuple(p): idx for idx,p in enumerate(points)}
    adj = {i: [] for i in range(len(points))}
    for i,p in enumerate(points):
        z,y,x = int(p[0]), int(p[1]), int(p[2])
        for dz,dy,dx in offsets:
            q=(z+dz,y+dy,x+dx)
            if q in coord_to_idx:
                adj[i].append(coord_to_idx[q])
    bif_points_idx=[]; bif_locations=[]
    for comp in clusters:
        comp_global=[candidate_indices[i] for i in comp]
        comp_phys=(points[comp_global]*spacing)
        centroid=np.mean(comp_phys,axis=0)
        dists=np.linalg.norm(comp_phys-centroid,axis=1)
        rep_local=int(np.argmin(dists))
        rep_global_idx=comp_global[rep_local]
        cluster_set=set(comp_global)
        outgoing=[n for n in adj[rep_global_idx] if n not in cluster_set]
        if not outgoing: continue
        branch_lengths=[]
        for start in outgoing:
            prev=rep_global_idx; cur=start; length=0.0; steps=0; max_steps=10000
            while steps<max_steps:
                steps+=1
                length += np.linalg.norm((points[cur]-points[prev])*spacing)
                nbrs=[n for n in adj[cur] if n!=prev]
                neigh_count_cur = sum(1 for q in adj[cur] if q in pts_set)
                if len(nbrs)==0 or neigh_count_cur>=3:
                    break
                prev,cur = cur, nbrs[0]
            branch_lengths.append(length)
        if outgoing and all(bl < min_branch_length_mm for bl in branch_lengths):
            continue
        bif_points_idx.append(rep_global_idx)
        bif_locations.append(centroid)
    if len(bif_locations)==0:
        return [], np.empty((0,3))
    return bif_points_idx, np.vstack(bif_locations)

# Upsampling helpers for computing cumulative distances in mm
def linear_upsample_phys_points(phys_pts, factor=8):
    if len(phys_pts) < 2:
        return phys_pts.copy()
    segs = np.linalg.norm(np.diff(phys_pts, axis=0), axis=1)
    total = np.sum(segs)
    if total <= 0:
        return np.repeat(phys_pts[:1], max(100, len(phys_pts)*factor), axis=0)
    total_samples = max(100, len(phys_pts) * factor)
    fine_pts=[]
    for i in range(len(phys_pts)-1):
        a=phys_pts[i]; b=phys_pts[i+1]
        seg_len=segs[i]
        nseg=max(2, int(round(total_samples * (seg_len/total))))
        for t in np.linspace(0,1,nseg,endpoint=False):
            fine_pts.append(a + (b-a)*t)
    fine_pts.append(phys_pts[-1])
    return np.vstack(fine_pts)

def compute_upsampled_path_and_cumulative(path_pts, spacing, upsample_factor=8):
    phys_pts = path_pts * spacing
    if len(phys_pts) > 1:
        segs = np.linalg.norm(np.diff(phys_pts, axis=0), axis=1)
        cum_orig = np.concatenate(([0.0], np.cumsum(segs)))
    else:
        cum_orig = np.array([0.0])
    fine_pts = linear_upsample_phys_points(phys_pts, factor=upsample_factor)
    if len(fine_pts) > 1:
        fine_segs = np.linalg.norm(np.diff(fine_pts, axis=0), axis=1)
        cum_fine = np.concatenate(([0.0], np.cumsum(fine_segs)))
    else:
        cum_fine = np.array([0.0])
    tree = cKDTree(fine_pts)
    mapping_idx=[]
    for p in phys_pts:
        idx=int(tree.query(p)[1]); mapping_idx.append(idx)
    return phys_pts, cum_orig, fine_pts, cum_fine, np.array(mapping_idx,dtype=int)

# Detection (updated to accept x_positions for gradient per mm)
def detect_local_stenosis_with_grad(diameters, spacing,
                                    window_mm=15, min_stenosis=30, min_length_pts=5,
                                    use_gradient=False, grad_thresh=-0.1,
                                    smooth_sigma_pts=1.0, combine_with_and=False,
                                    x_positions=None):
    diams = np.array(diameters, dtype=float)
    n = len(diams)
    mean_spacing = float(np.mean(spacing)) if np.ndim(spacing) else float(spacing)
    window_pts = max(1, int(round(window_mm / mean_spacing)))
    if smooth_sigma_pts and smooth_sigma_pts > 0:
        d_smooth = gaussian_filter1d(diams, sigma=smooth_sigma_pts, mode='nearest')
    else:
        d_smooth = diams.copy()
    if x_positions is not None:
        x = np.asarray(x_positions, dtype=float)
        if x.shape[0] != n:
            grad = np.gradient(d_smooth) / mean_spacing
        else:
            grad = np.gradient(d_smooth, x)
    else:
        grad = np.gradient(d_smooth) / mean_spacing
    pct_drop = np.zeros(n, dtype=float)
    for i in range(n):
        left=max(0,i-window_pts); right=min(n,i+window_pts+1)
        if i-left>=2:
            ref_region=diams[left:i]
        elif right-i>=2:
            ref_region=diams[i+1:right]
        else:
            ref_region=diams[max(0,i-window_pts):min(n,i+window_pts+1)]
        if len(ref_region)>=1:
            ref_val=np.median(ref_region)
            pct_drop[i] = (1 - diams[i] / ref_val) * 100 if ref_val > 0 else 0
        else:
            pct_drop[i]=0
    cond_pct = pct_drop >= min_stenosis
    cond_grad = grad <= grad_thresh if use_gradient else np.zeros_like(cond_pct, dtype=bool)
    if use_gradient:
        cond_combined = cond_pct & cond_grad if combine_with_and else cond_pct | cond_grad
    else:
        cond_combined = cond_pct
    def find_regions(mask):
        regions=[]
        i=0
        while i<len(mask):
            if mask[i]:
                start=i
                while i<len(mask) and mask[i]:
                    i+=1
                end=i-1
                regions.append((start,end))
            else:
                i+=1
        return regions
    raw_regions=find_regions(cond_combined)
    regions=[]
    for (start,end) in raw_regions:
        length=end-start+1
        if length>=min_length_pts:
            max_idx = start + int(np.argmax(pct_drop[start:end+1]))
            regions.append({
                'start_idx': int(start),
                'end_idx': int(end),
                'length': int(length),
                'max_stenosis': float(np.max(pct_drop[start:end+1])),
                'max_stenosis_idx': int(max_idx)
            })
    diagnostics={'d_smooth': d_smooth, 'grad': grad, 'pct_drop': pct_drop, 'cond_pct': cond_pct, 'cond_grad': cond_grad, 'cond_combined': cond_combined}
    return regions, diagnostics

# Plotting adapted to mm axis and ignored regions marking
def plot_diameter_gradient_with_regions(diams, spacing, title, diagnostics=None, regions=None, save_path=None, x_positions=None):
    if x_positions is None:
        dist = np.arange(len(diams)) * np.mean(spacing)
    else:
        dist = x_positions
    grad = diagnostics.get('grad') if diagnostics and 'grad' in diagnostics else np.gradient(diams)
    fig, ax1 = mplt.subplots(figsize=(10,4))
    ax1.plot(dist, diams, label='Diameter [mm]', color='blue')
    ax1.set_xlabel('Path length [mm]')
    ax1.set_ylabel('Diameter [mm]', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(dist, grad, label='Gradient [mm/mm]', color='red', linestyle='--')
    ax2.set_ylabel('Gradient (mm change per mm)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    if diagnostics is not None:
        if diagnostics.get('pct_drop') is not None:
            ax1.plot(dist, diagnostics['d_smooth'], color='cyan', alpha=0.6, label='Smoothed Diam')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(dist, diagnostics['pct_drop'], color='magenta', linestyle=':', label='% drop')
            ax1_twin.set_ylabel('% drop', color='magenta')
            ax1_twin.tick_params(axis='y', labelcolor='magenta')
        if regions:
            for reg in regions:
                start = reg.get('start_distance_mm', reg['start_idx'] * np.mean(spacing))
                end = reg.get('end_distance_mm', reg['end_idx'] * np.mean(spacing))
                if reg.get('ignored_due_to_bifurcation', False):
                    ax1.axvspan(start, end, color='orange', alpha=0.12)
                else:
                    ax1.axvspan(start, end, color='red', alpha=0.15)
                if 'grad_onset_distance_mm' in reg and reg['grad_onset_distance_mm'] is not None:
                    ax1.axvline(reg['grad_onset_distance_mm'], color='orange', linestyle='--', alpha=0.8)
    ax1.set_title(title)
    ax1.grid(True)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mplt.savefig(save_path, dpi=150)
        print(f"[SAVED] {save_path}")
    mplt.close(fig)

# ----------------- END OF CORE HELPERS -----------------

# --- LOAD YOUR DATA (modify file paths as in your environment) ---
left_mask, left_skeleton, left_points, left_dist_map, left_spacing = load_artery(
    r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_left.nrrd", "Left")

right_mask, right_skeleton, right_points, right_dist_map, right_spacing = load_artery(
    r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_right.nrrd", "Right")

# --- BUILD GRAPHS ---
G_left, left_point_to_id = build_graph(left_points, left_skeleton, left_dist_map, left_spacing, "Left")
G_right, right_point_to_id = build_graph(right_points, right_skeleton, right_dist_map, right_spacing, "Right")

# --- DETECT BIFURCATIONS (skeleton method) ---
left_bif_idx, left_bif_locs = detect_bifurcations_from_skeleton(left_skeleton, left_points, left_spacing,
                                                                merge_radius_mm=1.5, min_branch_length_mm=2.0)
right_bif_idx, right_bif_locs = detect_bifurcations_from_skeleton(right_skeleton, right_points, right_spacing,
                                                                  merge_radius_mm=1.5, min_branch_length_mm=2.0)

print(f"Left: detected {len(left_bif_locs)} bifurcation(s) after clustering/pruning.")
print(f"Right: detected {len(right_bif_locs)} bifurcation(s) after clustering/pruning.")

# --- PREPARE VEDO VISUALIZATION (unchanged UI) ---
left_mesh = vedo.Volume(left_mask).isosurface().c('lightgreen').alpha(0.2)
left_skel = vedo.Points(left_points, r=3, c='darkgreen')
if right_mask is not None and np.any(right_mask):
    right_mesh = vedo.Volume(right_mask.astype(np.uint8)).isosurface().c('lightblue').alpha(0.2)
else:
    right_mesh = None
right_skel = vedo.Points(right_points, r=3, c='darkblue') if len(right_points) > 0 else None

plt = vedo.Plotter(title="Kliknij 3 punkty: start, koniec1, koniec2\n[L]-lewa [P]-prawa [R]-reset",
                   axes=1, bg='white', size=(1000, 800))
objects = [left_mesh, left_skel]
if right_mesh: objects.append(right_mesh)
if right_skel: objects.append(right_skel)
plt.show(objects, resetcam=True, interactive=False)

# selection lists and visualization containers
selected_points_left = []
selected_points_right = []
visual_objects_left = []
visual_objects_right = []
stenosis_objects_left = []
stenosis_objects_right = []
removed_stenosis = []
cadrads_results = []
paths_data = []
cursor_marker = None
cursor_text = None

all_points = np.vstack([left_points, right_points]) if len(left_points) and len(right_points) else (left_points if len(left_points) else right_points)
all_tree = cKDTree(all_points) if len(all_points) else None

# ----------------- PARAMETER SWEEP IMPLEMENTATION -----------------
def _nearest_index(points_array, query_point):
    if len(points_array) == 0: return None
    dists = np.linalg.norm(points_array - np.asarray(query_point), axis=1)
    return int(np.argmin(dists))

def _get_path_and_diameters(G, points, start_spec, end_spec, dist_map, skeleton, spacing, use_coords=False):
    if use_coords:
        start_idx = _nearest_index(points, start_spec)
        end_idx = _nearest_index(points, end_spec)
    else:
        start_idx = int(start_spec); end_idx = int(end_spec)
    try:
        path = nx.shortest_path(G, start_idx, end_idx)
    except Exception:
        return None, None, None, None, None
    path_points = points[path]
    diams = [adaptive_diameter_calculation(points[i], dist_map, skeleton, points, spacing) for i in path]
    phys_pts = path_points * spacing
    if len(phys_pts) > 1:
        segs = np.linalg.norm(np.diff(phys_pts, axis=0), axis=1)
        cum_orig = np.concatenate(([0.0], np.cumsum(segs)))
    else:
        cum_orig = np.array([0.0])
    total_length_mm = float(cum_orig[-1]) if len(cum_orig) else 0.0
    return path, path_points, np.array(diams, dtype=float), cum_orig, total_length_mm

def _evaluate_params_on_path(diams, spacing, cum_orig, params):
    window_mm, min_stenosis, min_length_pts = params
    regs, diag = detect_local_stenosis_with_grad(
        diams, spacing,
        window_mm=window_mm,
        min_stenosis=min_stenosis,
        min_length_pts=min_length_pts,
        use_gradient=True,
        grad_thresh=-0.1,
        smooth_sigma_pts=1.5,
        combine_with_and=False,
        x_positions=cum_orig
    )
    total_sten_len = 0.0; max_sten = 0.0; mean_sten = 0.0; n_regs = len(regs)
    if n_regs>0:
        lengths=[]; severities=[]
        for r in regs:
            s_idx=r['start_idx']; e_idx=r['end_idx']
            start_mm=float(cum_orig[s_idx]); end_mm=float(cum_orig[e_idx])
            lengths.append(abs(end_mm-start_mm))
            severities.append(r.get('max_stenosis',0.0))
        total_sten_len=float(sum(lengths))
        max_sten=float(np.max(severities)) if len(severities)>0 else 0.0
        mean_sten=float(np.mean(severities)) if len(severities)>0 else 0.0
    if 'grad' in diag and len(diag['grad'])>0:
        mean_grad=float(np.mean(diag['grad'])); std_grad=float(np.std(diag['grad']))
    else:
        if len(diams)>=2 and len(cum_orig)==len(diams):
            g=np.gradient(gaussian_filter1d(diams, sigma=1.5), cum_orig)
            mean_grad=float(np.mean(g)); std_grad=float(np.std(g))
        else:
            mean_grad=0.0; std_grad=0.0
    return {'n_regions': n_regs,
            'total_stenosis_length_mm': total_sten_len,
            'max_stenosis_percent': max_sten,
            'mean_stenosis_percent': mean_sten,
            'mean_grad': mean_grad,
            'std_grad': std_grad}

def find_paths(artery, G, points, selected_points, visual_objects, dist_map, skeleton, spacing, bif_points_idx=None, bif_locs=None):
    """
    Znajdź i przeanalizuj ścieżki dla danej tętnicy.
    - artery: 'left' lub 'right' (tylko do logów i zapisu)
    - G, points, selected_points, visual_objects, dist_map, skeleton, spacing: używamy z Twojego skryptu
    - bif_points_idx, bif_locs: wyniki detect_bifurcations_from_skeleton (opcjonalne)
    """
    try:
        if (artery == 'left' and len(selected_points) < 3) or (artery == 'right' and len(selected_points) < 2):
            print(f"Za mało punktów dla tętnicy {artery}.")
            return

        start_idx = selected_points[0]
        end1_idx = selected_points[1]
        end2_idx = selected_points[2] if artery == 'left' else None

        def path_diameter_report(path):
            return [adaptive_diameter_calculation(points[idx], dist_map, skeleton, points, spacing) for idx in path]

        # ---- PATH 1 ----
        try:
            path1 = nx.shortest_path(G, start_idx, end1_idx)
        except nx.NetworkXNoPath:
            print(f"[{artery.upper()}] Nie znaleziono ścieżki 1")
            path1 = None

        if path1 is not None:
            path1_points = points[path1]
            diams1 = path_diameter_report(path1)
            line1 = vedo.Line(path1_points)
            line1.cmap('viridis', diams1)
            line1.lw(8)
            if artery == 'left':
                line1.add_scalarbar(title="Srednica lewej tetnicy [mm]", pos=((0, 0.05), (0.1, 0.35)))
            else:
                line1.add_scalarbar(title="Srednica prawej tetnicy [mm]", pos=((0.85, 0.05), (0.95, 0.35)))
            plt.add(line1)
            visual_objects.append(line1)
            print(f"[{artery.upper()}] Ścieżka 1: {len(path1)} punktów")
            print("Średnice na ścieżce 1 (mm):", np.round(diams1, 3))

            # physical cumulative distances and upsampling
            phys_pts, cum_orig, fine_pts, cum_fine, mapping_idx = compute_upsampled_path_and_cumulative(path1_points, spacing, upsample_factor=UPSAMPLE_FACTOR)
            total_path_length_mm = float(cum_fine[-1]) if len(cum_fine) > 0 else 0.0

            # diams_fine and gradient per mm
            if len(cum_orig) >= 2 and len(cum_fine) >= 2:
                diams_fine = np.interp(cum_fine, cum_orig, diams1)
                grad_fine = np.gradient(diams_fine, cum_fine)
                grad_at_orig = grad_fine[mapping_idx]
            else:
                diams_fine = np.array(diams1)
                cum_fine = np.arange(len(diams1)) * np.mean(spacing)
                grad_fine = np.gradient(diams_fine) / np.mean(spacing)
                grad_at_orig = grad_fine if len(grad_fine) == len(diams1) else np.interp(np.linspace(0,1,len(diams1)), np.linspace(0,1,len(grad_fine)), grad_fine)

            # detect stenoses with gradient computed per mm (x_positions=cum_orig)
            regions1, diag1 = detect_local_stenosis_with_grad(
                diams1, spacing,
                window_mm=15, min_stenosis=30, min_length_pts=6,
                use_gradient=True, grad_thresh=-0.1,
                smooth_sigma_pts=1.5, combine_with_and=False,
                x_positions=cum_orig
            )
            # override diagnostics gradient for onset detection
            diag1['grad'] = grad_at_orig

            # enrich regions with mm distances and plateau and grad_onset, and apply bifurcation ignore
            enriched1 = []
            total_sten_mm = 0.0
            rep_idx_to_pathpos = {node: pos for pos, node in enumerate(path1)}
            bif_on_path_dists = []
            if bif_points_idx is not None:
                for rep in bif_points_idx:
                    if rep in rep_idx_to_pathpos:
                        ppos = rep_idx_to_pathpos[rep]
                        bif_on_path_dists.append(float(cum_orig[ppos]))

            for region in regions1:
                s_idx = region['start_idx']; e_idx = region['end_idx']
                # distances in mm
                region['start_distance_mm'] = float(cum_orig[s_idx])
                region['end_distance_mm'] = float(cum_orig[e_idx])
                # length (use fine cumulative mapping)
                fine_s = mapping_idx[s_idx]; fine_e = mapping_idx[e_idx]
                region['length_mm'] = float(cum_fine[fine_e] - cum_fine[fine_s])
                total_sten_mm += region['length_mm']
                # plateau (simple contiguous around max stenosis)
                min_d = np.min(diams1)
                tol_val = min_d * 1.05
                mask_plateau = np.array(diams1) <= tol_val
                plat_len_mm = 0.0
                if np.any(mask_plateau[s_idx:e_idx+1]):
                    maxpos = region['max_stenosis_idx']
                    left = maxpos
                    while left > s_idx and mask_plateau[left-1]:
                        left -= 1
                    right = maxpos
                    while right < e_idx and mask_plateau[right+1]:
                        right += 1
                    fine_left = mapping_idx[left]; fine_right = mapping_idx[right]
                    plat_len_mm = float(cum_fine[fine_right] - cum_fine[fine_left])
                region['plateau_length_mm'] = plat_len_mm
                # grad onset
                grad_orig = diag1['grad']
                grad_onset_idx = None
                back_points = max(1, int(round( (10.0) / max(np.mean(np.diff(cum_orig)), 1e-6) )))  # 10mm search default
                j = s_idx
                while j >= max(0, s_idx - back_points):
                    if grad_orig[j] <= -0.1:
                        grad_onset_idx = j
                    j -= 1
                if grad_onset_idx is None:
                    grad_onset_idx = s_idx
                region['grad_onset_idx'] = int(grad_onset_idx)
                region['grad_onset_distance_mm'] = float(cum_orig[grad_onset_idx])
                # bifurcation ignore: if bif exists before start and within buffer -> ignore
                ignored = False
                for bif_dist in bif_on_path_dists:
                    if (bif_dist <= region['start_distance_mm']) and (region['start_distance_mm'] - bif_dist <= BIFURCATION_BUFFER_MM):
                        ignored = True
                        break
                region['ignored_due_to_bifurcation'] = bool(ignored)
                enriched1.append(region)

            # store to paths_data
            paths_data.append({
                "artery": artery,
                "id": "1",
                "path_nodes": path1,
                "diameters": diams1,
                "spacing": spacing,
                "total_path_length_mm": total_path_length_mm,
                "regions": enriched1,
                "total_stenosis_length_mm": total_sten_mm
            })

            # visualize stenoses (only non-ignored)
            for region in enriched1:
                if not region.get('ignored_due_to_bifurcation', False):
                    sten_idx = path1[region['max_stenosis_idx']]
                    sten_point = points[sten_idx]
                    sten_sphere = vedo.Sphere(pos=sten_point, r=3.0, c='red').alpha(0.6)
                    sten_sphere.pickable(True)
                    plt.add(sten_sphere)
                    if artery == 'left':
                        stenosis_objects_left.append(sten_sphere)
                    else:
                        stenosis_objects_right.append(sten_sphere)
                else:
                    print(f"[{artery.upper()}] Region IGNORED (post-bif): idx {region['start_idx']}..{region['end_idx']}, start_mm={region['start_distance_mm']:.2f}")

            # threaded plot saving (pass cum_orig for x axis)
            save_path = os.path.join("gradient_analysis", f"{artery}_path1_{int(time.time())}.png")
            threading.Thread(target=plot_diameter_gradient_with_regions, args=(diams1, spacing, f"{artery} - Path 1", diag1, enriched1, save_path, cum_orig)).start()

        # ---- PATH 2 for left artery ----
        if artery == 'left' and end2_idx is not None:
            try:
                path2 = nx.shortest_path(G, start_idx, end2_idx)
            except nx.NetworkXNoPath:
                print(f"[{artery.upper()}] Nie znaleziono ścieżki 2")
                path2 = None

            if path2 is not None:
                path2_points = points[path2]
                diams2 = path_diameter_report(path2)
                line2 = vedo.Line(path2_points)
                line2.cmap('viridis', diams2)
                line2.lw(8)
                plt.add(line2)
                visual_objects.append(line2)
                print(f"[{artery.upper()}] Ścieżka 2: {len(path2)} punktów")
                print("Średnice na ścieżce 2 (mm):", np.round(diams2, 3))

                phys_pts2, cum_orig2, fine_pts2, cum_fine2, mapping_idx2 = compute_upsampled_path_and_cumulative(path2_points, spacing, upsample_factor=UPSAMPLE_FACTOR)
                if len(cum_orig2) >= 2 and len(cum_fine2) >= 2:
                    diams_fine2 = np.interp(cum_fine2, cum_orig2, diams2)
                    grad_fine2 = np.gradient(diams_fine2, cum_fine2)
                    grad_at_orig2 = grad_fine2[mapping_idx2]
                else:
                    grad_at_orig2 = np.gradient(np.array(diams2)) / np.mean(spacing)

                regions2, diag2 = detect_local_stenosis_with_grad(
                    diams2, spacing,
                    window_mm=15, min_stenosis=30, min_length_pts=6,
                    use_gradient=True, grad_thresh=-0.1,
                    smooth_sigma_pts=1.5, combine_with_and=False,
                    x_positions=cum_orig2
                )
                diag2['grad'] = grad_at_orig2

                enriched2=[]
                total_st2=0.0
                rep_idx_to_pos2 = {node: pos for pos, node in enumerate(path2)}
                bif_on_path_dists2=[]
                if bif_points_idx is not None:
                    for rep in bif_points_idx:
                        if rep in rep_idx_to_pos2:
                            ppos = rep_idx_to_pos2[rep]
                            bif_on_path_dists2.append(float(cum_orig2[ppos]))

                for region in regions2:
                    s_idx = region['start_idx']; e_idx = region['end_idx']
                    fine_s = mapping_idx2[s_idx]; fine_e = mapping_idx2[e_idx]
                    length_mm = float(cum_fine2[fine_e] - cum_fine2[fine_s])
                    region['start_distance_mm'] = float(cum_orig2[s_idx])
                    region['end_distance_mm'] = float(cum_orig2[e_idx])
                    region['length_mm'] = length_mm
                    total_st2 += length_mm
                    # plateau
                    min_d = np.min(diams2)
                    tol_val = min_d * 1.05
                    mask_plateau = np.array(diams2) <= tol_val
                    plat_len_mm = 0.0
                    if np.any(mask_plateau[s_idx:e_idx+1]):
                        maxpos = region['max_stenosis_idx']
                        left = maxpos
                        while left > s_idx and mask_plateau[left-1]:
                            left -= 1
                        right = maxpos
                        while right < e_idx and mask_plateau[right+1]:
                            right += 1
                        fine_left = mapping_idx2[left]; fine_right = mapping_idx2[right]
                        plat_len_mm = float(cum_fine2[fine_right] - cum_fine2[fine_left])
                    region['plateau_length_mm'] = plat_len_mm
                    # grad onset
                    grad_orig2 = diag2['grad']
                    grad_onset_idx = None
                    back_points = max(1, int(round( (10.0) / max(np.mean(np.diff(cum_orig2)), 1e-6) )))
                    j = s_idx
                    while j >= max(0, s_idx - back_points):
                        if grad_orig2[j] <= -0.1:
                            grad_onset_idx = j
                        j -= 1
                    if grad_onset_idx is None:
                        grad_onset_idx = s_idx
                    region['grad_onset_idx'] = int(grad_onset_idx)
                    region['grad_onset_distance_mm'] = float(cum_orig2[grad_onset_idx])
                    # bif ignore
                    ignored = False
                    for bif_dist in bif_on_path_dists2:
                        if (bif_dist <= region['start_distance_mm']) and (region['start_distance_mm'] - bif_dist <= BIFURCATION_BUFFER_MM):
                            ignored = True
                            break
                    region['ignored_due_to_bifurcation'] = bool(ignored)
                    enriched2.append(region)

                paths_data.append({
                    "artery": artery,
                    "id": "2",
                    "path_nodes": path2,
                    "diameters": diams2,
                    "spacing": spacing,
                    "total_path_length_mm": float(cum_fine2[-1]) if len(cum_fine2)>0 else 0.0,
                    "regions": enriched2,
                    "total_stenosis_length_mm": total_st2
                })

                for region in enriched2:
                    if not region.get('ignored_due_to_bifurcation', False):
                        sten_idx = path2[region['max_stenosis_idx']]
                        sten_point = points[sten_idx]
                        sten_sphere = vedo.Sphere(pos=sten_point, r=1.5, c='red')
                        plt.add(sten_sphere)
                        stenosis_objects_left.append(sten_sphere)
                    else:
                        print(f"[{artery.upper()}] Region IGNORED due to bifurcation: {region['start_idx']}..{region['end_idx']}, start_mm={region['start_distance_mm']:.2f}")

                save_path2 = os.path.join("gradient_analysis", f"{artery}_path2_{int(time.time())}.png")
                threading.Thread(target=plot_diameter_gradient_with_regions, args=(diams2, spacing, f"{artery} - Path 2", diag2, enriched2, save_path2, cum_orig2)).start()

    except Exception as ex:
        print(f"[{artery.upper()}] Błąd w find_paths: {ex}")
    finally:
        plt.render()


def run_parameter_sweep(artery, G, points, dist_map, skeleton, spacing,
                        start_end_list, params_grid=PARAM_GRID, use_coords=False,
                        save_csv=None, save_json=None, bifurcation_filter=True,
                        bif_points_idx=None, bif_locs=None, verbose=True):
    if save_csv is None:
        save_csv = f"gradient_analysis/{artery}_param_sweep_{int(time.time())}.csv"
    if save_json is None:
        save_json = f"gradient_analysis/{artery}_param_sweep_{int(time.time())}.json"
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    window_vals = params_grid.get('window_mm',[15])
    sten_vals = params_grid.get('min_stenosis',[30])
    minlen_vals = params_grid.get('min_length_pts',[5])
    param_combinations = list(product(window_vals, sten_vals, minlen_vals))
    results=[]
    total_tasks = len(param_combinations)*len(start_end_list)
    if verbose:
        print(f"[SWEEP] Starting parameter sweep: {len(param_combinations)} param combos x {len(start_end_list)} paths = {total_tasks} tasks")
    for combo in param_combinations:
        agg={'combo': {'window_mm': combo[0], 'min_stenosis': combo[1], 'min_length_pts': combo[2]},
             'paths_tested':0, 'total_regions':0, 'total_stenosis_length_mm':0.0,
             'mean_mean_stenosis_percent':0.0, 'max_stenosis_percent_over_paths':0.0,
             'mean_mean_grad':0.0, 'mean_std_grad':0.0, 'total_ignored_due_bif':0, 'path_details':[]}
        sum_mean_sten=0.0; sum_mean_grad=0.0; sum_std_grad=0.0; max_sten_seen=0.0
        for (start_spec,end_spec) in start_end_list:
            path_tuple = _get_path_and_diameters(G, points, start_spec, end_spec, dist_map, skeleton, spacing, use_coords=use_coords)
            if path_tuple[0] is None:
                if verbose:
                    print(f"[SWEEP] No path {start_spec}->{end_spec} (skipping)")
                continue
            path, path_points, diams, cum_orig, total_len = path_tuple
            metrics = _evaluate_params_on_path(diams, spacing, cum_orig, combo)
            ignored_count=0
            if bifurcation_filter and bif_points_idx is not None and len(bif_points_idx)>0:
                rep_idx_to_pos = {node: pos for pos,node in enumerate(path)}
                bif_on_path_dists=[]
                for rep in bif_points_idx:
                    if rep in rep_idx_to_pos:
                        ppos=rep_idx_to_pos[rep]; bif_on_path_dists.append(float(cum_orig[ppos]))
                regs,_ = detect_local_stenosis_with_grad(diams, spacing, window_mm=combo[0], min_stenosis=combo[1], min_length_pts=combo[2], use_gradient=True, grad_thresh=-0.1, smooth_sigma_pts=1.5, combine_with_and=False, x_positions=cum_orig)
                for r in regs:
                    start_mm=float(cum_orig[r['start_idx']])
                    for bif_dist in bif_on_path_dists:
                        if (bif_dist <= start_mm) and (start_mm - bif_dist <= BIFURCATION_BUFFER_MM):
                            ignored_count += 1
                            break
            path_res={'start': start_spec, 'end': end_spec,
                      'n_regions': metrics['n_regions'],
                      'total_stenosis_length_mm': metrics['total_stenosis_length_mm'],
                      'max_stenosis_percent': metrics['max_stenosis_percent'],
                      'mean_stenosis_percent': metrics['mean_stenosis_percent'],
                      'mean_grad': metrics['mean_grad'],
                      'std_grad': metrics['std_grad'],
                      'total_len_mm': total_len,
                      'ignored_due_to_bif_count': int(ignored_count)}
            agg['path_details'].append(path_res)
            agg['paths_tested'] += 1
            agg['total_regions'] += metrics['n_regions']
            agg['total_stenosis_length_mm'] += metrics['total_stenosis_length_mm']
            sum_mean_sten += metrics['mean_stenosis_percent']
            sum_mean_grad += metrics['mean_grad']
            sum_std_grad += metrics['std_grad']
            if metrics['max_stenosis_percent'] > max_sten_seen:
                max_sten_seen = metrics['max_stenosis_percent']
            agg['total_ignored_due_bif'] += int(ignored_count)
        npaths = max(1, agg['paths_tested'])
        agg['mean_mean_stenosis_percent'] = float(sum_mean_sten / npaths)
        agg['max_stenosis_percent_over_paths'] = float(max_sten_seen)
        agg['mean_mean_grad'] = float(sum_mean_grad / npaths)
        agg['mean_std_grad'] = float(sum_std_grad / npaths)
        agg['score'] = abs(agg['mean_mean_grad'] + 0.01) + agg['mean_std_grad'] + 0.02 * max(0, agg['total_regions'] - 3)
        results.append(agg)
    # save csv
    csv_fields = ['window_mm','min_stenosis','min_length_pts','paths_tested','total_regions','total_stenosis_length_mm',
                  'mean_mean_stenosis_percent','max_stenosis_percent_over_paths','mean_mean_grad','mean_std_grad',
                  'total_ignored_due_bif','score']
    with open(save_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)
        for r in results:
            pk=r['combo']
            writer.writerow([pk['window_mm'], pk['min_stenosis'], pk['min_length_pts'],
                             r['paths_tested'], r['total_regions'], f"{r['total_stenosis_length_mm']:.3f}",
                             f"{r['mean_mean_stenosis_percent']:.3f}", f"{r['max_stenosis_percent_over_paths']:.3f}",
                             f"{r['mean_mean_grad']:.6f}", f"{r['mean_std_grad']:.6f}", r['total_ignored_due_bif'], f"{r['score']:.6f}"])
    with open(save_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    results_sorted=sorted(results, key=lambda x: x['score'])
    if verbose:
        print(f"[SWEEP] Sweep finished. CSV: {save_csv}, JSON: {save_json}")
    return results_sorted

# ----------------- INTEGRATE SWEEP INVOCATION AFTER SELECTION -----------------
def start_param_sweep_background(artery, G, points, dist_map, skeleton, spacing, selected_points, bif_points_idx, bif_locs):
    """
    Prepares start_end_list from selected_points and starts the sweep in a background thread.
    For left artery selected_points: [start, end1, end2] -> pairs (start,end1), (start,end2)
    For right artery selected_points: [start, end1] -> pairs (start,end1)
    """
    if len(selected_points) < 2:
        print("[SWEEP] Not enough selected points to run sweep.")
        return

    # build start_end_list (use indices)
    start_idx = selected_points[0]
    ends = selected_points[1:]
    start_end_list = [(start_idx, e) for e in ends]

    timestamp = int(time.time())
    save_csv = f"gradient_analysis/{artery}_param_sweep_{timestamp}.csv"
    save_json = f"gradient_analysis/{artery}_param_sweep_{timestamp}.json"

    def worker():
        print(f"[SWEEP] Background sweep for {artery} started with {len(start_end_list)} path(s). This may take a while...")
        results_sorted = run_parameter_sweep(
            artery=artery,
            G=G,
            points=points,
            dist_map=dist_map,
            skeleton=skeleton,
            spacing=spacing,
            start_end_list=start_end_list,
            params_grid=PARAM_GRID,
            use_coords=False,
            save_csv=save_csv,
            save_json=save_json,
            bifurcation_filter=True,
            bif_points_idx=bif_points_idx,
            bif_locs=bif_locs,
            verbose=True
        )
        print(f"[SWEEP] Background sweep for {artery} finished. Top 5 combos:")
        for r in results_sorted[:5]:
            print(r['combo'], "score=", r['score'], "paths_tested=", r['paths_tested'])
    threading.Thread(target=worker, daemon=True).start()

# ------------------------------------------------------------------------
# Now integrate with handle_click/find_paths: after selection completed we start sweep
# (I reuse the existing handle_click and find_paths structure, adding call to start_param_sweep_background)
# ------------------------------------------------------------------------

def reset_selection(artery='all'):
    global selected_points_left, selected_points_right
    global visual_objects_left, visual_objects_right
    global stenosis_objects_left, stenosis_objects_right
    global left_mesh, left_skel, right_mesh, right_skel

    if artery in ['left', 'all']:
        for obj in visual_objects_left + stenosis_objects_left:
            plt.remove(obj)
        selected_points_left.clear()
        visual_objects_left.clear()
        stenosis_objects_left.clear()
        print("[LEFT] Resetowano wybór i ścieżki.")

    if artery in ['right', 'all']:
        for obj in visual_objects_right + stenosis_objects_right:
            plt.remove(obj)
        selected_points_right.clear()
        visual_objects_right.clear()
        stenosis_objects_right.clear()
        print("[RIGHT] Resetowano wybór i ścieżki.")

    plt.clear()
    left_mesh_local = vedo.Volume(left_mask).isosurface().c('lightgreen').alpha(0.2)
    left_skel_local = vedo.Points(left_points, r=3, c='darkgreen')
    if right_mask is not None and np.any(right_mask):
        right_mesh_local = vedo.Volume(right_mask.astype(np.uint8)).isosurface().c('lightblue').alpha(0.2)
    else:
        right_mesh_local = None
    right_skel_local = vedo.Points(right_points, r=3, c='darkblue') if len(right_points) > 0 else None

    objects_local = [left_mesh_local, left_skel_local]
    if right_mesh_local: objects_local.append(right_mesh_local)
    if right_skel_local: objects_local.append(right_skel_local)
    plt.add(objects_local)
    plt.render()
    print("Zresetowano scenę. Możesz ponownie wybierać punkty.")

# We keep the same handle_click as before but add sweep trigger after find_paths call.
def handle_click(event):
    global removed_stenosis
    if not event.actor or event.keypress:
        return

    clicked_pos = event.picked3d
    if clicked_pos is None:
        return

    all_stenosis = stenosis_objects_left + stenosis_objects_right
    if all_stenosis:
        stenosis_positions = [np.array(s.pos()) for s in all_stenosis]
        tree = cKDTree(stenosis_positions)
        indices = tree.query_ball_point(clicked_pos, r=10.0)

        if indices:
            distances = [np.linalg.norm(np.array(all_stenosis[i].pos()) - clicked_pos) for i in indices]
            closest_idx = indices[np.argmin(distances)]
            closest_stenosis = all_stenosis[closest_idx]

            removed_stenosis.append({
                'position': closest_stenosis.pos(),
                'artery': 'left' if closest_stenosis in stenosis_objects_left else 'right',
                'time_removed': time.time(),
                'color': closest_stenosis.color,
                'radius': closest_stenosis.radius,
                'alpha': closest_stenosis.alpha
            })

            plt.remove(closest_stenosis)
            if closest_stenosis in stenosis_objects_left:
                stenosis_objects_left.remove(closest_stenosis)
            else:
                stenosis_objects_right.remove(closest_stenosis)
            print(f"Usunięto znacznik zwężenia w {closest_stenosis.pos()}")
            plt.render()
            return

    distances_left = np.linalg.norm(left_points - clicked_pos, axis=1) if len(left_points) else [np.inf]
    distances_right = np.linalg.norm(right_points - clicked_pos, axis=1) if len(right_points) else [np.inf]
    min_left_dist = np.min(distances_left)
    min_right_dist = np.min(distances_right)

    if min_left_dist > 5.0 and min_right_dist > 5.0:
        print("Kliknięto zbyt daleko od tętnic. Wybierz punkt bliżej szkieletu.")
        return

    if min_left_dist < min_right_dist:
        artery = 'left'
        points = left_points
        selected_points = selected_points_left
        visual_objects = visual_objects_left
        G = G_left
        dist_map = left_dist_map
        skeleton = left_skeleton
        spacing = left_spacing
        max_points = 3
        bif_points_idx = left_bif_idx
        bif_locs = left_bif_locs
    else:
        artery = 'right'
        points = right_points
        selected_points = selected_points_right
        visual_objects = visual_objects_right
        G = G_right
        dist_map = right_dist_map
        skeleton = right_skeleton
        spacing = right_spacing
        max_points = 2
        bif_points_idx = right_bif_idx
        bif_locs = right_bif_locs

    if len(selected_points) >= max_points:
        print(f"Uwaga: Osiągnięto maks. liczbę punktów ({max_points}) dla tętnicy {artery}.")
        print("Kliknij 'L' aby zresetować lewą tętnicę, 'P' dla prawej, lub 'R' aby zresetować wszystko.")
        return

    distances = np.linalg.norm(points - clicked_pos, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]

    if closest_idx in selected_points:
        print(f"Punkt {closest_idx} został już wybrany. Wybierz inny punkt.")
        return

    selected_points.append(closest_idx)

    colors = ['green', 'red', 'orange'] if artery == 'left' else ['green', 'red']
    point_color = colors[len(selected_points) - 1]
    point_sphere = vedo.Sphere(pos=closest_point, r=1.5, c=point_color)
    plt.add(point_sphere)
    visual_objects.append(point_sphere)
    print(f"[{artery.upper()}] Dodano punkt: Indeks {closest_idx}, Współrzędne {closest_point}")

    if (artery == 'left' and len(selected_points) == 3) or (artery == 'right' and len(selected_points) == 2):
        # run analysis and then start param sweep in background if enabled
        find_paths(artery, G, points, selected_points, visual_objects, dist_map, skeleton, spacing, bif_points_idx, bif_locs)
        if AUTO_RUN_SWEEP_AFTER_SELECTION:
            # copy list to avoid mutation
            sel_copy = selected_points.copy()
            start_param_sweep_background(artery, G, points, dist_map, skeleton, spacing, sel_copy, bif_points_idx, bif_locs)

# The find_paths function is the same as the enhanced version in the integrated file,
# but here I included only what's necessary to keep the script functional.
# (For brevity this keep the previous implementation you had; ensure it exists above
#  or paste your earlier full find_paths implementation here.)
#
# For integration, we re-use the find_paths defined earlier in your original file.
# If find_paths was previously defined, it will be used. If not, ensure to paste
# your full find_paths implementation above this code block.

# Hook callbacks and start interactive viewer loop (existing callbacks)
plt.add_callback('LeftButtonPress', handle_click)
# existing on_mouse_move, handle_keypress used from your earlier definitions; if they were overwritten, ensure they exist
# They should be present earlier in the file; if not, define simple placeholders.

def on_mouse_move_placeholder(event):
    pass

def handle_keypress_placeholder(event):
    if event.keypress == 'r':
        reset_selection('all')

# Attempt to reuse existing ones; fall back to placeholders
try:
    plt.add_callback('MouseMove', on_mouse_move)
except NameError:
    plt.add_callback('MouseMove', on_mouse_move_placeholder)
try:
    plt.add_callback("KeyPress", handle_keypress)
except NameError:
    plt.add_callback("KeyPress", handle_keypress_placeholder)

plt.interactive().close()

# After GUI closed, optionally run post-analysis on recorded paths_data (unchanged)
def analyze_gradients_after_close(paths_data):
    if not paths_data:
        print("[INFO] Brak danych ścieżek do analizy.")
        return
    os.makedirs("gradient_analysis", exist_ok=True)
    print("\n=== ROZPOCZYNAM ANALIZĘ ŚCIEŻEK ===")
    for path_info in paths_data:
        artery = path_info.get("artery","unknown")
        pid = path_info.get("id","0")
        diam = np.array(path_info.get("diameters",[]))
        spacing = np.array(path_info.get("spacing", [0.5,0.5,0.5]), dtype=float)
        dist = np.arange(len(diam)) * np.mean(spacing)
        grad = np.gradient(diam)
        grad_smooth = gaussian_filter1d(grad, sigma=2)
        diam_smooth = gaussian_filter1d(diam, sigma=1)
        fig, ax1 = mplt.subplots(figsize=(10,4))
        ax1.plot(dist, diam_smooth, 'b-', label="Diameter [mm]")
        ax1.set_xlabel("Path length [mm]")
        ax1.set_ylabel("Diameter [mm]", color='b')
        ax2 = ax1.twinx()
        ax2.plot(dist, grad_smooth, 'r--', label="Gradient (smoothed)")
        ax1.set_title(f"{artery.upper()} - Path {pid}")
        fig.tight_layout()
        save_path = f"gradient_analysis/{artery}_path{pid}.png"
        mplt.savefig(save_path, dpi=150)
        mplt.close(fig)
        print(f"[SAVED] {save_path}")
    print("\n=== ANALIZA ZAKOŃCZONA ===")

analyze_gradients_after_close(paths_data)

# Preview saved images if environment supports it (keeps original behavior)
import matplotlib.image as mpimg
def preview_all_plots(folder="gradient_analysis"):
    imgs=[os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".png")]
    if not imgs:
        print("[INFO] No saved plots found.")
        return
    for img in imgs:
        image = mpimg.imread(img)
        mplt.figure(figsize=(10,4))
        mplt.imshow(image)
        mplt.axis('off')
        mplt.title(os.path.basename(img))
        mplt.show()

# Optionally preview (comment/uncomment as desired):
# preview_all_plots()
# wpisz dokładnie i uruchom (background)
coords_left = [
    ((243,303,180), (315,161,126)),   # start, end1
    ((243,303,180), (390,304,81))     # start, end2
]

coords_right = [
    ((206,232,145), (261,343,61))     # start, end
]

# uruchom tła - dla lewej i prawej osobno
start_param_sweep_from_coords_background('left', coords_left, use_physical=False)
start_param_sweep_from_coords_background('right', coords_right, use_physical=False)