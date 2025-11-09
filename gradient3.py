
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
from itertools import groupby

import matplotlib
matplotlib.use('Agg')  # wymusza osobne okno Matplotlib, niezależne od Vedo

# ----------------- NEW CONFIG (tune these) -----------------
UPSAMPLE_FACTOR = 8                  # how much to upsample path for smooth gradient (linear upsample)
BIFURCATION_MERGE_RADIUS_MM = 5.0    # cluster nearby junction voxels into single bif point
BIFURCATION_MIN_BRANCH_LENGTH_MM = 7.0  # prune junction clusters with only tiny outgoing branches
BIFURCATION_BUFFER_MM = 10         # buffer after bifurcation to ignore stenoses
GRADIENT_SEARCH_BACK_MM = 15
# -----------------------------------------------------------

paths_data = []
stenosis_results = []

def load_artery(filepath, artery_name):
    """Ulepszone ładowanie z lepszą kontrolą błędów i preprocessing"""
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
    distances_to_skel = np.linalg.norm(
        (skel_points_abs - current_point) * spacing, axis=1
    )
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

def build_graph(points, skeleton, dist_map, spacing, artery_name,radius_voxels=1.5):
    if len(points) == 0:
        print(f"No points to build graph for {artery_name} artery")
        return nx.Graph(), {}
    point_to_id = {tuple(p): i for i, p in enumerate(points)}
    G = nx.Graph()

    for i, point in enumerate(points):
        diameter = adaptive_diameter_calculation(point, dist_map, skeleton, points, spacing)
        G.add_node(i, pos=point, diameter=diameter)

    # Budujemy KDTree dla przyspieszenia wyszukiwania sąsiadów
    scaled_points = points * spacing  # Przeskaluj do jednostek fizycznych (mm)
    tree = cKDTree(scaled_points)

    # Szukamy sąsiadów w promieniu `radius_voxels` (w mm)
    neighbor_indices_list = tree.query_ball_tree(tree, r=radius_voxels * np.min(spacing) * 1.75)

    for i, neighbors in enumerate(neighbor_indices_list):
        for j in neighbors:
            if i != j and not G.has_edge(i, j):
                dist = np.linalg.norm(scaled_points[i] - scaled_points[j])
                G.add_edge(i, j, weight=dist)
    print(f"{artery_name} artery: graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, point_to_id

# ---------- NEW: BIFURCATION DETECTION FROM SKELETON ----------
def detect_bifurcations_from_skeleton(skeleton, points, spacing,
                                      merge_radius_mm=BIFURCATION_MERGE_RADIUS_MM,
                                      min_branch_length_mm=BIFURCATION_MIN_BRANCH_LENGTH_MM):

    if len(points) == 0:
        return [], np.empty((0,3))

    # Build a set for fast lookup
    pts_set = {tuple(p) for p in points}

    # 26-neighborhood offsets
    offsets = [(dz, dy, dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1) if not (dz==0 and dy==0 and dx==0)]

    # Find candidate junction voxels (>=3 neighbors in skeleton)
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

    # cluster candidate physical points by radius
    tree = cKDTree(cand_phys)
    groups = tree.query_ball_tree(tree, r=merge_radius_mm)

    visited = set()
    clusters = []
    for i in range(len(cand_phys)):
        if i in visited:
            continue
        stack = [i]
        comp = []
        while stack:
            j = stack.pop()
            if j in visited:
                continue
            visited.add(j)
            comp.append(j)
            for nb in groups[j]:
                if nb not in visited:
                    stack.append(nb)
        clusters.append(comp)

    # Precompute adjacency in skeleton (26-neigh)
    # map coords to index in points
    coord_to_idx = {tuple(p): idx for idx, p in enumerate(points)}
    adj = {i: [] for i in range(len(points))}
    for i, p in enumerate(points):
        z,y,x = int(p[0]), int(p[1]), int(p[2])
        for dz,dy,dx in offsets:
            q = (z+dz, y+dy, x+dx)
            if q in coord_to_idx:
                adj[i].append(coord_to_idx[q])

    bif_points_idx = []
    bif_locations = []

    for comp in clusters:
        comp_global = [candidate_indices[i] for i in comp]
        comp_phys = (points[comp_global] * spacing)
        centroid = np.mean(comp_phys, axis=0)
        # pick representative voxel nearest centroid
        dists = np.linalg.norm(comp_phys - centroid, axis=1)
        rep_idx_local = int(np.argmin(dists))
        rep_global_idx = comp_global[rep_idx_local]

        # compute outgoing neighbors outside cluster
        cluster_set = set(comp_global)
        outgoing = [n for n in adj[rep_global_idx] if n not in cluster_set]
        # if no outgoing (weird), skip
        if not outgoing:
            continue

        # for each outgoing, walk along skeleton measure length until endpoint or next junction
        branch_lengths = []
        for start in outgoing:
            prev = rep_global_idx
            cur = start
            length = 0.0
            steps = 0
            max_steps = 10000
            while steps < max_steps:
                steps += 1
                length += np.linalg.norm((points[cur] - points[prev]) * spacing)
                nbrs = [n for n in adj[cur] if n != prev]
                # stop at endpoint or next junction (>=3 neighbors)
                neigh_count_cur = sum(1 for q in adj[cur] if q in pts_set)
                if len(nbrs) == 0 or neigh_count_cur >= 3:
                    break
                prev, cur = cur, nbrs[0]
            branch_lengths.append(length)
        # prune cluster if all outgoing branches are too short (likely spur)
        if outgoing and all(bl < min_branch_length_mm for bl in branch_lengths):
            continue

        bif_points_idx.append(rep_global_idx)
        bif_locations.append(centroid)

    if len(bif_locations) == 0:
        return [], np.empty((0,3))
    return bif_points_idx, np.vstack(bif_locations)


# ---------- NEW: upsample/linear interpolation to get cumulative distances (mm) ----------
def linear_upsample_phys_points(phys_pts, factor=8):
    """Linear upsample along polyline of physical points (mm). Returns fine_pts (Mx3)."""
    if len(phys_pts) < 2:
        return phys_pts.copy()
    segs = np.linalg.norm(np.diff(phys_pts, axis=0), axis=1)
    total = np.sum(segs)
    if total <= 0:
        return np.repeat(phys_pts[:1], max(100, len(phys_pts)*factor), axis=0)
    # allocate total samples proportional to segment length
    total_samples = max(100, len(phys_pts) * factor)
    fine_pts = []
    for i in range(len(phys_pts)-1):
        a = phys_pts[i]; b = phys_pts[i+1]
        seg_len = segs[i]
        nseg = max(2, int(round(total_samples * (seg_len/total))))
        for t in np.linspace(0, 1, nseg, endpoint=False):
            fine_pts.append(a + (b - a) * t)
    fine_pts.append(phys_pts[-1])
    return np.vstack(fine_pts)


def compute_upsampled_path_and_cumulative(path_pts, spacing, upsample_factor=UPSAMPLE_FACTOR):

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
    mapping_idx = []
    for p in phys_pts:
        idx = int(tree.query(p)[1])
        mapping_idx.append(idx)
    mapping_idx = np.array(mapping_idx, dtype=int)
    return phys_pts, cum_orig, fine_pts, cum_fine, mapping_idx

# ---------------- existing helper functions (kept, small edits integrated) ----------------
def rolling_average(arr, window):
    if window < 2:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='same')

def evaluate_gradient(diameters, spacing):
    """Analizuje gradient średnicy i zwraca trend oraz stabilność."""
    grad = np.gradient(diameters)
    mean_grad = np.mean(grad)
    std_grad = np.std(grad)
    return mean_grad, std_grad, grad

# Updated detect_local_stenosis_with_grad: accepts x_positions (distances in mm) to compute correct gradient
def detect_local_stenosis_with_grad(diameters, spacing,
                                    window_mm=15, min_stenosis=15, min_length_pts=5,
                                    use_gradient=False, grad_thresh=-0.05,
                                    smooth_sigma_pts=1.0, combine_with_and=False,
                                    x_positions=None):
    """
    Extended detection computing gradient wrt x_positions (mm) when provided.
    Returns regions, diagnostics (d_smooth, grad (per mm), pct_drop, conds).
    """
    diams = np.array(diameters, dtype=float)
    n = len(diams)
    mean_spacing = float(np.mean(spacing)) if np.ndim(spacing) else float(spacing)
    window_pts = max(1, int(round(window_mm / mean_spacing)))

    # smooth diameters
    if smooth_sigma_pts and smooth_sigma_pts > 0:
        d_smooth = gaussian_filter1d(diams, sigma=smooth_sigma_pts, mode='nearest')
    else:
        d_smooth = diams.copy()

    # gradient per mm if x_positions given
    if x_positions is not None:
        x = np.asarray(x_positions, dtype=float)
        if x.shape[0] != n:
            # if mismatch, fallback to computing grad per mean spacing
            grad = np.gradient(d_smooth) / mean_spacing
        else:
            grad = np.gradient(d_smooth, x)  # mm^-1
    else:
        grad = np.gradient(d_smooth) / mean_spacing

    # percent drop using local proximal reference (same as before)
    pct_drop = np.zeros(n, dtype=float)
    for i in range(n):
        left = max(0, i - window_pts)
        right = min(n, i + window_pts + 1)
        if i - left >= 2:
            ref_region = diams[left:i]
        elif right - i >= 2:
            ref_region = diams[i+1:right]
        else:
            ref_region = diams[max(0, i-window_pts):min(n, i+window_pts+1)]
        if len(ref_region) >= 1:
            ref_val = np.median(ref_region)
            pct_drop[i] = (1 - diams[i] / ref_val) * 100 if ref_val > 0 else 0
        else:
            pct_drop[i] = 0

    cond_pct = pct_drop >= min_stenosis
    cond_grad = grad <= grad_thresh if use_gradient else np.zeros_like(cond_pct, dtype=bool)

    if use_gradient:
        cond_combined = cond_pct & cond_grad if combine_with_and else cond_pct | cond_grad
    else:
        cond_combined = cond_pct

    # group contiguous True regions
    def find_regions(mask):
        regions = []
        i = 0
        while i < len(mask):
            if mask[i]:
                start = i
                while i < len(mask) and mask[i]:
                    i += 1
                end = i - 1
                regions.append((start, end))
            else:
                i += 1
        return regions

    raw_regions = find_regions(cond_combined)
    regions = []
    for (start, end) in raw_regions:
        length = end - start + 1
        if length >= min_length_pts:
            max_idx = start + int(np.argmax(pct_drop[start:end+1]))
            regions.append({
                'start_idx': int(start),
                'end_idx': int(end),
                'length': int(length),
                'max_stenosis': float(np.max(pct_drop[start:end+1])),
                'max_stenosis_idx': int(max_idx)
            })

    diagnostics = {
        'd_smooth': d_smooth,
        'grad': grad,
        'pct_drop': pct_drop,
        'cond_pct': cond_pct,
        'cond_grad': cond_grad,
        'cond_combined': cond_combined
    }

    return regions, diagnostics

# Replace/augment plotting to accept x_positions (mm) and mark ignored regions differently
def plot_diameter_gradient_with_regions(diams, spacing, title, diagnostics=None, regions=None, save_path=None, x_positions=None):
    # diagnostics['grad'] expected to be per mm when x_positions provided
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
    ax2.set_ylabel('Gradient', color='red')
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
        mplt.savefig(save_path, dpi=150)
        print(f"[SAVED] {save_path}")
    mplt.close(fig)

# ----------------------- rest of your original script, with small integration edits -----------------------

def detect_local_stenosis(diameters, spacing, window_mm=15, min_stenosis=30, min_length_pts=5):#
    diameters = np.array(diameters)
    n = len(diameters)
    mean_spacing = np.mean(spacing)
    window_pts = max(2, int(window_mm / mean_spacing))
    stenosis_values = []
    for i in range(n):
        left = max(0, i - window_pts)
        right = min(n, i + window_pts)
        # Lokalna referencja: proksymalny fragment
        region = diameters[left:i] if i > left else diameters[i+1:right]
        if len(region) < 2:
            stenosis_values.append(0)
            continue
        ref_diam = np.median(region)
        stenosis = (1 - diameters[i] / ref_diam) * 100 if ref_diam > 0 else 0
        stenosis = max(0, stenosis)
        stenosis_values.append(stenosis)
    # Szukanie regionów zwężeń
    regions = []
    current = []
    for idx, sten in enumerate(stenosis_values):
        if sten >= min_stenosis:
            current.append((idx, sten))
        else:
            if len(current) >= min_length_pts:
                max_sten = max(current, key=lambda x: x[1])
                regions.append({
                    'start_idx': current[0][0],
                    'end_idx': current[-1][0],
                    'length': len(current),
                    'max_stenosis': max_sten[1],
                    'max_stenosis_idx': max_sten[0]
                })
            current = []
    if len(current) >= min_length_pts:
        max_sten = max(current, key=lambda x: x[1])
        regions.append({
            'start_idx': current[0][0],
            'end_idx': current[-1][0],
            'length': len(current),
            'max_stenosis': max_sten[1],
            'max_stenosis_idx': max_sten[0]
        })
    return regions, stenosis_values

# --- Wczytanie danych ---
left_mask, left_skeleton, left_points, left_dist_map, left_spacing = load_artery(
    r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_left.nrrd", "Left")

right_mask, right_skeleton, right_points, right_dist_map, right_spacing = load_artery(
    r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_right.nrrd", "Right")


# --- Tworzenie grafów ---
G_left, left_point_to_id = build_graph(left_points, left_skeleton, left_dist_map, left_spacing, "Left")
G_right, right_point_to_id = build_graph(right_points, right_skeleton, right_dist_map, right_spacing, "Right")

# Detect bifurcations from skeletons (new)
left_bif_idx, left_bif_locs = detect_bifurcations_from_skeleton(left_skeleton, left_points, left_spacing,
                                                                merge_radius_mm=BIFURCATION_MERGE_RADIUS_MM,
                                                                min_branch_length_mm=BIFURCATION_MIN_BRANCH_LENGTH_MM)
right_bif_idx, right_bif_locs = detect_bifurcations_from_skeleton(right_skeleton, right_points, right_spacing,
                                                                   merge_radius_mm=BIFURCATION_MERGE_RADIUS_MM,
                                                                   min_branch_length_mm=BIFURCATION_MIN_BRANCH_LENGTH_MM)

print(f"Left: detected {len(left_bif_locs)} bifurcation(s) after clustering/pruning.")
print(f"Right: detected {len(right_bif_locs)} bifurcation(s) after clustering/pruning.")

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

selected_points_left = []
selected_points_right = []
visual_objects_left = []
visual_objects_right = []

selected_points_left = []
selected_points_right = []
visual_objects_left = []
visual_objects_right = []
stenosis_objects_left = []  
stenosis_objects_right = []  
removed_stenosis = []  
cadrads_results = []   

cursor_marker = None
cursor_text = None

all_points = np.vstack([left_points, right_points]) if len(left_points) and len(right_points) else (left_points if len(left_points) else right_points)
all_tree = cKDTree(all_points) if len(all_points) else None

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
    left_mesh = vedo.Volume(left_mask).isosurface().c('lightgreen').alpha(0.2)
    left_skel = vedo.Points(left_points, r=3, c='darkgreen')
    if right_mask is not None and np.any(right_mask):
        right_mesh = vedo.Volume(right_mask.astype(np.uint8)).isosurface().c('lightblue').alpha(0.2)
    else:
        right_mesh = None
    right_skel = vedo.Points(right_points, r=3, c='darkblue') if len(right_points) > 0 else None

    objects = [left_mesh, left_skel]
    if right_mesh: objects.append(right_mesh)
    if right_skel: objects.append(right_skel)
    plt.add(objects)
    plt.render()
    print("Zresetowano scenę. Możesz ponownie wybierać punkty.")

def evaluate_gradient(diameters, spacing):
    """Analizuje gradient średnicy i zwraca trend oraz stabilność."""
    grad = np.gradient(diameters)
    mean_grad = np.mean(grad)
    std_grad = np.std(grad)
    return mean_grad, std_grad, grad


def optimize_stenosis_parameters(diameters, spacing, visualize=True):
    """Testuje różne parametry (okno, min_length_pts, próg zwężenia) i wybiera najlepsze."""

    best_params = None
    best_score = np.inf
    results = []

    for w in range (5,30):
        for m in range(3,10):
            for t in range(15,30):
                regions, _ = detect_local_stenosis(diameters, spacing, window_mm=w,
                                                   min_stenosis=t, min_length_pts=m)
                mean_grad, std_grad, _ = evaluate_gradient(diameters, spacing)

                score = abs(mean_grad + 0.01) + std_grad
                results.append((w, m, t, mean_grad, std_grad, len(regions), score))

                if score < best_score:
                    best_score = score
                    best_params = (w, m, t)

    if visualize:
        print("\n=== GRADIENT OPTIMIZATION RESULTS ===")
        print(f"\nOptimal parameters: window={best_params[0]}, min_pts={best_params[1]}, "
              f"sten_th={best_params[2]}")

    return best_params

import numpy as np
from itertools import groupby

def rolling_average_vec(arr, window_pts):
    if window_pts < 2:
        return np.array(arr)
    return np.convolve(arr, np.ones(window_pts)/window_pts, mode='same')

def find_contiguous_regions(mask):
    """Zwraca listę (start, end) indeksów dla kolejnych True w mask."""
    regions = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            end = i-1
            regions.append((start, end))
        else:
            i += 1
    return regions

from scipy.ndimage import gaussian_filter1d

# Replace calls to detection in find_paths with x_positions=cum_orig and use bifurcations to ignore post-bifurcation stenoses.

def find_nearest_node(points, query_point):
    if len(points) == 0:
        return None, None
    scaled = points  
    dists = np.linalg.norm(scaled - query_point, axis=1)
    idx = np.argmin(dists)
    return idx, points[idx]


def nearest_node_index_on_graph(points, query_point):
    idx, pt = find_nearest_node(points, query_point)
    return idx


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

    colors = ['green', 'yellow', 'orange'] if artery == 'left' else ['green', 'yellow']
    point_color = colors[len(selected_points) - 1]
    point_sphere = vedo.Sphere(pos=closest_point, r=1.5, c=point_color)
    plt.add(point_sphere)
    visual_objects.append(point_sphere)
    print(f"[{artery.upper()}] Dodano punkt: Indeks {closest_idx}, Współrzędne {closest_point}")

    if (artery == 'left' and len(selected_points) == 3) or (artery == 'right' and len(selected_points) == 2):
        # call enhanced find_paths that uses physical distances and bifurcation info
        find_paths(artery, G, points, selected_points, visual_objects, dist_map, skeleton, spacing, bif_points_idx, bif_locs)


def find_paths(artery, G, points, selected_points, visual_objects, dist_map, skeleton, spacing, bif_points_idx, bif_locs):
    if (artery == 'left' and len(selected_points) < 3) or (artery == 'right' and len(selected_points) < 2):
        print(f"Za mało punktów dla tętnicy {artery}.")
        return

    start_idx = selected_points[0]
    end1_idx = selected_points[1]
    end2_idx = selected_points[2] if artery == 'left' else None

    def path_diameter_report(path, points, dist_map, skeleton, spacing):
        diameters = []
        for idx in path:
            p = points[idx]
            diam = adaptive_diameter_calculation(p, dist_map, skeleton, points, spacing)
            diameters.append(diam)
        return diameters

    try:
        path1 = nx.shortest_path(G, start_idx, end1_idx)
        path1_points = points[path1]
        diams1 = path_diameter_report(path1, points, dist_map, skeleton, spacing)
        line1 = vedo.Line(path1_points)
        line1.cmap('viridis', diams1)
        line1.lw(8)
        if artery == 'left':
            line1.add_scalarbar(title="Srednica lewej tetnicy [mm]", pos=((0, 0.05), (0.1, 0.35)))
        elif artery == 'right':
            line1.add_scalarbar(title="Srednica prawej tetnicy [mm]", pos=((0.85, 0.05), (0.95, 0.35)))
        plt.add(line1)
        visual_objects.append(line1)
        print(f"[{artery.upper()}] Ścieżka 1: {len(path1)} punktów")
        print("Średnice na ścieżce 1 (mm):", np.round(diams1, 2))

        # --- ENHANCEMENT: compute physical cumulative distances and optionally upsample ---
        phys_pts, cum_orig, fine_pts, cum_fine, mapping_idx = compute_upsampled_path_and_cumulative(path1_points, spacing, upsample_factor=UPSAMPLE_FACTOR)
        total_path_length_mm = float(cum_fine[-1]) if len(cum_fine) > 0 else 0.0
        # build fine-interpolated diameter + gradient on fine samples then map back
        if len(cum_orig) >= 2 and len(cum_fine) >= 2:
            diams_fine = np.interp(cum_fine, cum_orig, diams1)
            grad_fine = np.gradient(diams_fine, cum_fine)
            grad_at_orig = grad_fine[mapping_idx]
        else:
            diams_fine = np.array(diams1)
            cum_fine = np.arange(len(diams1)) * np.mean(spacing)
            grad_fine = np.gradient(diams_fine) / np.mean(spacing)
            grad_at_orig = grad_fine if len(grad_fine) == len(diams1) else np.interp(np.linspace(0,1,len(diams1)), np.linspace(0,1,len(grad_fine)), grad_fine)

        # find which bifurcation reps lie on this path and their distances
        rep_idx_to_pos = {node: pos for pos, node in enumerate(path1)}
        bif_on_path_pos = []
        bif_on_path_dists = []
        for rep in (bif_points_idx if bif_points_idx is not None else []):
            if rep in rep_idx_to_pos:
                ppos = rep_idx_to_pos[rep]
                bif_on_path_pos.append(ppos)
                bif_on_path_dists.append(float(cum_orig[ppos]))
        print(f"[{artery.upper()}] Bifurcations on path at indices {bif_on_path_pos} distances {bif_on_path_dists}")

        # Call detection passing x_positions=cum_orig so gradient is computed per mm
        regions1, diag1 = detect_local_stenosis_with_grad(
            diams1, spacing,
            window_mm=15, min_stenosis=20, min_length_pts=10,
            use_gradient=True, grad_thresh=-0.11,
            smooth_sigma_pts=0.8, combine_with_and=False,
            x_positions=cum_orig
        )
        # override diagnostic grad with grad_at_orig (from fine interpolation) for more consistent onset detection
        diag1['grad'] = grad_at_orig
        grad = diag1['grad']
        print("DEBUG: grad min,max,mean =", grad.min(), grad.max(), grad.mean())
        print("DEBUG: count grad <= -0.1:", np.sum(grad <= -0.1))
        print("DEBUG: indices grad <= -0.1:", np.where(grad <= -0.1)[0])
        print("DEBUG: pct_drop at those indices:", diag1['pct_drop'][np.where(grad <= -0.1)[0]])
        print("DEBUG: detected regions:", regions1)
        print(f"\n[{artery.upper()}] Analiza gradientu dla ścieżki 1:")
        opt_params = optimize_stenosis_parameters(diams1, spacing, visualize=True)

        # proper threaded plotting: pass target function and args
        threading.Thread(target=plot_diameter_gradient_with_regions, args=(diams1, spacing, artery + " - Path 1", diag1, regions1, os.path.join("gradient_analysis", f"{artery}_path1.png"), cum_orig)).start()

        paths_data.append({
            "artery": artery,
            "id": "1",
            "diameters": diams1,
            "spacing": spacing,
            "total_path_length_mm": total_path_length_mm
        })


        if regions1:
            # enrich regions with mm distances, plateau length and check bifurcation proximity
            enriched = []
            total_sten_mm = 0.0
            for region in regions1:
                s_idx = region['start_idx']; e_idx = region['end_idx']
                region['start_distance_mm'] = float(cum_orig[s_idx])
                region['end_distance_mm'] = float(cum_orig[e_idx])
                # length via fine cumulative mapping
                fine_s = mapping_idx[s_idx]; fine_e = mapping_idx[e_idx]
                region['length_mm'] = float(cum_fine[fine_e] - cum_fine[fine_s])
                total_sten_mm += region['length_mm']
                # plateau detection (use original diams1 and mapping to fine)
                min_d = np.min(diams1)
                tol_val = min_d * 1.05
                mask_plateau = np.array(diams1) <= tol_val
                # simple plateau length: contiguous region containing max_stenosis_idx
                plat_len_mm = 0.0
                s = region['start_idx']; e = region['end_idx']
                # compute plateau length by mapping plateau region to fine and summing fine distances
                plateau_mask = mask_plateau[s:e+1]
                if np.any(plateau_mask):
                    # find contiguous chunk that contains max idx
                    maxpos = region['max_stenosis_idx']
                    # find plateau region boundaries around maxpos
                    left = maxpos
                    while left > s and mask_plateau[left-1]:
                        left -= 1
                    right = maxpos
                    while right < e and mask_plateau[right+1]:
                        right += 1
                    fine_left = mapping_idx[left]; fine_right = mapping_idx[right]
                    plat_len_mm = float(cum_fine[fine_right] - cum_fine[fine_left])
                region['plateau_length_mm'] = plat_len_mm

                # find gradient onset: search backwards up to GRADIENT_SEARCH_BACK_MM for grad <= grad_thresh
                grad_orig = diag1['grad']
                grad_onset_idx = None
                back_points = max(1, int(round(GRADIENT_SEARCH_BACK_MM / max(np.mean(np.diff(cum_orig)), 1e-6))))
                j = s_idx
                while j >= max(0, s_idx - back_points):
                    if grad_orig[j] <= -0.1:
                        grad_onset_idx = j
                    j -= 1
                if grad_onset_idx is None:
                    grad_onset_idx = s_idx
                region['grad_onset_idx'] = int(grad_onset_idx)
                region['grad_onset_distance_mm'] = float(cum_orig[grad_onset_idx])

                # determine ignore flag due to bifurcation proximity:
                ignored = False
                for bif_dist in bif_on_path_dists:
                    # if bif occurs before region start and region start is within buffer after bif -> ignore
                    if (bif_dist <= region['start_distance_mm']) and (region['start_distance_mm'] - bif_dist <= BIFURCATION_BUFFER_MM):
                        ignored = True
                        break
                region['ignored_due_to_bifurcation'] = bool(ignored)
                enriched.append(region)

            # print and visualize only non-ignored regions as red; ignored marked but not added to stenosis_objects
            print(f"[{artery.upper()}] Detected {len(enriched)} stenosis regions (some may be ignored due to bifurcation).")
            for region in enriched:
                if region.get('ignored_due_to_bifurcation'):
                    print(f" Region IGNORED due to bifurcation: indices {region['start_idx']}..{region['end_idx']}, start_dist={region['start_distance_mm']:.2f} mm")
                else:
                    print(f" Region: indices {region['start_idx']}..{region['end_idx']}, length={region['length_mm']:.2f} mm, max stenosis {region['max_stenosis']:.1f}%")
                    sten_point = points[path1[region['max_stenosis_idx']]]
                    sten_sphere = vedo.Sphere(pos=sten_point, r=3.0, c='red').alpha(0.5)
                    sten_sphere.pickable(True) 
                    plt.add(sten_sphere)
                    if artery == 'left':
                        stenosis_objects_left.append(sten_sphere)
                    else:
                        stenosis_objects_right.append(sten_sphere)
            # optionally you can add logic to store enriched regions in paths_data
            paths_data[-1]['regions'] = enriched
            paths_data[-1]['total_stenosis_length_mm'] = total_sten_mm
            valid_regions = [r for r in regions1 if not r.get('ignored_due_to_bifurcation', False)]
            if valid_regions:
                    max_stenosis1 = max(r['max_stenosis'] for r in valid_regions)
                    stenosis_results.append({
                        "artery": artery,
                        "path": "1",
                        "side": "L" if artery == 'left' else "P",
                        "stenosis": max_stenosis1
                    })
        else:
            print(f"[{artery.upper()}] Nie znaleziono istotnych zwężeń dla ścieżki 1")

    except nx.NetworkXNoPath:
        print(f"[{artery.upper()}] Nie znaleziono ścieżki 1")

    if artery == 'left':
        try:
            path2 = nx.shortest_path(G, start_idx, end2_idx)
            path2_points = points[path2]
            diams2 = path_diameter_report(path2, points, dist_map, skeleton, spacing)
            line2 = vedo.Line(path2_points)
            line2.cmap('viridis', diams2)
            line2.lw(8)
            plt.add(line2)
            visual_objects.append(line2)
            print(f"[{artery.upper()}] Ścieżka 2: {len(path2)} punktów")
            print("Średnice na ścieżce 2 (mm):", np.round(diams2, 2))

            # Repeat same enhanced analysis for path2
            phys_pts, cum_orig2, fine_pts2, cum_fine2, mapping_idx2 = compute_upsampled_path_and_cumulative(path2_points, spacing, upsample_factor=UPSAMPLE_FACTOR)
            if len(cum_orig2) >= 2 and len(cum_fine2) >= 2:
                diams_fine2 = np.interp(cum_fine2, cum_orig2, diams2)
                grad_fine2 = np.gradient(diams_fine2, cum_fine2)
                grad_at_orig2 = grad_fine2[mapping_idx2]
            else:
                grad_at_orig2 = np.gradient(np.array(diams2)) / np.mean(spacing)

            regions2, diag2 = detect_local_stenosis_with_grad(
                diams2, spacing,
                window_mm=15, min_stenosis=20, min_length_pts=10,
                use_gradient=True, grad_thresh=-0.11,
                smooth_sigma_pts=0.8, combine_with_and=False,
                x_positions=cum_orig2
            )
            diag2['grad'] = grad_at_orig2
            grad = diag2['grad']
            print("DEBUG: grad min,max,mean =", grad.min(), grad.max(), grad.mean())
            print("DEBUG: count grad <= -0.1:", np.sum(grad <= -0.1))
            print("DEBUG: indices grad <= -0.1:", np.where(grad <= -0.1)[0])
            print("DEBUG: pct_drop at those indices:", diag2['pct_drop'][np.where(grad <= -0.1)[0]])
            print("DEBUG: detected regions:", regions2)
            print(f"\n[{artery.upper()}] Analiza gradientu dla ścieżki 2:")
            opt_params2 = optimize_stenosis_parameters(diams2, spacing, visualize=True)
            threading.Thread(target=plot_diameter_gradient_with_regions, args=(diams2, spacing, artery + " - Path 2", diag2, regions2, os.path.join("gradient_analysis", f"{artery}_path2.png"), cum_orig2)).start()

            paths_data.append({
                "artery": artery,
                "id": "2",
                "diameters": diams2,
                "spacing": spacing
            })

            if regions2:
                enriched2 = []
                total_st2 = 0.0
                # compute mm distances etc as above, and apply bifurcation filtering
                rep_idx_to_pos2 = {node: pos for pos, node in enumerate(path2)}
                bif_on_path_pos2 = []
                bif_on_path_dists2 = []
                for rep in (bif_points_idx if bif_points_idx is not None else []):
                    if rep in rep_idx_to_pos2:
                        ppos = rep_idx_to_pos2[rep]
                        bif_on_path_pos2.append(ppos)
                        bif_on_path_dists2.append(float(cum_orig2[ppos]))

                for region in regions2:
                    s_idx = region['start_idx']; e_idx = region['end_idx']
                    fine_s = mapping_idx2[s_idx]; fine_e = mapping_idx2[e_idx]
                    length_mm = float(cum_fine2[fine_e] - cum_fine2[fine_s])
                    region['start_distance_mm'] = float(cum_orig2[s_idx])
                    region['end_distance_mm'] = float(cum_orig2[e_idx])
                    region['length_mm'] = length_mm
                    total_st2 += length_mm
                    # plateau (simple)
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
                    back_points = max(1, int(round(GRADIENT_SEARCH_BACK_MM / max(np.mean(np.diff(cum_orig2)), 1e-6))))
                    j = s_idx
                    while j >= max(0, s_idx - back_points):
                        if grad_orig2[j] <= -0.1:
                            grad_onset_idx = j
                        j -= 1
                    if grad_onset_idx is None:
                        grad_onset_idx = s_idx
                    region['grad_onset_idx'] = int(grad_onset_idx)
                    region['grad_onset_distance_mm'] = float(cum_orig2[grad_onset_idx])
                    # bifurcation ignore
                    ignored = False
                    for bif_dist in bif_on_path_dists2:
                        if (bif_dist <= region['start_distance_mm']) and (region['start_distance_mm'] - bif_dist <= BIFURCATION_BUFFER_MM):
                            ignored = True
                            break
                    region['ignored_due_to_bifurcation'] = bool(ignored)
                    enriched2.append(region)

                paths_data[-1]['regions'] = enriched2
                paths_data[-1]['total_stenosis_length_mm'] = total_st2
                valid_regions2 = [r for r in regions2 if not r.get('ignored_due_to_bifurcation', False)]
                if valid_regions2:
                    max_stenosis2 = max(r['max_stenosis'] for r in valid_regions2)
                    stenosis_results.append({
                        "artery": artery,
                        "path": "2",
                        "side": "L",  # ścieżka 2 istnieje tylko dla lewej
                        "stenosis": max_stenosis2
                            })
                for region in enriched2:
                    if region.get('ignored_due_to_bifurcation'):
                        print(f" Region IGNORED due to bifurcation: indices {region['start_idx']}..{region['end_idx']}, start_dist={region['start_distance_mm']:.2f} mm")
                    else:
                        sten_point = points[path2[region['max_stenosis_idx']]]
                        sten_sphere = vedo.Sphere(pos=sten_point, r=1.5, c='red')
                        plt.add(sten_sphere)
                        stenosis_objects_left.append(sten_sphere)
            else:
                print(f"[{artery.upper()}] Nie znaleziono istotnych zwężeń dla ścieżki 2")

            if 'path1' in locals() and 'path2' in locals():
                common_length = min(len(path1), len(path2))
                branch_point_idx = 0
                for i in range(common_length):
                    if path1[i] != path2[i]:
                        break
                    branch_point_idx = i
                branch_point = points[path1[branch_point_idx]]
                branch_sphere = vedo.Sphere(pos=branch_point, r=1, c='purple')
                plt.add(branch_sphere)
                visual_objects.append(branch_sphere)
                print(f"[{artery.upper()}] Punkt rozgałęzienia: {branch_point}")
        except nx.NetworkXNoPath:
            print(f"[{artery.upper()}] Nie znaleziono ścieżki 2")

    plt.render()


def on_mouse_move(event):
    for obj in stenosis_objects_left + stenosis_objects_right:
        obj.color('red').alpha(0.7) 

    if event.actor and (event.actor in stenosis_objects_left or event.actor in stenosis_objects_right):
        event.actor.color('yellow').alpha(0.9) 
        plt.render()


def reload_removed_stenosis():
    global stenosis_objects_left, stenosis_objects_right

    for stenosis in removed_stenosis:
        sten_sphere = vedo.Sphere(
            pos=stenosis['position'],
            r=stenosis['radius']
        ).color(stenosis['color']).alpha(stenosis['alpha'])

        if stenosis['artery'] == 'left':
            stenosis_objects_left.append(sten_sphere)
        else:
            stenosis_objects_right.append(sten_sphere)

        plt.add(sten_sphere)

    removed_stenosis.clear()
    plt.render()


def cadrads_score():
    """
    Klasyfikacja CAD-RADS (0–5, z 4A/4B) na podstawie wykrytych zwężeń w stenosis_results.
    """
    results_summary = []

    if not stenosis_results:
        print("Brak danych o zwężeniach – nie można obliczyć CAD-RADS.")
        return None

    # policz ile istotnych (>=70%) zwężeń jest po każdej stronie
    total_high_left = sum(1 for r in stenosis_results if r["side"] == "L" and r["stenosis"] >= 70)
    total_high_right = sum(1 for r in stenosis_results if r["side"] == "P" and r["stenosis"] >= 70)

    for res in stenosis_results:
        stenosis_geom_pct = res["stenosis"]
        side = res["side"]

        # przeliczenie geometrycznego na "kliniczne" (bardziej realistyczne)
        stenosis_clinical = 100 * (1 - (1 - stenosis_geom_pct / 100) ** 2)

        # klasyfikacja CAD-RADS
        if stenosis_clinical < 1:
            cadrads = "0"
        elif stenosis_clinical < 25:
            cadrads = "1"
        elif stenosis_clinical < 50:
            cadrads = "2"
        elif stenosis_clinical < 70:
            cadrads = "3"
        elif stenosis_clinical < 100:
            # rozróżnienie 4A / 4B — jeśli więcej niż jedno istotne zwężenie po tej stronie
            if (side == "L" and total_high_left >= 2) or (side == "P" and total_high_right >= 2):
                cadrads = "4B"
            else:
                cadrads = "4A"
        else:
            cadrads = "5"

        results_summary.append({
            "artery": res["artery"],
            "path": res["path"],
            "side": side,
            "stenosis_geom": stenosis_geom_pct,
            "stenosis_clinical": round(stenosis_clinical, 2),
            "CAD-RADS": cadrads
        })

    print("\n===== CAD-RADS CLASSIFICATION =====")
    for r in results_summary:
        print(f"[{r['side']}] Path {r['path']} | Stenosis: {r['stenosis_geom']:.1f}% "
              f"(clinical {r['stenosis_clinical']:.1f}%) → CAD-RADS {r['CAD-RADS']}")
    print("==================================\n")

    return results_summary


# --- FINAL CAD-RADS SUMMARY ---
if stenosis_results:
    n_left_high = sum(1 for s in stenosis_results if s["side"] == "L" and s["stenosis"] >= 55)
    n_right_high = sum(1 for s in stenosis_results if s["side"] == "P" and s["stenosis"] >= 55)

    for s in stenosis_results:
        cadrads, stenosis_clinical = cadrads_score(s["stenosis"], s["side"], n_left_high, n_right_high)
        s["stenosis_clinical"] = stenosis_clinical
        s["CAD-RADS"] = cadrads

    print("\n=== PODSUMOWANIE CAD-RADS ===")
    for s in stenosis_results:
        print(f"Tętnica {s['artery']} ścieżka {s['path']} ({s['side']}): CAD-RADS {s['CAD-RADS']} "
              f"| Zwężenie kliniczne {s['stenosis_clinical']:.1f}% (geometryczne: {s['stenosis']:.1f}%)")

    global_cadrads = max(s["CAD-RADS"] for s in stenosis_results)
    print(f"\nGlobalny wynik CAD-RADS: {global_cadrads}")
else:
    print("\nBrak wykrytych zwężeń do klasyfikacji CAD-RADS.")

def show_cadrads_report():
    if not cadrads_results:
        print("Brak danych do wygenerowania raportu CAD-RADS")
        return

    latest_report = cadrads_results[-1]

    report_text = f"""
    ===== RAPORT CAD-RADS =====
    Data: {latest_report['timestamp']}
    Laczna liczba zwezen: {latest_report['total_stenosis']}
    - Lewa tetnica: {latest_report['left_artery']}
    - Prawa tetnica: {latest_report['right_artery']}
    Ocena CAD-RADS: {latest_report['cadrads_score']}
    """

    print(report_text)

    txt = vedo.Text2D(report_text, pos='top-left', c='k', bg='y', alpha=0.8)
    plt.add(txt).render()

def on_mouse_move(event):
    global cursor_marker, cursor_text
    for obj in stenosis_objects_left + stenosis_objects_right:
        try:
            obj.color('red').alpha(0.6)
        except Exception:
            pass

    if event.actor and (event.actor in stenosis_objects_left or event.actor in stenosis_objects_right):
        event.actor.color('yellow').alpha(0.9)

    picked3d = event.picked3d
    if picked3d is None:
        return

    global all_tree
    if all_tree is None:
        return
    dist, idx = all_tree.query(picked3d)
    if dist > 10.0:
        if cursor_marker:
            plt.remove(cursor_marker)
            cursor_marker = None
        if cursor_text:
            plt.remove(cursor_text)
            cursor_text = None
        plt.render()
        return

    nearest_pt = all_points[idx]

    if len(left_points) and np.any(np.all(nearest_pt == left_points, axis=1)):
        pt_idx = np.where(np.all(nearest_pt == left_points, axis=1))[0][0]
        diam = adaptive_diameter_calculation(left_points[pt_idx], left_dist_map, left_skeleton, left_points, left_spacing)
    else:
        pt_idx = np.where(np.all(nearest_pt == right_points, axis=1))[0][0]
        diam = adaptive_diameter_calculation(right_points[pt_idx], right_dist_map, right_skeleton, right_points, right_spacing)

    if cursor_marker:
        plt.remove(cursor_marker)
    cursor_marker = vedo.Sphere(pos=nearest_pt, r=1.2, c='white').alpha(0.9)
    plt.add(cursor_marker)

    if cursor_text:
        plt.remove(cursor_text)
    txt = f"D={diam:.2f} mm\npt={nearest_pt.tolist()}"
    cursor_text = vedo.Text2D(txt, pos='top-left', c='k', bg='w', alpha=0.8)
    plt.add(cursor_text)

    plt.render()


def handle_keypress(event):
    if event.keypress == 'r':  
        reset_selection('all')
    elif event.keypress == 'l':   
        reload_removed_stenosis()
    elif event.keypress == 'm':   
        cadrads_score()
        show_cadrads_report()

plt.add_callback('LeftButtonPress', handle_click)
plt.add_callback('MouseMove', on_mouse_move)
plt.add_callback("KeyPress", handle_keypress)
plt.interactive().close()

def analyze_gradients_after_close(paths_data):
    """
    Analizuje wszystkie ścieżki po zamknięciu okna Vedo:
    - testuje różne parametry (window, min_pts, sten_th),
    - wybiera najlepsze,
    - pokazuje wygładzoną krzywą gradientu i średnicy,
    - zapisuje wykresy do folderu.
    """
    if not paths_data:
        print("[INFO] Brak danych ścieżek do analizy.")
        return

    os.makedirs("gradient_analysis", exist_ok=True)
    print("\n=== ROZPOCZYNAM ANALIZĘ ŚCIEŻEK ===")

    for path_info in paths_data:
        artery = path_info["artery"]
        pid = path_info["id"]
        diam = np.array(path_info["diameters"])
        spacing = np.array(path_info["spacing"], dtype=float)

        def evaluate_params(d, spacing, window, min_pts, sten_th):
            regs, _ = detect_local_stenosis(d, spacing=spacing,
                                            window_mm=window,
                                            min_stenosis=sten_th,
                                            min_length_pts=min_pts)
            grad = np.gradient(d)
            mean_grad = np.mean(grad)
            std_grad = np.std(grad)
            score = abs(mean_grad + 0.01) + std_grad + 0.02 * max(0, len(regs) - 3)
            return {"regs": regs, "score": score, "mean_grad": mean_grad, "std_grad": std_grad}

        best = {"score": np.inf}
        for w in [10, 15, 20, 25, 30]:
            for m in [4, 5, 6, 7]:
                for t in [20, 25, 30]:
                    res = evaluate_params(diam, spacing, w, m, t)
                    if res["score"] < best["score"]:
                        best = res | {"window": w, "min_pts": m, "sten_th": t}

        dist = np.arange(len(diam)) * np.mean(spacing)
        grad = np.gradient(diam)
        grad_smooth = gaussian_filter1d(grad, sigma=2)
        diam_smooth = gaussian_filter1d(diam, sigma=1)

        fig, ax1 = mplt.subplots(figsize=(10, 4))
        ax1.plot(dist, diam_smooth, 'b-', label="Diameter [mm]")
        ax1.set_xlabel("Path length [mm]")
        ax1.set_ylabel("Diameter [mm]", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(dist, grad_smooth, 'r--', label="Gradient (smoothed)")
        ax2.set_ylabel("Gradient", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax1.set_title(f"{artery.upper()} - Path {pid}")
        fig.tight_layout()
        save_path = f"gradient_analysis/{artery}_path{pid}.png"
        mplt.savefig(save_path, dpi=150)
        print(f"[SAVED] {save_path}")
        mplt.close(fig)


    print("\n=== ANALIZA ZAKOŃCZONA ===")

analyze_gradients_after_close(paths_data)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def preview_all_plots(folder="gradient_analysis"):
    imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
    if not imgs:
        print("[INFO] No saved plots found.")
        return
    for img in imgs:
        image = mpimg.imread(img)
        plt.figure(figsize=(10, 4))
        plt.imshow(image)
        plt.axis('off')
        plt.title(os.path.basename(img))
        plt.show()

# Na samym końcu programu:
preview_all_plots()
