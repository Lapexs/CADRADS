#!/usr/bin/env python3
"""
Improved CAD-RADS analysis script with selection fixes.

Main fix in this version:
 - Picking/selection no longer depends on event.actor being present.
 - Click handler always uses event.picked3d (3D picked position) and a KDTree
   of skeleton points to determine nearest skeleton point, so clicking on
   the mesh/points or slightly off them will still select the nearest point.
 - Additional defensive checks and clearer debug prints around selection.

Controls in Vedo window:
 - Left mouse click: select (or remove stenosis marker if clicked near one)
 - r: reset selections
 - l: reload removed stenosis
 - c: calculate CAD-RADS
 - h: toggle hybrid detection
 - v: save diagnostics for last computed path(s)
 - q: quit

Update paths to NRRD files at the bottom before running.
"""

import os
import time
import threading

import nrrd
import numpy as np
import networkx as nx
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

import vedo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mplt

# ------------------------
# Global configuration / toggles
# ------------------------
DEBUG_SAVE_DIAGNOSTICS = False
USE_HYBRID = True
HYBRID_PARAMS = {
    'window_mm': 15,
    'sten_th': 15,
    'min_length_pts': 3,
    'smooth_pts': 9,
    'grad_z_thresh': -0.8,
    'use_and': False
}

paths_data = []

# ------------------------
# I/O and spacing extraction
# ------------------------
def extract_spacing(header):
    try:
        if 'space directions' in header:
            space_dirs = header['space directions']
            if space_dirs is not None:
                spacing = []
                for direction in space_dirs:
                    if direction is not None and hasattr(direction, '__len__'):
                        vals = [x for x in direction if x is not None and not (isinstance(x, float) and np.isnan(x))]
                        norm = np.linalg.norm(vals) if len(vals) else 1.0
                        spacing.append(float(norm if norm > 0 else 1.0))
                    else:
                        spacing.append(1.0)
                if len(spacing) >= 3:
                    return np.array(spacing[:3], dtype=float)
        if 'spacing' in header and header['spacing'] is not None:
            spacing = header['spacing']
            return np.array([float(s if s is not None and not (isinstance(s, float) and np.isnan(s)) else 1.0) for s in spacing[:3]])
    except Exception:
        pass
    print("WARNING: Spacing not found or invalid in NRRD header; assuming [0.5,0.5,0.5] mm")
    return np.array([0.5, 0.5, 0.5], dtype=float)


def load_artery(filepath, artery_name):
    try:
        data, header = nrrd.read(filepath)
    except Exception as e:
        print(f"Error reading {artery_name} file: {e}")
        return None, None, np.array([], dtype=int), None, np.array([1.0, 1.0, 1.0])

    mask = (data > 0).astype(np.uint8)
    if mask.sum() == 0:
        print(f"{artery_name}: empty mask")
        return mask, np.zeros_like(mask, dtype=np.uint8), np.array([], dtype=int), None, np.array([1.0, 1.0, 1.0])

    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) > 0:
        largest_component = max(props, key=lambda x: x.area)
        mask = (labeled == largest_component.label).astype(np.uint8)

    mask_smooth = gaussian(mask.astype(float), sigma=0.5) > 0.5
    skeleton = skeletonize(mask_smooth).astype(np.uint8)
    points = np.argwhere(skeleton > 0)
    spacing = extract_spacing(header)
    dist_map = ndimage.distance_transform_edt(mask.astype(float), sampling=spacing)

    print(f"{artery_name} artery: {len(points)} skeleton points; spacing={spacing}")
    return mask.astype(np.uint8), skeleton.astype(np.uint8), points.astype(int), dist_map.astype(float), spacing.astype(float)


# ------------------------
# Diameter estimation
# ------------------------
def adaptive_diameter_calculation(p, dist_map, skeleton, points, spacing, base_window=7):
    if dist_map is None or dist_map.size == 0:
        return 0.0
    z, y, x = [int(v) for v in p]
    if not (0 <= z < dist_map.shape[0] and 0 <= y < dist_map.shape[1] and 0 <= x < dist_map.shape[2]):
        return 0.0

    initial_radius_mm = float(dist_map[z, y, x])
    min_spacing = max(1e-6, float(np.min(spacing)))
    adaptive_window = max(2, min(base_window, int(np.ceil(initial_radius_mm / min_spacing)) + 1))

    z0 = max(0, z - adaptive_window); z1 = min(dist_map.shape[0], z + adaptive_window + 1)
    y0 = max(0, y - adaptive_window); y1 = min(dist_map.shape[1], y + adaptive_window + 1)
    x0 = max(0, x - adaptive_window); x1 = min(dist_map.shape[2], x + adaptive_window + 1)

    skel_region = skeleton[z0:z1, y0:y1, x0:x1]
    skel_pts = np.argwhere(skel_region > 0)
    if skel_pts.size == 0:
        return 2.0 * initial_radius_mm

    skel_pts_abs = skel_pts + np.array([z0, y0, x0])
    diffs = (skel_pts_abs - np.array([z, y, x])) * spacing
    dists_mm = np.linalg.norm(diffs, axis=1)

    radius_threshold_mm = max(1.0, initial_radius_mm * 1.25)
    nearby_idx = dists_mm <= radius_threshold_mm
    if nearby_idx.sum() == 0:
        radii = [float(dist_map[tuple(pt)]) for pt in skel_pts_abs[:min(10, len(skel_pts_abs))]]
        if len(radii) == 0:
            return 2.0 * initial_radius_mm
        return float(2.0 * np.median(radii))

    diameters_mm = []
    for idx in np.where(nearby_idx)[0]:
        sz, sy, sx = skel_pts_abs[idx]
        if 0 <= sz < dist_map.shape[0] and 0 <= sy < dist_map.shape[1] and 0 <= sx < dist_map.shape[2]:
            r = float(dist_map[sz, sy, sx])
            if r > 0:
                diameters_mm.append(2.0 * r)
    if len(diameters_mm) == 0:
        return 2.0 * initial_radius_mm
    return float(np.median(diameters_mm))


# ------------------------
# Build graph from skeleton points
# ------------------------
def build_graph(points, skeleton, dist_map, spacing, artery_name, neighbor_radius_mm=3.0):
    if points is None or len(points) == 0:
        print(f"No points to build graph for {artery_name}")
        return nx.Graph(), {}
    point_to_id = {tuple(p): i for i, p in enumerate(points)}
    G = nx.Graph()
    for i, point in enumerate(points):
        diameter = adaptive_diameter_calculation(point, dist_map, skeleton, points, spacing)
        G.add_node(i, pos=tuple(point.tolist()), diameter=float(diameter))

    scaled_points = points.astype(float) * spacing.reshape((1, 3))
    tree = cKDTree(scaled_points)
    neighbor_indices_list = tree.query_ball_tree(tree, r=float(neighbor_radius_mm))

    for i, neighbors in enumerate(neighbor_indices_list):
        for j in neighbors:
            if i == j:
                continue
            if not G.has_edge(i, j):
                dist_mm = float(np.linalg.norm(scaled_points[i] - scaled_points[j]))
                G.add_edge(i, j, weight=dist_mm)
    print(f"{artery_name} artery: graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, point_to_id


# ------------------------
# Stenosis detection and hybrid detection
# ------------------------
def detect_local_stenosis(diameters, spacing, window_mm=15, min_stenosis=30, min_length_pts=5):
    diameters = np.asarray(diameters, dtype=float)
    n = len(diameters)
    if n == 0:
        return [], []
    mean_spacing = float(np.mean(spacing))
    window_pts = max(1, int(round(window_mm / mean_spacing)))

    diam_smooth = gaussian_filter1d(diameters, sigma=1.0) if n > 2 else diameters.copy()
    grad = np.gradient(diam_smooth) / mean_spacing

    stenosis_values = []
    for i in range(n):
        left = max(0, i - window_pts)
        right = min(n, i + window_pts + 1)
        if i - left >= 2:
            region = diameters[left:i]
        elif right - i >= 2:
            region = diameters[i+1:right]
        else:
            region = diameters[max(0, i-window_pts):min(n, i+window_pts+1)]
        if region.size < 1 or np.median(region) == 0:
            stenosis_values.append(0.0)
        else:
            ref = float(np.median(region))
            sten = max(0.0, (1.0 - diameters[i] / ref) * 100.0)
            stenosis_values.append(float(sten))

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
                    'max_stenosis': float(max_sten[1]),
                    'max_stenosis_idx': int(max_sten[0])
                })
            current = []
    if len(current) >= min_length_pts:
        max_sten = max(current, key=lambda x: x[1])
        regions.append({
            'start_idx': current[0][0],
            'end_idx': current[-1][0],
            'length': len(current),
            'max_stenosis': float(max_sten[1]),
            'max_stenosis_idx': int(max_sten[0])
        })

    filtered = []
    for r in regions:
        start, end = r['start_idx'], r['end_idx']
        diam_range = np.ptp(diameters[start:end+1]) if end >= start else 0.0
        region_grad_mean = float(np.mean(grad[start:end+1])) if end >= start else 0.0
        # we want negative gradient => keep when region_grad_mean < negative threshold
        if diam_range >= 0.1 and region_grad_mean < -0.02:
            filtered.append(r)
    return filtered, stenosis_values


def rolling_average_vec(arr, window_pts):
    if window_pts < 2:
        return np.array(arr)
    return np.convolve(arr, np.ones(window_pts)/window_pts, mode='same')


def find_contiguous_regions(mask):
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


def detect_hybrid_stenosis(diams, spacing, window_mm=15, sten_th=30,
                           min_length_pts=5, smooth_pts=9, grad_z_thresh=-0.8,
                           use_and=False):
    diams = np.array(diams, dtype=float)
    mean_spacing = float(np.mean(spacing))
    window_pts = max(1, int(window_mm / mean_spacing))

    d_smooth = rolling_average_vec(diams, smooth_pts) if len(diams) > 1 else diams.copy()
    grad = np.gradient(d_smooth) / mean_spacing
    grad_mean = np.mean(grad)
    grad_std = np.std(grad) if np.std(grad) > 1e-6 else 1.0
    grad_z = (grad - grad_mean) / grad_std

    n = len(diams)
    pct_drop = np.zeros(n)
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
            if ref_val > 0:
                pct_drop[i] = (1 - diams[i]/ref_val) * 100
            else:
                pct_drop[i] = 0
        else:
            pct_drop[i] = 0

    cond_pct = pct_drop >= sten_th
    cond_grad = grad_z <= grad_z_thresh
    if use_and:
        cond_combined = cond_pct & cond_grad
    else:
        cond_combined = cond_pct | cond_grad

    regions = []
    for (start, end) in find_contiguous_regions(cond_combined):
        length = end - start + 1
        if length >= min_length_pts:
            max_sten_idx = start + int(np.argmax(pct_drop[start:end+1]))
            regions.append({
                'start_idx': start,
                'end_idx': end,
                'length': length,
                'max_stenosis': float(np.max(pct_drop[start:end+1])),
                'max_stenosis_idx': int(max_sten_idx)
            })

    diagnostics = {
        'd_smooth': d_smooth,
        'grad': grad,
        'grad_z': grad_z,
        'pct_drop': pct_drop,
        'cond_pct': cond_pct,
        'cond_grad': cond_grad,
        'cond_combined': cond_combined
    }
    return regions, diagnostics


# ------------------------
# Diagnostics plotting
# ------------------------
def save_diagnostics_plot(diams, spacing, diagnostics, regions, title, outdir="gradient_analysis"):
    os.makedirs(outdir, exist_ok=True)
    dist = np.arange(len(diams)) * float(np.mean(spacing))
    fig, ax1 = mplt.subplots(figsize=(10, 4))
    ax1.plot(dist, diams, color='blue', label='Diameter')
    if diagnostics is not None and 'd_smooth' in diagnostics:
        ax1.plot(dist, diagnostics['d_smooth'], color='cyan', alpha=0.6, label='Smoothed diam')
    ax1.set_xlabel('Path length [mm]')
    ax1.set_ylabel('Diameter [mm]', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    if diagnostics is not None:
        ax2.plot(dist, diagnostics['grad'], color='red', linestyle='--', label='Gradient')
    ax2.set_ylabel('Gradient', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    if diagnostics is not None and 'pct_drop' in diagnostics:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(dist, diagnostics['pct_drop'], color='magenta', linestyle=':', label='% drop')
        ax1_twin.set_ylabel('% drop', color='magenta')
        ax1_twin.tick_params(axis='y', labelcolor='magenta')

    if regions:
        for reg in regions:
            start = reg['start_idx'] * float(np.mean(spacing))
            end = reg['end_idx'] * float(np.mean(spacing))
            ax1.axvspan(start, end, color='red', alpha=0.12)

    ax1.set_title(title)
    fig.tight_layout()
    outpath = os.path.join(outdir, f"{title.replace(' ', '_')}.png")
    mplt.savefig(outpath, dpi=150)
    mplt.close(fig)
    print(f"[SAVED] diagnostics -> {outpath}")
    return outpath


# ------------------------
# Visualization & interaction
# ------------------------
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

all_points = None
all_tree = None

left_mask = None
left_skeleton = None
left_points = np.array([], dtype=int)
left_dist_map = None
left_spacing = np.array([1.0, 1.0, 1.0])

right_mask = None
right_skeleton = None
right_points = np.array([], dtype=int)
right_dist_map = None
right_spacing = np.array([1.0, 1.0, 1.0])

G_left = nx.Graph()
G_right = nx.Graph()

left_mesh = None
left_skel = None
right_mesh = None
right_skel = None

last_computed_paths = []


def reset_selection(plotter, artery='all'):
    global selected_points_left, selected_points_right
    global visual_objects_left, visual_objects_right
    global stenosis_objects_left, stenosis_objects_right
    if artery in ['left', 'all']:
        for obj in visual_objects_left + stenosis_objects_left:
            try:
                plotter.remove(obj)
            except Exception:
                pass
        selected_points_left.clear()
        visual_objects_left.clear()
        stenosis_objects_left.clear()
    if artery in ['right', 'all']:
        for obj in visual_objects_right + stenosis_objects_right:
            try:
                plotter.remove(obj)
            except Exception:
                pass
        selected_points_right.clear()
        visual_objects_right.clear()
        stenosis_objects_right.clear()
    objs = []
    if left_mesh is not None: objs.append(left_mesh)
    if left_skel is not None: objs.append(left_skel)
    if right_mesh is not None: objs.append(right_mesh)
    if right_skel is not None: objs.append(right_skel)
    plotter.clear()
    plotter.add(objs)
    plotter.render()
    print(f"[{artery.upper()}] Reset selection")


def find_nearest_node(points, query_point):
    if points is None or len(points) == 0:
        return None, None
    dists = np.linalg.norm(points - query_point, axis=1)
    idx = int(np.argmin(dists))
    return idx, points[idx]


def handle_click(event):
    """
    Main fix is here: do not rely on event.actor. Use event.picked3d coordinates
    and a KDTree (all_tree) to find nearest skeleton point. If there are stenosis
    markers nearby, prefer deletion of those.
    """
    global removed_stenosis, all_tree, all_points
    try:
        # Ignore keypress events routed to this callback
        if event.keypress:
            return

        clicked_pos = getattr(event, 'picked3d', None)
        if clicked_pos is None:
            # Fallback: if vedo provides pick position under another attribute, try event.picked2d->project
            print("[DEBUG] No picked3d position from event. Click ignored.")
            return
        clicked_pos = np.array(clicked_pos)

        # First, check if clicking near an existing stenosis marker -> remove it
        all_stenosis = stenosis_objects_left + stenosis_objects_right
        if len(all_stenosis) > 0:
            sten_pos = [np.array(s.pos()) for s in all_stenosis]
            tree = cKDTree(sten_pos)
            idxs = tree.query_ball_point(clicked_pos, r=10.0)
            if idxs:
                dists = [np.linalg.norm(sten_pos[i] - clicked_pos) for i in idxs]
                chosen = idxs[int(np.argmin(dists))]
                chosen_obj = all_stenosis[chosen]
                removed_stenosis.append({
                    'position': chosen_obj.pos(),
                    'artery': 'left' if chosen_obj in stenosis_objects_left else 'right',
                    'time_removed': time.time(),
                    'color': chosen_obj.color,
                    'radius': getattr(chosen_obj, 'r', 2.0),
                    'alpha': getattr(chosen_obj, 'alpha', 0.6)
                })
                try:
                    event.plotter.remove(chosen_obj)
                except Exception:
                    pass
                if chosen_obj in stenosis_objects_left:
                    stenosis_objects_left.remove(chosen_obj)
                else:
                    stenosis_objects_right.remove(chosen_obj)
                event.plotter.render()
                print("Removed stenosis marker")
                return

        # If no stenosis clicked, pick nearest skeleton point using KDTree of all skeleton points
        if all_points is None or all_tree is None:
            print("No skeleton points loaded (all_points/all_tree missing).")
            return

        dist, idx = all_tree.query(clicked_pos)
        # threshold in mm/voxels? our all_points are voxel coordinates. For safety use a slightly larger threshold.
        # all_points are in voxel coords so dist is in voxels. Use threshold 20 voxels as in previous code.
        if dist > 20.0:
            print("Clicked too far from any vessel skeleton (dist {:.2f}).".format(dist))
            return

        nearest_pt = all_points[idx]

        # Determine whether nearest point belongs to left or right skeleton via membership checks
        from_left = False
        from_right = False
        if len(left_points) and np.any(np.all(nearest_pt == left_points, axis=1)):
            from_left = True
        if len(right_points) and np.any(np.all(nearest_pt == right_points, axis=1)):
            from_right = True

        # Prefer left if both (rare)
        if from_left:
            artery = 'left'
            points = left_points
            selected_points = selected_points_left
            visual_objects = visual_objects_left
            G = G_left
            dist_map = left_dist_map
            skeleton = left_skeleton
            spacing = left_spacing
            max_points = 3
        elif from_right:
            artery = 'right'
            points = right_points
            selected_points = selected_points_right
            visual_objects = visual_objects_right
            G = G_right
            dist_map = right_dist_map
            skeleton = right_skeleton
            spacing = right_spacing
            max_points = 2
        else:
            print("Clicked point not recognized as left/right skeleton point.")
            return

        # find index in that points array
        distances = np.linalg.norm(points - nearest_pt, axis=1)
        closest_idx = int(np.argmin(distances))

        if closest_idx in selected_points:
            print("Point already selected; choose another.")
            return

        if len(selected_points) >= max_points:
            print(f"Max points ({max_points}) reached for {artery}. Press 'r' to reset this artery.")
            return

        selected_points.append(closest_idx)
        color_seq = ['green', 'red', 'orange'] if artery == 'left' else ['green', 'red']
        pt_color = color_seq[len(selected_points) - 1]
        sphere = vedo.Sphere(pos=points[closest_idx], r=1.6, c=pt_color)
        sphere.pickable(True)
        event.plotter.add(sphere)
        visual_objects.append(sphere)
        event.plotter.render()
        print(f"[{artery.upper()}] Selected point idx={closest_idx}, coord={points[closest_idx].tolist()}")

        if (artery == 'left' and len(selected_points) == 3) or (artery == 'right' and len(selected_points) == 2):
            find_paths(event.plotter, artery, G, points, selected_points, visual_objects, dist_map, skeleton, spacing)

    except Exception as ex:
        print("Exception in handle_click:", ex)


def find_paths(plotter, artery, G, points, selected_points, visual_objects, dist_map, skeleton, spacing):
    global paths_data, last_computed_paths
    last_computed_paths.clear()

    start_idx = selected_points[0]
    end1_idx = selected_points[1]
    end2_idx = selected_points[2] if artery == 'left' and len(selected_points) > 2 else None

    def compute_path_and_visualize(sid, eid, label_id):
        try:
            path = nx.shortest_path(G, sid, eid, weight='weight')
        except nx.NetworkXNoPath:
            print(f"No path between {sid} and {eid}")
            return None, None, None
        path_pts = points[path]
        diams = [adaptive_diameter_calculation(points[i], dist_map, skeleton, points, spacing) for i in path]
        ln = vedo.Line(path_pts, c='grey')
        try:
            ln.cmap('jet', diams)
        except Exception:
            pass
        ln.lw(6)
        plotter.add(ln)
        visual_objects.append(ln)
        print(f"[{artery.upper()}] Path {label_id}: {len(path)} pts")
        return path, diams, path_pts

    # path 1
    path1, diams1, path1_pts = compute_path_and_visualize(start_idx, end1_idx, '1')
    diagnostics1 = None
    regions1 = []
    if path1 is not None and diams1 is not None:
        if USE_HYBRID:
            regions1, diagnostics1 = detect_hybrid_stenosis(diams1, spacing, **HYBRID_PARAMS)
        else:
            regions1, _ = detect_local_stenosis(diams1, spacing, window_mm=HYBRID_PARAMS['window_mm'],
                                                min_stenosis=HYBRID_PARAMS['sten_th'],
                                                min_length_pts=HYBRID_PARAMS['min_length_pts'])
        paths_data.append({"artery": artery, "id": "1", "diameters": diams1, "spacing": spacing})
        last_computed_paths.append(("1", path1, diams1, spacing, diagnostics1, regions1))
        if regions1:
            for region in regions1:
                sten_idx = path1[region['max_stenosis_idx']]
                sten_pt = points[sten_idx]
                sphere = vedo.Sphere(pos=sten_pt, r=2.8, c='red').alpha(0.5)
                sphere.pickable(True)
                plotter.add(sphere)
                if artery == 'left':
                    stenosis_objects_left.append(sphere)
                else:
                    stenosis_objects_right.append(sphere)
        else:
            print(f"[{artery.upper()}] No significant stenosis on path 1 (USE_HYBRID={USE_HYBRID})")

    # path 2 for left
    if artery == 'left' and end2_idx is not None:
        path2, diams2, path2_pts = compute_path_and_visualize(start_idx, end2_idx, '2')
        diagnostics2 = None
        regions2 = []
        if path2 is not None and diams2 is not None:
            if USE_HYBRID:
                regions2, diagnostics2 = detect_hybrid_stenosis(diams2, spacing, **HYBRID_PARAMS)
            else:
                regions2, _ = detect_local_stenosis(diams2, spacing, window_mm=HYBRID_PARAMS['window_mm'],
                                                    min_stenosis=HYBRID_PARAMS['sten_th'],
                                                    min_length_pts=HYBRID_PARAMS['min_length_pts'])
            paths_data.append({"artery": artery, "id": "2", "diameters": diams2, "spacing": spacing})
            last_computed_paths.append(("2", path2, diams2, spacing, diagnostics2, regions2))
            if regions2:
                for region in regions2:
                    sten_idx = path2[region['max_stenosis_idx']]
                    sten_pt = points[sten_idx]
                    sphere = vedo.Sphere(pos=sten_pt, r=2.2, c='red').alpha(0.5)
                    sphere.pickable(True)
                    plotter.add(sphere)
                    stenosis_objects_left.append(sphere)
            else:
                print(f"[{artery.upper()}] No significant stenosis on path 2 (USE_HYBRID={USE_HYBRID})")
            if path1 is not None and path2 is not None:
                common_len = min(len(path1), len(path2))
                branch_i = 0
                for i in range(common_len):
                    if path1[i] != path2[i]:
                        break
                    branch_i = i
                branch_pt = points[path1[branch_i]]
                bs = vedo.Sphere(pos=branch_pt, r=1.2, c='purple')
                plotter.add(bs)
                visual_objects.append(bs)

    plotter.render()


def reload_removed_stenosis(plotter):
    global removed_stenosis
    for st in removed_stenosis:
        s = vedo.Sphere(pos=st['position'], r=st.get('radius', 2.0)).color(st.get('color', 'red')).alpha(st.get('alpha', 0.6))
        if st['artery'] == 'left':
            stenosis_objects_left.append(s)
        else:
            stenosis_objects_right.append(s)
        plotter.add(s)
    removed_stenosis = []
    plotter.render()


def calculate_cadrads():
    total = len(stenosis_objects_left) + len(stenosis_objects_right)
    if total == 0:
        score = "CAD-RADS 0"
    elif total == 1:
        score = "CAD-RADS 3"
    elif total >= 3:
        score = "CAD-RADS 4B"
    else:
        score = "CAD-RADS 3"
    report = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_stenosis': total,
        'left_artery': len(stenosis_objects_left),
        'right_artery': len(stenosis_objects_right),
        'cadrads_score': score
    }
    cadrads_results.append(report)
    print("CAD-RADS report:", report)
    return report


# ------------------------
# Post-plot analysis
# ------------------------
def analyze_gradients_after_close(paths_data_local):
    if not paths_data_local:
        print("[INFO] No path data to analyze.")
        return
    os.makedirs("gradient_analysis", exist_ok=True)
    print("\n=== RUNNING POST-HOC PATH ANALYSIS ===")
    for path_info in paths_data_local:
        artery = path_info['artery']
        pid = path_info['id']
        diam = np.asarray(path_info['diameters'], dtype=float)
        spacing = np.asarray(path_info['spacing'], dtype=float)
        best = {'score': np.inf}
        def score_for_params(d, spacing, w, m, t):
            regs, _ = detect_local_stenosis(d, spacing=spacing, window_mm=w, min_stenosis=t, min_length_pts=m)
            g = np.gradient(d) if d.size else np.array([0.0])
            meang = float(np.mean(g))
            stdg = float(np.std(g))
            score = abs(meang + 0.01) + stdg + 0.02 * max(0, len(regs) - 3)
            return {'regs': regs, 'score': score}
        for w in [10, 15, 20, 25]:
            for m in [3, 4, 5]:
                for t in [15, 20, 25]:
                    res = score_for_params(diam, spacing, w, m, t)
                    if res['score'] < best['score']:
                        best = {'score': res['score'], 'regs': res['regs'], 'w': w, 'm': m, 't': t}
        grad = np.gradient(diam) if diam.size else np.array([])
        grad_smooth = gaussian_filter1d(grad, sigma=2) if grad.size else grad
        diam_smooth = gaussian_filter1d(diam, sigma=1) if diam.size else diam
        title = f"{artery}_path{pid}"
        outpath = os.path.join("gradient_analysis", f"{title}.png")
        fig, ax1 = mplt.subplots(figsize=(10,4))
        dist = np.arange(len(diam)) * float(np.mean(spacing)) if diam.size else np.array([])
        ax1.plot(dist, diam_smooth, 'b-', label="Diameter [mm]")
        ax1.set_xlabel("Path length [mm]")
        ax1.set_ylabel("Diameter [mm]", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(dist, grad_smooth, 'r--', label="Gradient (smoothed)")
        ax2.set_ylabel("Gradient", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        for r in best.get('regs', []):
            ax1.axvspan(dist[r['start_idx']] if dist.size else r['start_idx'],
                        dist[r['end_idx']] if dist.size else r['end_idx'],
                        color='red', alpha=0.12)
        ax1.set_title(f"{artery.upper()} - Path {pid}   best_params: w={best.get('w')}, m={best.get('m')}, t={best.get('t')}")
        fig.tight_layout()
        mplt.savefig(outpath, dpi=150)
        mplt.close(fig)
        print(f"[SAVED] {outpath}")
    print("\n=== POST-HOC ANALYSIS DONE ===")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    # Update these paths to your NRRD files before running
    left_path = r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_left.nrrd"
    right_path = r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_right.nrrd"

    left_mask, left_skeleton, left_points, left_dist_map, left_spacing = load_artery(left_path, "Left")
    right_mask, right_skeleton, right_points, right_dist_map, right_spacing = load_artery(right_path, "Right")

    G_left, left_point_to_id = build_graph(left_points, left_skeleton, left_dist_map, left_spacing, "Left", neighbor_radius_mm=3.0)
    G_right, right_point_to_id = build_graph(right_points, right_skeleton, right_dist_map, right_spacing, "Right", neighbor_radius_mm=3.0)

    left_mesh = vedo.Volume(left_mask).isosurface().c('lightgreen').alpha(0.18) if left_mask is not None else None
    left_skel = vedo.Points(left_points, r=3, c='darkgreen') if len(left_points) else None
    right_mesh = vedo.Volume(right_mask.astype(np.uint8)).isosurface().c('lightblue').alpha(0.18) if (right_mask is not None and right_mask.sum() > 0) else None
    right_skel = vedo.Points(right_points, r=3, c='darkblue') if len(right_points) else None

    # prepare combined KDTree for hover and proximity queries (voxel coords)
    if len(left_points) and len(right_points):
        all_points = np.vstack([left_points, right_points])
    elif len(left_points):
        all_points = left_points.copy()
    elif len(right_points):
        all_points = right_points.copy()
    else:
        all_points = np.zeros((0, 3))
    all_tree = cKDTree(all_points) if len(all_points) else None

    plt = vedo.Plotter(title="Kliknij 3 punkty: start, koniec1, koniec2\n[h]-toggle hybrid [v]-save diag [L]-reload stenosis [R]-reset [C]-CAD-RADS [Q]-quit",
                       axes=1, bg='white', size=(1000, 800))

    base_objs = []
    if left_mesh is not None:
        base_objs.append(left_mesh)
    if left_skel is not None:
        base_objs.append(left_skel)
    if right_mesh is not None:
        base_objs.append(right_mesh)
    if right_skel is not None:
        base_objs.append(right_skel)
    plt.add(base_objs)

    plt.add_callback('LeftButtonPress', handle_click)

    def keypress_cb(event):
        global USE_HYBRID, DEBUG_SAVE_DIAGNOSTICS
        k = event.keypress.lower() if event.keypress else ''
        if k == 'r':
            reset_selection(event.plotter, artery='all')
        elif k == 'l':
            reload_removed_stenosis(event.plotter)
        elif k == 'c':
            calculate_cadrads()
        elif k == 'q':
            event.plotter.close()
        elif k == 'h':
            USE_HYBRID = not USE_HYBRID
            print(f"Hybrid detection toggled -> USE_HYBRID = {USE_HYBRID}")
        elif k == 'v':
            if not last_computed_paths:
                print("No last paths to save diagnostics for.")
                return
            for (pid, path_idxs, diams, spacing, diagnostics, regions) in last_computed_paths:
                title = f"diag_{pid}_{time.strftime('%Y%m%d_%H%M%S')}"
                save_diagnostics_plot(diams, spacing, diagnostics, regions, title)
            print("Saved diagnostics for last computed path(s).")

    plt.add_callback('KeyPress', keypress_cb)

    plt.show(interactive=True, resetcam=True)

    analyze_gradients_after_close(paths_data)

    print("Script finished.")