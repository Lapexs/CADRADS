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


import matplotlib
matplotlib.use('Agg')  # wymusza osobne okno Matplotlib, niezależne od Vedo

paths_data = []


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

def rolling_average(arr, window):
    if window < 2:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='same')

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

def detect_local_stenosis(diameters, spacing, window_mm=15, min_stenosis=30, min_length_pts=5):
    """
    Detekcja lokalnych zwężeń na podstawie zmian średnicy naczynia.
    Dodano filtrację fałszywych zwężeń na podstawie gradientu i głębokości spadku.
    """
    diameters = np.array(diameters)
    n = len(diameters)
    mean_spacing = np.mean(spacing)
    window_pts = max(2, int(window_mm / mean_spacing))

    stenosis_values = []
    for i in range(n):
        left = max(0, i - window_pts)
        right = min(n, i + window_pts)

        region = diameters[left:i] if i > left else diameters[i+1:right]
        if len(region) < 2:
            stenosis_values.append(0)
            continue

        ref_diam = np.median(region)
        stenosis = (1 - diameters[i] / ref_diam) * 100 if ref_diam > 0 else 0
        stenosis = max(0, stenosis)
        stenosis_values.append(stenosis)

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

    grad = np.gradient(diameters)

    filtered_regions = []
    for region in regions:
        start, end = region['start_idx'], region['end_idx']


        region_grad_mean = np.mean(grad[start:end])

        diameter_drop = np.max(diameters[start:end]) - np.min(diameters[start:end])

        if region_grad_mean < -0.1:   
            continue
        if diameter_drop < 0.3:    
            continue

        filtered_regions.append(region)

    return filtered_regions, stenosis_values


def enhanced_visualization(path_points, diameters, stenosis_regions, artery_name):
    distance_mm = np.arange(len(diameters)) * 0.5  # Przybliżona odległość
    window_size = 9
    diameters_smooth = rolling_average(np.array(diameters), window_size)
    print(f"\n=== DETAILED ANALYSIS - {artery_name} ===")
    print(f"Mean diameter: {np.mean(diameters_smooth):.2f} mm")
    print(f"Min diameter: {np.min(diameters_smooth):.2f} mm")
    print(f"Diameter variability (std): {np.std(diameters_smooth):.2f} mm")
    print(f"Number of stenosis regions: {len(stenosis_regions)}")
    for i, region in enumerate(stenosis_regions, 1):
        print(f"Region {i}: stenosis {region['max_stenosis']:.1f}%, length {region['length']} points")

# --- Wczytanie danych ---
left_mask, left_skeleton, left_points, left_dist_map, left_spacing = load_artery(
    r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_left.nrrd", "Left")

right_mask, right_skeleton, right_points, right_dist_map, right_spacing = load_artery(
    r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_right.nrrd", "Right")


# --- Tworzenie grafów ---
G_left, left_point_to_id = build_graph(left_points, left_skeleton, left_dist_map, left_spacing, "Left")
G_right, right_point_to_id = build_graph(right_points, right_skeleton, right_dist_map, right_spacing, "Right")


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

def detect_local_stenosis_with_grad(diameters, spacing,
                                    window_mm=15, min_stenosis=30, min_length_pts=5,
                                    use_gradient=False, grad_thresh=-0.1,
                                    smooth_sigma_pts=1.0, combine_with_and=False):
    """
    Wersja rozszerzona: uwzględnia opcjonalnie spadek gradientu.
    - diameters: list/np.array średnic [mm]
    - spacing: 3-element array voxel spacing (mm) -> używamy mean_spacing
    - window_mm: okno do lokalnej referencji (mm)
    - min_stenosis: procentowy próg spadku [%]
    - min_length_pts: minimalna liczba punktów aby region uznać za zwężenie
    - use_gradient: jeśli True dodajemy warunek na gradient
    - grad_thresh: próg gradientu (ujemny), np. -0.1 (mm/mm)
    - smooth_sigma_pts: sigma do gaussian_filter1d użyte do wygładzenia przed gradientem (w punktach)
    - combine_with_and: jeśli True -> wymagamy (pct_drop >= min_stenosis) AND (grad <= grad_thresh)
                       jeśli False -> OR (bardziej czułe)
    Returns: regions, diagnostics (pct_drop, grad, d_smooth, cond_pct, cond_grad, cond_combined)
    """

    diams = np.array(diameters, dtype=float)
    n = len(diams)
    mean_spacing = float(np.mean(spacing)) if np.ndim(spacing) else float(spacing)
    window_pts = max(1, int(round(window_mm / mean_spacing)))

    # 1) Wygładź średnice do policzenia gradientu (małe sigma np.1-3)
    if smooth_sigma_pts and smooth_sigma_pts > 0:
        d_smooth = gaussian_filter1d(diams, sigma=smooth_sigma_pts, mode='nearest')
    else:
        d_smooth = diams.copy()

    # 2) gradient (mm per mm)
    grad = np.gradient(d_smooth) / mean_spacing

    # 3) percent drop (lokalna referencja proksymalna tak jak wcześniej)
    pct_drop = np.zeros(n, dtype=float)
    for i in range(n):
        left = max(0, i - window_pts)
        right = min(n, i + window_pts + 1)
        # proksymalna referencja
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

    # 4) warunki
    cond_pct = pct_drop >= min_stenosis
    cond_grad = grad <= grad_thresh if use_gradient else np.zeros_like(cond_pct, dtype=bool)

    if use_gradient:
        if combine_with_and:
            cond_combined = cond_pct & cond_grad
        else:
            cond_combined = cond_pct | cond_grad
    else:
        cond_combined = cond_pct

    # 5) grupuj ciągłe regiony i filtruj po minimalnej długości
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


def plot_diameter_gradient_with_regions(diams, spacing, title, diagnostics=None, regions=None):
    mean_grad, std_grad, grad = evaluate_gradient(diams, spacing)
    dist = np.arange(len(diams)) * np.mean(spacing)

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

        pct = diagnostics.get('pct_drop')
        if pct is not None:
            ax1.plot(dist, diagnostics['d_smooth'], color='cyan', alpha=0.6, label='Smoothed Diam')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(dist, pct, color='magenta', linestyle=':', label='% drop')
            ax1_twin.set_ylabel('% drop', color='magenta')
            ax1_twin.tick_params(axis='y', labelcolor='magenta')

        if regions:
            for reg in regions:
                start = reg['start_idx'] * np.mean(spacing)
                end = reg['end_idx'] * np.mean(spacing)
                ax1.axvspan(start, end, color='red', alpha=0.15)

    ax1.set_title(title)
    ax1.grid(True)
    fig.tight_layout()
    mplt.close(fig)


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
        find_paths(artery, G, points, selected_points, visual_objects, dist_map, skeleton, spacing)


def find_paths(artery, G, points, selected_points, visual_objects, dist_map, skeleton, spacing):
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

        regions1, diag1 = detect_local_stenosis_with_grad(
            diams1, spacing,
            window_mm=15, min_stenosis=30, min_length_pts=6,
            use_gradient=True, grad_thresh=-0.1,
            smooth_sigma_pts=1.5, combine_with_and=False
        )

        print(f"\n[{artery.upper()}] Analiza gradientu dla ścieżki 1:")
        opt_params = optimize_stenosis_parameters(diams1, spacing, visualize=True)

        threading.Thread(target=plot_diameter_gradient_with_regions(diams1, spacing, artery + " - Path 1", diagnostics=diag1, regions=regions1)).start()

        paths_data.append({
            "artery": artery,
            "id": "1",
            "diameters": diams1,
            "spacing": spacing
        })


        if regions1:
            enhanced_visualization(path1_points, diams1, regions1, artery + " - Ścieżka 1")
            for region in regions1:
                sten_point = points[path1[region['max_stenosis_idx']]]
                sten_sphere = vedo.Sphere(pos=sten_point, r=3.0, c='red').alpha(0.5)
                sten_sphere.pickable(True) 
                plt.add(sten_sphere)
                if artery == 'left':
                    stenosis_objects_left.append(sten_sphere)
                else:
                    stenosis_objects_right.append(sten_sphere)
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

            regions2, diag2 = detect_local_stenosis_with_grad(
                diams2, spacing,
                window_mm=15, min_stenosis=30, min_length_pts=6,
                use_gradient=True, grad_thresh=-0.1,
                smooth_sigma_pts=1.5, combine_with_and=False
            )
                    # === TEST PARAMETRÓW I GRADIENT ===
            print(f"\n[{artery.upper()}] Analiza gradientu dla ścieżki 2:")
            opt_params2 = optimize_stenosis_parameters(diams2, spacing, visualize=True)
            threading.Thread(target=plot_diameter_gradient_with_regions(diams2, spacing, artery + " - Path 1", diagnostics=diag2, regions=regions2)).start()

            paths_data.append({
                "artery": artery,
                "id": "2",
                "diameters": diams2,
                "spacing": spacing
            })

            if regions2:
                enhanced_visualization(path2_points, diams2, regions2, artery + " - Ścieżka 2")
                for region in regions2:
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

def calculate_cadrads():
    global cadrads_results

    current_stenosis = []
    for obj in stenosis_objects_left:
        current_stenosis.append({'artery': 'left', 'position': obj.pos()})
    for obj in stenosis_objects_right:
        current_stenosis.append({'artery': 'right', 'position': obj.pos()})

    cadrads_score = "CAD-RADS 0"
    if len(current_stenosis) > 0:
        if len(current_stenosis) >= 3:
            cadrads_score = "CAD-RADS 4B"
        elif len(current_stenosis) >= 1:
            cadrads_score = "CAD-RADS 3"

    report = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_stenosis': len(current_stenosis),
        'left_artery': len(stenosis_objects_left),
        'right_artery': len(stenosis_objects_right),
        'cadrads_score': cadrads_score,
        'details': current_stenosis
    }

    cadrads_results.append(report)
    return report

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
    elif event.keypress == 'c':   
        calculate_cadrads()
        show_cadrads_report()

plt.add_callback('LeftButtonPress', handle_click)
plt.add_callback('MouseMove', on_mouse_move)
plt.add_callback("KeyPress", handle_keypress)
plt.interactive().close()

import os
from scipy.ndimage import gaussian_filter1d

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

        #for r in best["regs"]:
         #   ax1.axvspan(dist[r['start_idx']], dist[r['end_idx']], color='red', alpha=0.15)

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
import os

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
