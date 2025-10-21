import nrrd
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from skimage.filters import gaussian
import vedo
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time

def debug_print_header(header):
    # Bezpieczne odczytanie spacing / space directions
    if 'space directions' in header:
        try:
            sd = header['space directions']
            sd_norms = [np.linalg.norm(v) if v is not None else None for v in sd]
            print(" header: space directions norms:", sd_norms)
        except Exception as e:
            print(" header: problem reading space directions:", e)
    if 'spacing' in header:
        print(" header: spacing:", header['spacing'])
    if 'sizes' in header:
        print(" header: sizes:", header['sizes'])
    if 'kinds' in header:
        print(" header: kinds:", header['kinds'])

def load_artery(filepath, artery_name, do_postproc=True, min_component_voxels=100):
    """
    Bezpieczne ładowanie NRRD -> wybór największego komponentu -> skeletonizacja.
    do_postproc: czy wykonać binary_closing/gładzenie
    min_component_voxels: jeżeli największy komponent jest mniejszy niż to, ostrzeż i nie wybieraj go
    """
    try:
        data, header = nrrd.read(filepath)
        print(f"=== Loading: {filepath} ===")
        print("data shape:", data.shape)
        # pokaż kilka informacji z headera
        debug_print_header(header)

        unique_vals = np.unique(data)
        print("unique values (sample up to 20):", unique_vals[:20], " total unique:", len(unique_vals))

        # 1) Jeżeli plik to labelmap (np. wartości 0,1,2,3) — spróbuj wybrać odpowiednią etykietę
        # Domyślnie: jeśli tylko 0 i 1 -> traktuj jako binary maskę.
        if np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [0, 1]):
            mask = (data > 0).astype(np.uint8)
        else:
            # Jeśli jest wiele etykiet, wybierz największą nie-zero etykietę
            # policz voxele per label
            labels, counts = np.unique(data, return_counts=True)
            # usuń label 0 (tło)
            idx_nonzero = labels != 0
            labels_nz = labels[idx_nonzero]
            counts_nz = counts[idx_nonzero]
            if len(labels_nz) == 0:
                print("WARNING: Brak niezerowych etykiet w pliku -> tworzenie maski (data>0)")
                mask = (data > 0).astype(np.uint8)
            else:
                # wybierz label z największą liczbą voxeli
                max_label = labels_nz[np.argmax(counts_nz)]
                print(f"Detected multiple labels. Selecting largest label {max_label} with {np.max(counts_nz)} voxels")
                mask = (data == max_label).astype(np.uint8)

        print("Initial mask voxels:", int(np.sum(mask)))

        # 2) Usuń małe komponenty -> wybierz największy spójny komponent
        labeled = label(mask)
        props = regionprops(labeled)
        if len(props) == 0:
            print("WARNING: mask has no components after labeling.")
            skeleton = np.zeros_like(mask, dtype=np.uint8)
            points = np.argwhere(skeleton > 0)
            spacing = extract_spacing_from_header_safe(header)
            dist_map = ndimage.distance_transform_edt(mask.astype(np.float64), sampling=spacing)
            return mask, skeleton, points, dist_map, spacing

        largest = max(props, key=lambda p: p.area)
        if largest.area < min_component_voxels:
            print(f"WARNING: largest component is very small ({largest.area} voxels). You may be using wrong label/threshold.")
        # rebuild mask as only largest component
        mask_largest = (labeled == largest.label).astype(np.uint8)
        print("Largest component voxels:", int(np.sum(mask_largest)))

        # 3) opcjonalne post-processing: zamknięcie morfologiczne + gładzenie
        if do_postproc:
            from scipy.ndimage import binary_closing
            mask_proc = binary_closing(mask_largest, structure=np.ones((3,3,3)), iterations=1)
            # delikatne wygładzenie progu
            mask_proc = gaussian(mask_proc.astype(float), sigma=0.4) > 0.5
            mask_proc = mask_proc.astype(np.uint8)
        else:
            mask_proc = mask_largest

        # skeletonize expects boolean
        skeleton = skeletonize(mask_proc > 0).astype(np.uint8)
        points = np.argwhere(skeleton > 0)
        print(f"{artery_name} artery: {len(points)} skeleton points after preprocessing")

        spacing = extract_spacing_from_header_safe(header)
        print(f"Spacing (mm/voxel): {spacing}")

        # distance transform (use mask_proc to compute EDT)
        mask_float = mask_proc.astype(np.float64)
        dist_map = ndimage.distance_transform_edt(mask_float, sampling=spacing)

        return mask_proc, skeleton, points, dist_map, spacing

    except Exception as e:
        print(f"Błąd ładowania {artery_name} tętnicy: {str(e)}")
        return None, None, np.array([]), None, np.array([1,1,1])

def extract_spacing_from_header_safe(header):
    # bezpieczne wyciągnięcie spacing (kopiuj z Twojej wcześniejszej funkcji)
    try:
        if 'space directions' in header:
            spacing = np.array([np.linalg.norm(v) for v in header['space directions']])
            spacing[np.isnan(spacing)] = 1.0
            spacing[spacing == 0] = 1.0
            return spacing
        if 'spacing' in header:
            sp = np.array(header['spacing'], dtype=float)
            sp[np.isnan(sp)] = 1.0
            sp[sp == 0] = 1.0
            return sp
    except Exception:
        pass
    # fallback
    print("WARNING: could not read spacing -> using 1.0 mm/voxel fallback")
    return np.array([1.0, 1.0, 1.0])

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

def detect_local_stenosis(diameters, spacing, window_mm=10, min_stenosis=30, min_length_pts=3):
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
# ... (previous imports remain the same)

selected_points_left = []
selected_points_right = []
visual_objects_left = []
visual_objects_right = []
stenosis_objects_left = []  # Lista do śledzenia znaczników zwężeń
stenosis_objects_right = []  # Lista do śledzenia znaczników zwężeń
removed_stenosis = []  # Przechowuje usunięte zwężenia
cadrads_results = []   # Przechowuje wyniki analizy

# Marker kursora
cursor_marker = None
cursor_text = None

# KDTree dla szybkiego wyszukiwania najbliższego punktu szkieletu (łączymy obie tętnice)
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


def find_nearest_node(points, query_point):
    # points: N x 3 array, query_point: 3-element
    if len(points) == 0:
        return None, None
    scaled = points  # już w indeksach (voxel coords)
    dists = np.linalg.norm(scaled - query_point, axis=1)
    idx = np.argmin(dists)
    return idx, points[idx]


def nearest_node_index_on_graph(points, query_point):
    # zwraca indeks i w tablicy points oraz index przeskalowany w all_points jeśli potrzeba
    idx, pt = find_nearest_node(points, query_point)
    return idx


def handle_click(event):
    global removed_stenosis
    if not event.actor or event.keypress:
        return

    clicked_pos = event.picked3d
    if clicked_pos is None:
        return

    # 1. PRIORYTET: Znajdź najbliższy znacznik zwężenia (nawet głęboko położony)
    all_stenosis = stenosis_objects_left + stenosis_objects_right
    if all_stenosis:
        stenosis_positions = [np.array(s.pos()) for s in all_stenosis]
        tree = cKDTree(stenosis_positions)
        indices = tree.query_ball_point(clicked_pos, r=10.0)

        if indices:
            distances = [np.linalg.norm(np.array(all_stenosis[i].pos()) - clicked_pos) for i in indices]
            closest_idx = indices[np.argmin(distances)]
            closest_stenosis = all_stenosis[closest_idx]

            # ZAPISZ USUNIĘTY ZNACZNIK (POPRAWIONA WERSJA)
            removed_stenosis.append({
                'position': closest_stenosis.pos(),
                'artery': 'left' if closest_stenosis in stenosis_objects_left else 'right',
                'time_removed': time.time(),
                'color': closest_stenosis.color,  # Pobierz kolor
                'radius': closest_stenosis.radius,  # Pobierz promień
                'alpha': closest_stenosis.alpha  # Pobierz przezroczystość
            })

            plt.remove(closest_stenosis)
            if closest_stenosis in stenosis_objects_left:
                stenosis_objects_left.remove(closest_stenosis)
            else:
                stenosis_objects_right.remove(closest_stenosis)
            print(f"Usunięto znacznik zwężenia w {closest_stenosis.pos()}")
            plt.render()
            return


    # 2. Jeśli nie znaleziono znacznika, kontynuuj normalny wybór punktów
    distances_left = np.linalg.norm(left_points - clicked_pos, axis=1) if len(left_points) else [np.inf]
    distances_right = np.linalg.norm(right_points - clicked_pos, axis=1) if len(right_points) else [np.inf]
    min_left_dist = np.min(distances_left)
    min_right_dist = np.min(distances_right)

    # Reszta funkcji pozostaje bez zmian...

    # Sprawdź czy kliknięto wystarczająco blisko którejś tętnicy
    if min_left_dist > 5.0 and min_right_dist > 5.0:
        print("Kliknięto zbyt daleko od tętnic. Wybierz punkt bliżej szkieletu.")
        return

    # Wybierz tętnicę na podstawie najbliższego punktu
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

    # 3. Sprawdź czy można dodać nowy punkt do wybranej tętnicy
    if len(selected_points) >= max_points:
        print(f"Uwaga: Osiągnięto maks. liczbę punktów ({max_points}) dla tętnicy {artery}.")
        print("Kliknij 'L' aby zresetować lewą tętnicę, 'P' dla prawej, lub 'R' aby zresetować wszystko.")
        return

    # 4. Dodaj nowy punkt
    distances = np.linalg.norm(points - clicked_pos, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]

    # Sprawdź czy punkt nie jest już wybrany
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

    # 5. Jeśli osiągnięto wymaganą liczbę punktów, znajdź ścieżki
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

        regions1, stenosis_values1 = detect_local_stenosis(
            diams1,
            spacing=spacing,
            window_mm=8,
            min_stenosis=20,
            min_length_pts=4
        )

        if regions1:
            enhanced_visualization(path1_points, diams1, regions1, artery + " - Ścieżka 1")
            for region in regions1:
                sten_point = points[path1[region['max_stenosis_idx']]]
                sten_sphere = vedo.Sphere(pos=sten_point, r=3.0, c='red').alpha(0.5)
                sten_sphere.pickable(True)  # Umożliwia kliknięcie przez inne obiekty
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

            regions2, stenosis_values2 = detect_local_stenosis(
                diams2,
                spacing=spacing,
                window_mm=8,
                min_stenosis=20,
                min_length_pts=4
            )

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
    # Przywróć poprzednie kolory wszystkim obiektom
    for obj in stenosis_objects_left + stenosis_objects_right:
        obj.color('red').alpha(0.7)  # Czerwony z przezroczystością

    # Podświetl aktualny obiekt
    if event.actor and (event.actor in stenosis_objects_left or event.actor in stenosis_objects_right):
        event.actor.color('yellow').alpha(0.9)  # Żółty z mniejszą przezroczystością
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

    # Analizuj obecne zwężenia
    current_stenosis = []
    for obj in stenosis_objects_left:
        current_stenosis.append({'artery': 'left', 'position': obj.pos()})
    for obj in stenosis_objects_right:
        current_stenosis.append({'artery': 'right', 'position': obj.pos()})

    # Tutaj dodaj rzeczywistą logikę klasyfikacji CAD-RADS
    # Przykład uproszczony:
    cadrads_score = "CAD-RADS 0"
    if len(current_stenosis) > 0:
        if len(current_stenosis) >= 3:
            cadrads_score = "CAD-RADS 4B"
        elif len(current_stenosis) >= 1:
            cadrads_score = "CAD-RADS 3"

    # Generuj raport
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

    # Wizualizacja raportu
    report_text = f"""
    ===== RAPORT CAD-RADS =====
    Data: {latest_report['timestamp']}
    Laczna liczba zwezen: {latest_report['total_stenosis']}
    - Lewa tetnica: {latest_report['left_artery']}
    - Prawa tetnica: {latest_report['right_artery']}
    Ocena CAD-RADS: {latest_report['cadrads_score']}
    """

    print(report_text)

    # Możesz też wyświetlić to w oknie Vedo
    txt = vedo.Text2D(report_text, pos='top-left', c='k', bg='y', alpha=0.8)
    plt.add(txt).render()

def on_mouse_move(event):
    global cursor_marker, cursor_text
    # Przywróć domyślne kolory znaczników zwężeń
    for obj in stenosis_objects_left + stenosis_objects_right:
        try:
            obj.color('red').alpha(0.6)
        except Exception:
            pass

    # Podświetl aktualny obiekt (jeśli nad nim)
    if event.actor and (event.actor in stenosis_objects_left or event.actor in stenosis_objects_right):
        event.actor.color('yellow').alpha(0.9)

    picked3d = event.picked3d
    if picked3d is None:
        return

    # Znajdź najbliższy punkt w all_points
    global all_tree
    if all_tree is None:
        return
    dist, idx = all_tree.query(picked3d)
    if dist > 10.0:
        # jeśli zbyt daleko - ukryj marker
        if cursor_marker:
            plt.remove(cursor_marker)
            cursor_marker = None
        if cursor_text:
            plt.remove(cursor_text)
            cursor_text = None
        plt.render()
        return

    nearest_pt = all_points[idx]

    # Znajdź do której tętnicy należy ten punkt
    if len(left_points) and np.any(np.all(nearest_pt == left_points, axis=1)):
        pt_idx = np.where(np.all(nearest_pt == left_points, axis=1))[0][0]
        diam = adaptive_diameter_calculation(left_points[pt_idx], left_dist_map, left_skeleton, left_points, left_spacing)
    else:
        pt_idx = np.where(np.all(nearest_pt == right_points, axis=1))[0][0]
        diam = adaptive_diameter_calculation(right_points[pt_idx], right_dist_map, right_skeleton, right_points, right_spacing)

    # Aktualizuj marker i tekst
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
    if event.keypress == 'r':  # Reset
        reset_selection('all')
    elif event.keypress == 'l':  # Załaduj usunięte zwężenia
        reload_removed_stenosis()
    elif event.keypress == 'c':  # Oblicz CAD-RADS
        calculate_cadrads()
        show_cadrads_report()

plt.add_callback('LeftButtonPress', handle_click)
plt.add_callback('MouseMove', on_mouse_move)
plt.add_callback("KeyPress", handle_keypress)
plt.interactive().close()