import nrrd
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import vedo
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d, binary_closing, generate_binary_structure, maximum_filter
from scipy.spatial import KDTree
from scipy.signal import savgol_filter


def load_artery(filepath, artery_name):
    try:
        data, header = nrrd.read(filepath)
        mask = (data > 0).astype(np.uint8)

        # Wygładzanie maski przed skeletonizacją
        mask = binary_closing(mask, structure=np.ones((3, 3, 3)))

        skeleton = skeletonize(mask).astype(np.uint8)
        points = np.argwhere(skeleton > 0)
        print(f"{artery_name} tętnica: {len(points)} punktów szkieletu")

        # Odczytaj spacing z nagłówka
        if 'space directions' in header:
            spacing = np.array([np.linalg.norm(v) for v in header['space directions']])
        elif 'spacing' in header:
            spacing = np.array(header['spacing'])
        else:
            spacing = np.array([1.0, 1.0, 1.0])
            print(f"UWAGA: Brak spacing w pliku NRRD dla {artery_name}, przyjęto 1mm/voxel")

        # Oblicz transformatę odległości z uwzględnieniem spacing
        spacing = np.array(spacing, dtype=np.float64)
        mask_float = mask.astype(np.float64)
        dist_map = ndimage.distance_transform_edt(mask_float, sampling=spacing)

        return mask, skeleton, points, dist_map, spacing
    except Exception as e:
        print(f"Błąd ładowania {artery_name} tętnicy: {str(e)}")
        return None, None, np.array([]), None, np.array([1, 1, 1])


def enhanced_local_diameter(p, dist_map, spacing, radius_mm=1.5):
    z, y, x = p

    rz = int(np.ceil(radius_mm / spacing[0]))
    ry = int(np.ceil(radius_mm / spacing[1]))
    rx = int(np.ceil(radius_mm / spacing[2]))

    # Wyciągnięcie regionu
    region = dist_map[max(0, z - rz):z + rz + 1,
                      max(0, y - ry):y + ry + 1,
                      max(0, x - rx):x + rx + 1]

    if region.size == 0:
        return 0.0

    valid = region[region > 0.1]
    if len(valid) < 3:
        return 2 * dist_map[z, y, x]

    return 2 * np.median(valid)



def extract_centerline(points, dist_map, spacing):
    """Ulepszona ekstrakcja linii środkowej z lokalnych maksimów"""
    # Znajdź lokalne maksima transformaty odległości
    neighborhood = generate_binary_structure(3, 2)
    local_max = maximum_filter(dist_map, footprint=neighborhood) == dist_map
    local_max[dist_map < 0.5] = 0  # Próg minimalnego promienia

    centerline_points = np.argwhere(local_max)

    # Połącz punkty w graf z uwzględnieniem spacing
    tree = KDTree(centerline_points)
    graph = nx.Graph()

    for i, point in enumerate(centerline_points):
        # Znajdź sąsiadów w promieniu 2mm (uwzględniając spacing)
        radius = 3.5 / np.min(spacing)
        neighbors = tree.query_ball_point(point, r=radius)
        for n in neighbors:
            if i != n:
                # Waga krawędzi jako odległość fizyczna w mm
                dist = np.linalg.norm((centerline_points[i] - centerline_points[n]) * spacing)
                graph.add_edge(i, n, weight=dist)

    return centerline_points, graph


def build_graph(points, spacing, artery_name):
    """Ulepszona funkcja budowy grafu z uwzględnieniem spacing"""
    if len(points) == 0:
        print(f"Brak punktów do budowy grafu dla {artery_name} tętnicy")
        return nx.Graph(), {}

    point_to_id = {tuple(p): i for i, p in enumerate(points)}
    G = nx.Graph()

    for i, point in enumerate(points):
        point = tuple(point)
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    neighbor = (point[0] + dz, point[1] + dy, point[2] + dx)
                    if neighbor in point_to_id:
                        # Oblicz odległość fizyczną z uwzględnieniem spacing
                        dist = np.sqrt((dz * spacing[0]) ** 2 + (dy * spacing[1]) ** 2 + (dx * spacing[2]) ** 2)
                        G.add_edge(point_to_id[point], point_to_id[neighbor], weight=dist)

    print(f"{artery_name} tętnica: graf ma {G.number_of_nodes()} węzłów i {G.number_of_edges()} krawędzi")
    return G, point_to_id


def advanced_stenosis_analysis(diameters, spacing, min_stenosis=30):
    """Ulepszona analiza zwężeń z automatycznym wykrywaniem segmentów referencyjnych"""
    # Wygładzenie profilu średnic
    window_size = max(5, int(5.0 / np.mean(spacing)))  # Dostosuj do spacing
    smoothed_diameters = savgol_filter(diameters,
                                       window_length=window_size,
                                       polyorder=2)

    # Znajdź minima lokalne
    minima_indices = []
    for i in range(2, len(smoothed_diameters) - 2):
        if (smoothed_diameters[i] < smoothed_diameters[i - 1] and
                smoothed_diameters[i] < smoothed_diameters[i + 1]):
            minima_indices.append(i)

    # Analizuj każde potencjalne zwężenie
    stenosis_results = []
    for idx in minima_indices:
        # Szukaj referencji w zdrowym segmencie (proxymalnie)
        ref_start = max(0, idx - 20)
        ref_segment = smoothed_diameters[ref_start:idx]

        if len(ref_segment) < 5:
            continue

        ref_diameter = np.percentile(ref_segment, 90)

        # Pomijaj bardzo małe naczynia
        if ref_diameter < 1.2:
            continue

        stenosis = (1 - smoothed_diameters[idx] / ref_diameter) * 100

        if stenosis >= min_stenosis:
            # Oblicz długość zwężenia
            length = calculate_stenosis_length(smoothed_diameters, idx, spacing)

            stenosis_results.append({
                'index': idx,
                'ref_diameter': ref_diameter,
                'min_diameter': smoothed_diameters[idx],
                'stenosis_percent': stenosis,
                'length_mm': length
            })

    return smoothed_diameters, stenosis_results


def classify_cadrads(stenosis_percent):
    """Klasyfikacja CAD-RADS z dodatkowymi kryteriami"""
    if stenosis_percent < 25:
        return "CAD-RADS 1 (łagodne)"
    elif stenosis_percent < 50:
        return "CAD-RADS 2 (umiarkowane)"
    elif stenosis_percent < 70:
        return "CAD-RADS 3 (znaczne)"
    elif stenosis_percent < 90:
        return "CAD-RADS 4 (ciężkie)"
    else:
        return "CAD-RADS 5 (krytyczne)"


# Ładowanie danych
left_mask, left_skeleton, left_points, left_dist_map, left_spacing = load_artery(
    r"C:\Users\student.VIRMED\Desktop\Slicer_JM\CADRADS 1\Segmentation_left.nrrd", "Lewa")
right_mask, right_skeleton, right_points, right_dist_map, right_spacing = load_artery(
    r"C:\Users\student.VIRMED\Desktop\Slicer_JM\CADRADS 1\Segmentation_right.nrrd", "Prawa")

# Ekstrakcja linii środkowej
left_centerline, left_centerline_graph = extract_centerline(left_points, left_dist_map, left_spacing)
right_centerline, right_centerline_graph = extract_centerline(right_points, right_dist_map, right_spacing)

# Budowa grafów
G_left, left_point_to_id = build_graph(left_points, left_spacing, "Lewa")
G_right, right_point_to_id = build_graph(right_points, right_spacing, "Prawa")

# Wizualizacja
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


# [Pozostałe funkcje (reset_selection, handle_click, find_paths, handle_keypress)
# pozostają bez zmian, ale będą używały nowych ulepszonych funkcji]

def find_paths(artery):
    if artery == 'left':
        G = G_left
        points = left_points
        selected_points = selected_points_left
        visual_objects = visual_objects_left
        dist_map = left_dist_map
        spacing = left_spacing
    else:
        G = G_right
        points = right_points
        selected_points = selected_points_right
        visual_objects = visual_objects_right
        dist_map = right_dist_map
        spacing = right_spacing

    if (artery == 'left' and len(selected_points) < 3) or (artery == 'right' and len(selected_points) < 2):
        print(f"Zbyt mało punktów dla {artery} tętnicy.")
        return

    start_idx = selected_points[0]
    end1_idx = selected_points[1]
    end2_idx = selected_points[2] if artery == 'left' else None

    def path_diameter_report(path, points, dist_map, spacing):
        diameters = []
        for idx in path:
            p = points[idx]
            diam = 2 * enhanced_local_diameter(p, dist_map, spacing, window=5)
            diameters.append(diam)
        return diameters

    try:
        path1 = nx.shortest_path(G, start_idx, end1_idx, weight='weight')
        diams1 = path_diameter_report(path1, points, dist_map, spacing)

        # Zaawansowana analiza zwężeń
        smoothed_diams1, stenosis_results1 = advanced_stenosis_analysis(diams1, spacing)

        # Wizualizacja
        path1_points = points[path1]
        line1 = vedo.Line(path1_points)
        line1.cmap('jet', smoothed_diams1)
        line1.lw(8)

        if artery == 'left':
            line1.add_scalarbar(title="Lewa średnica [mm]", pos=((0, 0.05), (0.1, 0.35)))
        else:
            line1.add_scalarbar(title="Prawa średnica [mm]", pos=((0.85, 0.05), (0.95, 0.35)))

        plt.add(line1)
        visual_objects.append(line1)

        print(f"[{artery.upper()}] Ścieżka 1: {len(path1)} punktów")
        print("Średnice (mm):", np.round(smoothed_diams1, 2))

        # Zaznacz zwężenia
        for result in stenosis_results1:
            sten_point = points[path1[result['index']]]
            sten_sphere = vedo.Sphere(pos=sten_point, r=1.5, c='red')
            plt.add(sten_sphere)
            visual_objects.append(sten_sphere)

            cadrads = classify_cadrads(result['stenosis_percent'])
            print(f"Zwężenie {result['stenosis_percent']:.1f}% -> {cadrads}")

    except nx.NetworkXNoPath:
        print(f"[{artery.upper()}] Nie znaleziono ścieżki 1")

    if artery == 'left':
        try:
            path2 = nx.shortest_path(G, start_idx, end2_idx, weight='weight')
            diams2 = path_diameter_report(path2, points, dist_map, spacing)
            smoothed_diams2, stenosis_results2 = advanced_stenosis_analysis(diams2, spacing)

            path2_points = points[path2]
            line2 = vedo.Line(path2_points)
            line2.cmap('jet', smoothed_diams1)
            line2.lw(8)
            plt.add(line2)
            visual_objects.append(line2)

            print(f"[{artery.upper()}] Ścieżka 2: {len(path2)} punktów")
            print("Średnice (mm):", np.round(smoothed_diams2, 2))

            for result in stenosis_results2:
                sten_point = points[path2[result['index']]]
                sten_sphere = vedo.Sphere(pos=sten_point, r=1.5, c='red')
                plt.add(sten_sphere)
                visual_objects.append(sten_sphere)

                cadrads = classify_cadrads(result['stenosis_percent'])
                print(f"Zwężenie {result['stenosis_percent']:.1f}% -> {cadrads}")

        except nx.NetworkXNoPath:
            print(f"[{artery.upper()}] Nie znaleziono ścieżki 2")

    plt.render()


def calculate_stenosis_length(diameters, idx, spacing):
    """Oblicza długość zwężenia w mm"""
    avg_spacing = np.mean(spacing)
    threshold = diameters[idx] * 1.2  # Punkt gdzie zwężenie się kończy

    # Szukaj początku zwężenia
    start = idx
    while start > 0 and diameters[start] < threshold:
        start -= 1

    # Szukaj końca zwężenia
    end = idx
    while end < len(diameters) - 1 and diameters[end] < threshold:
        end += 1

    return (end - start) * avg_spacing


def handle_click(event):
    if not event.actor or event.keypress:
        return
    clicked_pos = event.picked3d
    if clicked_pos is None:
        return

    # Oblicz odległości do punktów szkieletu
    if len(left_points) > 0:
        distances_left = np.linalg.norm((left_points - clicked_pos) * left_spacing, axis=1)
        closest_left = np.min(distances_left)
    else:
        closest_left = np.inf

    if len(right_points) > 0:
        distances_right = np.linalg.norm((right_points - clicked_pos) * right_spacing, axis=1)
        closest_right = np.min(distances_right)
    else:
        closest_right = np.inf

    # Wybierz tętnice (lewa ma pierwszeństwo przy tej samej odległości)
    if closest_left <= closest_right:
        artery = 'left'
        points = left_points
        selected_points = selected_points_left
        visual_objects = visual_objects_left
        spacing = left_spacing
    else:
        artery = 'right'
        points = right_points
        selected_points = selected_points_right
        visual_objects = visual_objects_right
        spacing = right_spacing

    if len(points) == 0:
        print(f"Brak punktów szkieletu dla {artery.upper()}!")
        return

    max_points = 3 if artery == 'left' else 2
    if len(selected_points) >= max_points:
        print(f"Już wybrano {max_points} punktów dla {artery} tętnicy.")
        return

    # Znajdź najbliższy punkt szkieletu (uwzględniając spacing)
    distances = np.linalg.norm((points - clicked_pos) * spacing, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]
    selected_points.append(closest_idx)

    # Kolory punktów
    colors = ['green', 'red', 'orange'] if artery == 'left' else ['green', 'red']
    point_color = colors[len(selected_points) - 1]

    # Dodaj wizualizację punktu
    point_sphere = vedo.Sphere(pos=closest_point, r=1, c=point_color)
    plt.add(point_sphere)
    visual_objects.append(point_sphere)

    print(f"[{artery.upper()}] Wybrano punkt {len(selected_points)}: indeks {closest_idx}, pozycja {closest_point}")

    # Sprawdź czy wybrano wystarczającą liczbę punktów
    if (artery == 'left' and len(selected_points) == 3) or (artery == 'right' and len(selected_points) == 2):
        find_paths(artery)


plt.add_callback('LeftButtonPress', handle_click)
plt.interactive().close()