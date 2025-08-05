import nrrd
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import vedo
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

def load_artery(filepath, artery_name):
    try:
        data, header = nrrd.read(filepath)
        mask = (data > 0).astype(np.uint8)
        skeleton = skeletonize(mask).astype(np.uint8)
        points = np.argwhere(skeleton > 0)
        print(f"{artery_name} tętnica: {len(points)} punktów szkieletu")

        # --- Odczytaj spacing ---
        if 'space directions' in header:
            spacing = np.array([np.linalg.norm(v) for v in header['space directions']])
        elif 'spacing' in header:
            spacing = np.array(header['spacing'])
        else:
            spacing = np.array([1.0, 1.0, 1.0])  # fallback
            print(f"UWAGA: Nie znaleziono spacing w pliku NRRD dla {artery_name}, zakładam 1mm/voxel")

        # Odległość od ściany w mm
        spacing = np.array(spacing, dtype=np.float64)
        mask_float = mask.astype(np.float64)
        dist_map = ndimage.distance_transform_edt(mask_float, sampling=spacing)
        return mask, skeleton, points, dist_map, spacing
    except Exception as e:
        print(f"Błąd ładowania {artery_name} tętnicy: {str(e)}")
        return None, None, np.array([]), None, np.array([1,1,1])

def local_average_diameter(p, dist_map, window=3):
    z, y, x = p
    region = dist_map[max(0, z-window):z+window+1,
                      max(0, y-window):y+window+1,
                      max(0, x-window):x+window+1]
    valid = region[region > 0]
    if len(valid) < 5:
        return 2 * dist_map[tuple(p)]
    return 2 * np.mean(valid)

left_mask, left_skeleton, left_points, left_dist_map, left_spacing = load_artery(
    r"C:\Users\student.VIRMED\Desktop\Slicer_JM\CADRADS 1\Segmentation_left.nrrd.nrrd", "Lewa")
right_mask, right_skeleton, right_points, right_dist_map, right_spacing = load_artery(
    r"C:\Users\student.VIRMED\Desktop\Slicer_JM\CADRADS 1\Segmentation_right.nrrd.nrrd", "Prawa")

# --- Odległość od ściany (dla średnicy) ---
left_dist_map = ndimage.distance_transform_edt(left_mask)
right_dist_map = ndimage.distance_transform_edt(right_mask) if right_mask is not None else None

# --- Tworzenie grafów ---
def build_graph(points, skeleton, artery_name):
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
                        G.add_edge(point_to_id[point], point_to_id[neighbor])
    print(f"{artery_name} tętnica: graf ma {G.number_of_nodes()} węzłów i {G.number_of_edges()} krawędzi")
    return G, point_to_id

G_left, left_point_to_id = build_graph(left_points, left_skeleton, "Lewa")
G_right, right_point_to_id = build_graph(right_points, right_skeleton, "Prawa")

# --- Wizualizacja, zmienne globalne ---
global left_mesh, left_skel, right_mesh, right_skel

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

def reset_selection(artery='all'):
    global selected_points_left, selected_points_right
    global visual_objects_left, visual_objects_right
    global left_mesh, left_skel, right_mesh, right_skel

    if artery in ['left', 'all']:
        for obj in visual_objects_left:
            plt.remove(obj)
        selected_points_left.clear()
        visual_objects_left.clear()
        print("[LEFT] Resetowano wybory i ścieżki.")

    if artery in ['right', 'all']:
        for obj in visual_objects_right:
            plt.remove(obj)
        selected_points_right.clear()
        visual_objects_right.clear()
        print("[RIGHT] Resetowano wybory i ścieżki.")

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
    print("Scena została całkowicie zresetowana. Możesz ponownie wybierać punkty.")

def handle_click(event):
    if not event.actor or event.keypress:
        return
    clicked_pos = event.picked3d
    if clicked_pos is None:
        return
    distances_left = np.linalg.norm(left_points - clicked_pos, axis=1) if len(left_points) else [np.inf]
    distances_right = np.linalg.norm(right_points - clicked_pos, axis=1) if len(right_points) else [np.inf]
    closest_left = np.min(distances_left)
    closest_right = np.min(distances_right)
    if closest_left < closest_right:
        artery = 'left'
        points = left_points
        selected_points = selected_points_left
        visual_objects = visual_objects_left
    else:
        artery = 'right'
        points = right_points
        selected_points = selected_points_right
        visual_objects = visual_objects_right
    if len(points) == 0:
        print(f"Brak punktów szkieletu dla {artery.upper()}!")
        return
    max_points = 3 if artery == 'left' else 2
    if len(selected_points) >= max_points:
        print(f"Już wybrano {max_points} punktów dla {artery} tętnicy.")
        return
    distances = np.linalg.norm(points - clicked_pos, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]
    selected_points.append(closest_idx)
    # Kolory punktów
    colors = ['green', 'red', 'orange'] if artery == 'left' else ['green', 'red']
    point_color = colors[len(selected_points) - 1]
    point_sphere = vedo.Sphere(pos=closest_point, r=1, c=point_color)
    plt.add(point_sphere)
    visual_objects.append(point_sphere)
    print(f"[{artery.upper()}] Indeks klikniętego punktu: {closest_idx}, Koordynaty: {closest_point}")
    if (artery == 'left' and len(selected_points) == 3) or (artery == 'right' and len(selected_points) == 2):
        find_paths(artery)

def find_paths(artery):
    if artery == 'left':
        G = G_left
        points = left_points
        selected_points = selected_points_left
        visual_objects = visual_objects_left
        dist_map = left_dist_map
    else:
        G = G_right
        points = right_points
        selected_points = selected_points_right
        visual_objects = visual_objects_right
        dist_map = right_dist_map

    if (artery == 'left' and len(selected_points) < 3) or (artery == 'right' and len(selected_points) < 2):
        print(f"Zbyt mało punktów dla {artery} tętnicy.")
        return

    start_idx = selected_points[0]
    end1_idx = selected_points[1]
    end2_idx = selected_points[2] if artery == 'left' else None

    def path_diameter_report(path, points, dist_map):
        diameters = []
        for idx in path:
            p = points[idx]
            diam = local_average_diameter(p, dist_map)
            dist = diam / 2
            diameters.append((tuple(p), dist, diam))
        return diameters

    try:
        path1 = nx.shortest_path(G, start_idx, end1_idx)
        path1_points = points[path1]
        diam_info1 = path_diameter_report(path1,points,dist_map)
        diams1 = [d[2] for d in diam_info1]
        line1 = vedo.Line(path1_points)
        line1.cmap('jet', diams1)
        line1.lw(8)
        if artery == 'left':
            line1.add_scalarbar(title="Lewa srednica [mm]",pos=((0,0.05),(0.1,0.35)))
        elif artery == 'right':
            line1.add_scalarbar(title="Prawa srednica [mm]",pos=((0.85,0.05),(0.95,0.35)))
        plt.add(line1)
        visual_objects.append(line1)
        print(f"[{artery.upper()}] Ścieżka 1: {len(path1)} punktów")
        print("Średnica na ścieżce 1 (mm):", np.round(diams1, 2))

        score1, max_stenosis1, regions1 = analyze_cadrads(diams1,spacing=left_spacing if artery == 'left' else right_spacing,min_stenosis=50, max_markers=1)
        print(f"[{artery.upper()}] CAD-RADS Path 1: {score1}, maksymalne zwężenie: {max_stenosis1}%")
        for idx, sten in regions1:
            sten_point = points[path1[idx]]
            sten_sphere = vedo.Sphere(pos=sten_point, r=1.5, c='red')
            plt.add(sten_sphere)
            visual_objects.append(sten_sphere)

    except nx.NetworkXNoPath:
        print(f"[{artery.upper()}] Nie znaleziono ścieżki 1")

    if artery == 'left':
        try:
            path2 = nx.shortest_path(G, start_idx, end2_idx)
            path2_points = points[path2]
            diam_info2 = path_diameter_report(path2,points, dist_map)
            diams2 = [d[2] for d in diam_info2]
            line2 = vedo.Line(path2_points)
            line2.cmap('jet', diams2)
            line2.lw(8)
            #line2.add_scalarbar(title="Średnica [mm]")
            plt.add(line2)
            visual_objects.append(line2)
            print(f"[{artery.upper()}] Ścieżka 2: {len(path2)} punktów")
            print("Średnica na ścieżce 2 (mm):", np.round(diams2, 2))

            score2, max_stenosis2, regions2 = analyze_cadrads(diams2,spacing=left_spacing if artery == 'left' else right_spacing, min_stenosis=50, max_markers=1)
            print(f"[{artery.upper()}] CAD-RADS Path 2: {score2}, maksymalne zwężenie: {max_stenosis2}%")
            for idx, sten in regions2:
                sten_point = points[path2[idx]]
                sten_sphere = vedo.Sphere(pos=sten_point, r=1.5, c='red')
                plt.add(sten_sphere)
                visual_objects.append(sten_sphere)

        except nx.NetworkXNoPath:
            print(f"[{artery.upper()}] Nie znaleziono ścieżki 2")
        # Szukanie punktu rozgałęzienia
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

    print(f"[{artery.upper()}] Sprawdzam ścieżki między indeksami: {selected_points}")
    plt.render()

def analyze_cadrads(diameters, spacing, step=1, window=3, min_stenosis=30, max_markers=3):
    """
    Analizuje lokalne zwężenia CAD-RADS wzdłuż listy średnic.
    diameters: lista średnic w mm wzdłuż ścieżki.
    step: co ile kroków sprawdzamy.
    window: ile kroków wstecz porównujemy.
    min_stenosis: minimalne zwężenie (%) do oznaczenia
    max_markers: maksymalna liczba znaczników do wyświetlenia
    """
    max_narrowing = 0
    narrowing_regions = []

    diameters = smooth_diameters(diameters, sigma=0.2)

    for i in range(window, len(diameters), step):
        prev_window = diameters[i - window:i - 1]
        curr = diameters[i]

        if len(prev_window) < 2 or curr < 1.0:
            continue

        ref_diam = np.percentile(prev_window, 75)
        if ref_diam <= 0 or ref_diam < 1.0:
            continue

        stenosis = (1 - curr / ref_diam) * 100
        if stenosis > max_narrowing:
            max_narrowing = stenosis
        if stenosis >= min_stenosis:
            narrowing_regions.append((i, stenosis))

    # Sortuj i ogranicz
    narrowing_regions.sort(key=lambda x: x[1], reverse=True)
    narrowing_regions = narrowing_regions[:max_markers]

    # Klasyfikacja CAD-RADS
    if max_narrowing == 0:
        score = "CAD-RADS 0 (brak zwężenie)"
    elif max_narrowing < 25:
        score = "CAD-RADS 1 (minimalne zwężenie)"
    elif max_narrowing < 50:
        score = "CAD-RADS 2 (łagodne zwężenie)"
    elif max_narrowing < 70:
        score = "CAD-RADS 3 (umiarkowane)"
    elif max_narrowing < 90:
        score = "CAD-RADS 4 (ciężkie)"
    else:
        score = "CAD-RADS 5 (prawie całkowita okluzja)"

    return score, round(max_narrowing, 2), narrowing_regions


def smooth_diameters(diams, sigma=0.2):
    """Wygładza zmienność średnic naczynia, aby uniknąć szumów i artefaktów."""
    return gaussian_filter1d(diams, sigma=sigma)


def handle_keypress(event):
    key = event.keypress.lower()
    if key == 'r':
        reset_selection('all')
    elif key == 'l':
        reset_selection('left')
    elif key == 'p':
        reset_selection('right')

plt.add_callback('LeftButtonPress', handle_click)
plt.add_callback("KeyPress", handle_keypress)
plt.interactive().close()
