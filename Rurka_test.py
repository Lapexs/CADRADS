import nrrd
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import vedo
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d


def load_artery(filepath):
    try:
        data, header = nrrd.read(filepath)
        mask = (data > 0).astype(np.uint8)
        skeleton = skeletonize(mask).astype(np.uint8)
        points = np.argwhere(skeleton > 0)
        print(f"Tętnica: {len(points)} punktów szkieletu")

        # Odczytaj spacing
        if 'space directions' in header:
            spacing = np.array([np.linalg.norm(v) for v in header['space directions']])
        elif 'spacing' in header:
            spacing = np.array(header['spacing'])
        else:
            spacing = np.array([1.0, 1.0, 1.0])  # fallback
            print("UWAGA: Nie znaleziono spacing w pliku NRRD, zakładam 1mm/voxel")

        print(f"Spacing (mm/voxel): {spacing}")

        # Odległość od ściany w mm - POPRAWKA: przekaż spacing do distance_transform_edt
        mask_float = mask.astype(np.float64)
        dist_map = ndimage.distance_transform_edt(mask_float, sampling=spacing)
        return mask, skeleton, points, dist_map, spacing
    except Exception as e:
        print(f"Błąd ładowania tętnicy: {str(e)}")
        return None, None, np.array([]), None, np.array([1, 1, 1])


def local_average_diameter(p, dist_map, spacing, window=3):
    """
    POPRAWKA: Zmniejszono okno i dodano lepsze sprawdzanie granic
    """
    z, y, x = p
    z_start = max(0, z - window)
    z_end = min(dist_map.shape[0], z + window + 1)
    y_start = max(0, y - window)
    y_end = min(dist_map.shape[1], y + window + 1)
    x_start = max(0, x - window)
    x_end = min(dist_map.shape[2], x + window + 1)

    region = dist_map[z_start:z_end, y_start:y_end, x_start:x_end]
    valid = region[region > 0]

    if len(valid) < 3:
        # Fallback do pojedynczego punktu
        return 2 * dist_map[z, y, x]

    # Średnica = 2 * promień (odległość od ściany)
    return 2 * np.mean(valid)


# Wczytaj pojedynczy plik
mask, skeleton, points, dist_map, spacing = load_artery(
    r"C:\Users\student.VIRMED\Desktop\Slicer_JM\TEST\Lumen_50.nrrd")


# Tworzenie grafu
def build_graph(points, skeleton):
    if len(points) == 0:
        print("Brak punktów do budowy grafu")
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
    print(f"Graf ma {G.number_of_nodes()} węzłów i {G.number_of_edges()} krawędzi")
    return G, point_to_id


G, point_to_id = build_graph(points, skeleton)

# Wizualizacja
mesh = vedo.Volume(mask).isosurface().c('lightgreen').alpha(0.2)
skel = vedo.Points(points, r=3, c='darkgreen')

plt = vedo.Plotter(title="Kliknij 2 punkty: start, koniec",
                   axes=1, bg='white', size=(1000, 800))
plt.show([mesh, skel], resetcam=True, interactive=False)

selected_points = []
visual_objects = []


def reset_selection():
    global selected_points, visual_objects, mesh, skel

    for obj in visual_objects:
        plt.remove(obj)
    selected_points.clear()
    visual_objects.clear()
    print("Resetowano wybory i ścieżki.")

    plt.clear()
    mesh = vedo.Volume(mask).isosurface().c('lightgreen').alpha(0.2)
    skel = vedo.Points(points, r=4, c='darkgreen')
    plt.add([mesh, skel])
    plt.render()
    print("Scena została zresetowana. Możesz ponownie wybierać punkty.")


def handle_click(event):
    if not event.actor or event.keypress:
        return
    clicked_pos = event.picked3d
    if clicked_pos is None:
        return

    if len(points) == 0:
        print("Brak punktów szkieletu!")
        return

    distances = np.linalg.norm(points - clicked_pos, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]

    # Sprawdź czy punkt już został wybrany
    if closest_idx in selected_points:
        print("Ten punkt został już wybrany!")
        return

    if len(selected_points) >= 2:
        print("Już wybrano 2 punkty.")
        return

    selected_points.append(closest_idx)

    # Kolory punktów
    colors = ['green', 'red']
    point_color = colors[len(selected_points) - 1]
    point_sphere = vedo.Sphere(pos=closest_point, r=5, c=point_color)
    plt.add(point_sphere)
    visual_objects.append(point_sphere)
    print(f"Indeks klikniętego punktu: {closest_idx}, Koordynaty: {closest_point}")

    if len(selected_points) == 2:
        find_paths()


def find_paths():
    if len(selected_points) < 2:
        print("Zbyt mało punktów.")
        return

    start_idx = selected_points[0]
    end1_idx = selected_points[1]

    def path_diameter_report(path, points, dist_map, spacing):
        """
        POPRAWKA: Dodano spacing jako parametr i poprawiono obliczenia
        """
        diameters = []
        positions = []
        for idx in path:
            p = points[idx]
            # Średnica już w mm dzięki poprawnej distance_transform_edt
            diam = local_average_diameter(p, dist_map, spacing)
            diameters.append(diam)
            positions.append(tuple(p))
        return diameters, positions

    try:
        path1 = nx.shortest_path(G, start_idx, end1_idx)
        diams1, positions1 = path_diameter_report(path1, points, dist_map, spacing)

        # Debug info
        print(f"Ścieżka 1: {len(path1)} punktów")
        print(f"Średnica na ścieżce 1 (mm): min={min(diams1):.2f}, max={max(diams1):.2f}")
        print(f"Przykładowe średnice: {np.round(diams1[::len(diams1) // 10], 2)}")

        show_diameter_profile(diams1)
        line1 = vedo.Line(points[path1])
        line1.cmap('jet', diams1)
        line1.lw(8)
        line1.add_scalarbar(title="Średnica [mm]", pos=((0, 0.05), (0.1, 0.35)))
        plt.add(line1)
        visual_objects.append(line1)

        score1, max_stenosis1, regions1 = analyze_cadrads(diams1, spacing=spacing)
        print(f"CAD-RADS Path 1: {score1}, maksymalne zwężenie: {max_stenosis1}%")

        for idx, sten in regions1:
            if idx < len(positions1):
                sten_point = positions1[idx]
                sten_sphere = vedo.Sphere(pos=sten_point, r=2, c='red')
                plt.add(sten_sphere)
                visual_objects.append(sten_sphere)

    except nx.NetworkXNoPath:
        print("Nie znaleziono ścieżki 1")
    plt.render()


def show_diameter_profile(diams):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(diams, 'b-', linewidth=2, label='Średnica')
    plt.title("Profil średnicy wzdłuż ścieżki", fontsize=14)
    plt.xlabel("Indeks punktu wzdłuż ścieżki", fontsize=12)
    plt.ylabel("Średnica (mm)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Dodaj statystyki
    plt.axhline(y=np.mean(diams), color='r', linestyle='--', alpha=0.7, label=f'Średnia: {np.mean(diams):.2f}mm')
    plt.axhline(y=np.min(diams), color='orange', linestyle='--', alpha=0.7, label=f'Min: {np.min(diams):.2f}mm')
    plt.axhline(y=np.max(diams), color='green', linestyle='--', alpha=0.7, label=f'Max: {np.max(diams):.2f}mm')
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_cadrads(diameters, spacing, min_points_for_reference=20, min_stenosis=10, max_markers=3):
    """
    POPRAWKA: Całkowicie przepisana funkcja analizy zwężeń
    """
    if len(diameters) < 10:
        return "CAD-RADS 0 (za mało danych)", 0, []

    # Wygładź średnice aby zredukować szum
    diameters = smooth_diameters(diameters, sigma=2.0)

    print(
        f"Analizowane średnice po wygładzeniu: min={min(diameters):.2f}, max={max(diameters):.2f}, średnia={np.mean(diameters):.2f}")

    max_narrowing = 0
    narrowing_regions = []

    # Ustal średnicę referencyjną jako średnią z pierwszych 20% punktów (zakładamy że to zdrowy obszar)
    ref_length = max(min_points_for_reference, len(diameters) // 5)
    reference_diameter = np.mean(diameters[:ref_length])

    print(f"Średnica referencyjna (z pierwszych {ref_length} punktów): {reference_diameter:.2f}mm")

    # Sprawdź każdy punkt względem średnicy referencyjnej
    for i, curr_diam in enumerate(diameters):
        if reference_diameter <= 0:
            continue

        stenosis = (1 - curr_diam / reference_diameter) * 100

        if stenosis > max_narrowing:
            max_narrowing = stenosis

        if stenosis >= min_stenosis:
            narrowing_regions.append((i, stenosis))

    # Sortuj zwężenia malejąco i ogranicz liczbę
    narrowing_regions.sort(key=lambda x: x[1], reverse=True)
    narrowing_regions = narrowing_regions[:max_markers]

    # Klasyfikacja CAD-RADS
    if max_narrowing < 0:  # Brak zwężenia lub poszerzenie
        score = "CAD-RADS 0 (brak zwężenia)"
    elif max_narrowing < 25:
        score = "CAD-RADS 1 (minimalne zwężenie)"
    elif max_narrowing < 50:
        score = "CAD-RADS 2 (łagodne zwężenie)"
    elif max_narrowing < 70:
        score = "CAD-RADS 3 (umiarkowane zwężenie)"
    elif max_narrowing < 90:
        score = "CAD-RADS 4 (ciężkie zwężenie)"
    else:
        score = "CAD-RADS 5 (prawie całkowita okluzja)"

    print(f"Znalezione zwężenia: {len(narrowing_regions)} regionów")
    for i, (idx, sten) in enumerate(narrowing_regions):
        print(f"  Region {i + 1}: punkt {idx}, zwężenie {sten:.1f}%")

    return score, round(max_narrowing, 2), narrowing_regions


def smooth_diameters(diams, sigma=1.0):
    """
    POPRAWKA: Rzeczywiste wygładzenie średnic przy użyciu filtru Gaussa
    """
    if len(diams) < 3:
        return np.array(diams)

    # Zastosuj filtr Gaussa do wygładzenia
    smoothed = gaussian_filter1d(np.array(diams), sigma=sigma)
    return smoothed


def handle_keypress(event):
    key = event.keypress.lower()
    if key == 'r':
        reset_selection()


plt.add_callback('LeftButtonPress', handle_click)
plt.add_callback("KeyPress", handle_keypress)
plt.interactive().close()