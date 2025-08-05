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
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ArteryData:
    """Klasa przechowująca dane tętnicy"""
    name: str
    mask: Optional[np.ndarray] = None
    skeleton: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    dist_map: Optional[np.ndarray] = None
    spacing: Optional[np.ndarray] = None
    graph: Optional[nx.Graph] = None
    point_to_id: Optional[Dict] = None


@dataclass
class StenosisRegion:
    """Klasa przechowująca informacje o regionie stenozy"""
    start_idx: int
    end_idx: int
    length: int
    max_stenosis: float
    max_stenosis_idx: int
    mean_stenosis: float = 0.0
    location_mm: float = 0.0


class ArteryLoader:
    """Klasa odpowiedzialna za ładowanie i preprocessing danych tętnic"""

    @staticmethod
    def load_artery(filepath: str, artery_name: str) -> ArteryData:
        """Ulepszone ładowanie z lepszą kontrolą błędów i preprocessing"""
        artery_data = ArteryData(name=artery_name)

        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            data, header = nrrd.read(filepath)
            logger.info(f"Loaded {artery_name} artery from {filepath}")

            # Preprocessing
            mask = ArteryLoader._preprocess_mask(data)
            skeleton = ArteryLoader._create_skeleton(mask)
            points = np.argwhere(skeleton > 0)
            spacing = ArteryLoader._extract_spacing(header)
            dist_map = ArteryLoader._compute_distance_map(mask, spacing)

            artery_data.mask = mask
            artery_data.skeleton = skeleton
            artery_data.points = points
            artery_data.spacing = spacing
            artery_data.dist_map = dist_map

            logger.info(f"{artery_name} artery: {len(points)} skeleton points after preprocessing")
            logger.info(f"Spacing (mm/voxel): {spacing}")

            return artery_data

        except Exception as e:
            logger.error(f"Error loading {artery_name} artery: {str(e)}")
            return artery_data

    @staticmethod
    def _preprocess_mask(data: np.ndarray) -> np.ndarray:
        """Preprocessing maski z lepszą filtracją"""
        mask = (data > 0).astype(np.uint8)

        # Usuń małe komponenty
        labeled = label(mask)
        props = regionprops(labeled)

        if len(props) > 0:
            # Zachowaj tylko największy komponent
            largest_component = max(props, key=lambda x: x.area)
            mask = (labeled == largest_component.label).astype(np.uint8)

            # Opcjonalnie: usuń komponenty mniejsze niż 1% największego
            min_size = largest_component.area * 0.01
            for prop in props:
                if prop.area < min_size:
                    mask[labeled == prop.label] = 0

        # Wygładzenie
        mask_smooth = gaussian(mask.astype(float), sigma=0.5) > 0.5
        return mask_smooth.astype(np.uint8)

    @staticmethod
    def _create_skeleton(mask: np.ndarray) -> np.ndarray:
        """Tworzenie szkieletu z lepszą kontrolą jakości"""
        skeleton = skeletonize(mask).astype(np.uint8)

        # Opcjonalne: czyszczenie szkieletu z artefaktów
        # Usuń pojedyncze piksele
        labeled_skel = label(skeleton)
        props = regionprops(labeled_skel)

        for prop in props:
            if prop.area < 3:  # Usuń bardzo małe komponenty szkieletu
                skeleton[labeled_skel == prop.label] = 0

        return skeleton

    @staticmethod
    def _extract_spacing(header: Dict) -> np.ndarray:
        """Ulepszone wyciąganie spacing z obsługą różnych formatów"""
        # Próbuj space directions
        if 'space directions' in header and header['space directions'] is not None:
            space_dirs = header['space directions']
            spacing = []
            for direction in space_dirs:
                if direction is not None and hasattr(direction, '__len__'):
                    valid_values = [x for x in direction if x is not None and not np.isnan(x)]
                    if valid_values:
                        norm = np.linalg.norm(valid_values)
                        spacing.append(norm if norm > 0 else 1.0)
                    else:
                        spacing.append(1.0)
                else:
                    spacing.append(1.0)
            if len(spacing) >= 3:
                return np.array(spacing[:3])

        # Próbuj spacing
        if 'spacing' in header and header['spacing'] is not None:
            spacing = header['spacing']
            valid_spacing = [s if s is not None and not np.isnan(s) and s > 0 else 1.0 for s in spacing]
            if len(valid_spacing) >= 3:
                return np.array(valid_spacing[:3])

        # Domyślne wartości
        logger.warning("Spacing not found in NRRD file, using default 0.5mm/voxel")
        return np.array([0.5, 0.5, 0.5])

    @staticmethod
    def _compute_distance_map(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
        """Obliczenie mapy odległości z lepszą kontrolą"""
        try:
            mask_float = mask.astype(np.float64)
            dist_map = ndimage.distance_transform_edt(mask_float, sampling=spacing)
            return dist_map
        except Exception as e:
            logger.error(f"Error computing distance map: {e}")
            return np.zeros_like(mask, dtype=np.float64)


class DiameterCalculator:
    """Klasa do obliczania średnicy naczyń"""

    @staticmethod
    def adaptive_diameter_calculation(
            point: np.ndarray,
            dist_map: np.ndarray,
            skeleton: np.ndarray,
            all_points: np.ndarray,
            spacing: np.ndarray,
            base_window: int = 7,
            radius_threshold: float = 2.0
    ) -> float:
        """Ulepszone obliczanie średnicy z lepszą adaptacją"""
        z, y, x = point.astype(int)

        # Sprawdź granice
        if not (0 <= z < dist_map.shape[0] and 0 <= y < dist_map.shape[1] and 0 <= x < dist_map.shape[2]):
            return 0.0

        initial_radius = dist_map[z, y, x]

        # Adaptacyjne okno
        min_spacing = np.min(spacing)
        adaptive_window = max(3, min(base_window, int(initial_radius / min_spacing + 1)))

        # Definiuj region
        z_start = max(0, z - adaptive_window)
        z_end = min(dist_map.shape[0], z + adaptive_window + 1)
        y_start = max(0, y - adaptive_window)
        y_end = min(dist_map.shape[1], y + adaptive_window + 1)
        x_start = max(0, x - adaptive_window)
        x_end = min(dist_map.shape[2], x + adaptive_window + 1)

        # Wyciągnij region
        skel_region = skeleton[z_start:z_end, y_start:y_end, x_start:x_end]
        skel_points_in_region = np.argwhere(skel_region > 0)

        if len(skel_points_in_region) == 0:
            return 2 * initial_radius

        # Konwertuj do współrzędnych absolutnych
        skel_points_abs = skel_points_in_region + np.array([z_start, y_start, x_start])
        current_point = np.array([z, y, x])

        # Oblicz odległości w przestrzeni fizycznej
        distances_to_skel = np.linalg.norm(
            (skel_points_abs - current_point) * spacing, axis=1
        )

        # Znajdź pobliskie punkty
        nearby_indices = distances_to_skel <= radius_threshold
        if np.sum(nearby_indices) < 3:
            nearby_indices = distances_to_skel <= (radius_threshold * 2)

        if np.sum(nearby_indices) == 0:
            return 2 * initial_radius

        # Oblicz średnice dla pobliskich punktów
        nearby_skel_points = skel_points_abs[nearby_indices]
        diameters = []

        for skel_pt in nearby_skel_points:
            sz, sy, sx = skel_pt.astype(int)
            if (0 <= sz < dist_map.shape[0] and
                    0 <= sy < dist_map.shape[1] and
                    0 <= sx < dist_map.shape[2]):
                radius = dist_map[sz, sy, sx]
                if radius > 0:
                    diameters.append(2 * radius)

        if len(diameters) == 0:
            return 2 * initial_radius

        # Użyj median dla stabilności
        return np.median(diameters)


class GraphBuilder:
    """Klasa do budowania grafów z punktów szkieletu"""

    @staticmethod
    def build_graph(
            artery_data: ArteryData,
            radius_voxels: float = 1.5
    ) -> Tuple[nx.Graph, Dict]:
        """Ulepszone budowanie grafu z lepszą kontrolą połączeń"""
        points = artery_data.points
        skeleton = artery_data.skeleton
        dist_map = artery_data.dist_map
        spacing = artery_data.spacing
        artery_name = artery_data.name

        if len(points) == 0:
            logger.warning(f"No points to build graph for {artery_name} artery")
            return nx.Graph(), {}

        point_to_id = {tuple(p): i for i, p in enumerate(points)}
        G = nx.Graph()

        # Dodaj węzły z atrybutami
        for i, point in enumerate(points):
            diameter = DiameterCalculator.adaptive_diameter_calculation(
                point, dist_map, skeleton, points, spacing
            )
            G.add_node(i, pos=point, diameter=diameter)

        # Buduj KDTree dla efektywnego wyszukiwania
        scaled_points = points * spacing
        tree = cKDTree(scaled_points)

        # Znajdź sąsiadów
        search_radius = radius_voxels * np.min(spacing) * 1.75
        neighbor_indices_list = tree.query_ball_tree(tree, r=search_radius)

        # Dodaj krawędzie z wagami
        for i, neighbors in enumerate(neighbor_indices_list):
            for j in neighbors:
                if i != j and not G.has_edge(i, j):
                    dist = np.linalg.norm(scaled_points[i] - scaled_points[j])
                    G.add_edge(i, j, weight=dist)

        # Sprawdź spójność grafu
        components = list(nx.connected_components(G))
        if len(components) > 1:
            logger.warning(f"{artery_name}: Graph has {len(components)} components")
            # Zachowaj tylko największy komponent
            largest_component = max(components, key=len)
            G = G.subgraph(largest_component).copy()

        logger.info(f"{artery_name} artery: graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        artery_data.graph = G
        artery_data.point_to_id = point_to_id

        return G, point_to_id


class BifurcationDetector:
    """Klasa do wykrywania bifurkacji i rozgałęzień"""

    @staticmethod
    def detect_bifurcations(
            graph: nx.Graph,
            path_indices: List[int],
            points: np.ndarray,
            spacing: np.ndarray,
            bifurcation_threshold: int = 3
    ) -> List[int]:
        """
        Wykrywa bifurkacje na podstawie stopnia węzłów w grafie

        Args:
            graph: Graf reprezentujący strukturę naczynia
            path_indices: Indeksy punktów na ścieżce
            points: Wszystkie punkty szkieletu
            spacing: Spacing obrazu
            bifurcation_threshold: Minimalny stopień węzła do uznania za bifurkację

        Returns:
            Lista indeksów punktów będących bifurkacjami
        """
        bifurcation_indices = []

        for i, path_idx in enumerate(path_indices):
            if path_idx in graph:
                degree = graph.degree(path_idx)

                # Węzeł o stopniu >= 3 to potencjalna bifurkacja
                if degree >= bifurcation_threshold:
                    bifurcation_indices.append(i)  # Index w ścieżce, nie w całym grafie

        return bifurcation_indices

    @staticmethod
    def detect_bifurcations_geometric(
            path_points: np.ndarray,
            spacing: np.ndarray,
            angle_threshold: float = 30.0,
            distance_threshold_mm: float = 5.0
    ) -> List[int]:
        """
        Wykrywa bifurkacje na podstawie zmian kierunku i krzywizny

        Args:
            path_points: Punkty ścieżki w kolejności
            spacing: Spacing obrazu
            angle_threshold: Próg kąta w stopniach
            distance_threshold_mm: Minimalna odległość między bifurkacjami

        Returns:
            Lista indeksów punktów będących bifurkacjami
        """
        if len(path_points) < 5:
            return []

        bifurcations = []
        n = len(path_points)
        min_distance_pts = max(3, int(distance_threshold_mm / np.mean(spacing)))

        for i in range(2, n - 2):
            # Oblicz wektory przed i po punkcie
            vec_before = (path_points[i] - path_points[i - 2]) * spacing
            vec_after = (path_points[i + 2] - path_points[i]) * spacing

            # Normalizuj wektory
            norm_before = np.linalg.norm(vec_before)
            norm_after = np.linalg.norm(vec_after)

            if norm_before > 0 and norm_after > 0:
                vec_before_norm = vec_before / norm_before
                vec_after_norm = vec_after / norm_after

                # Oblicz kąt między wektorami
                dot_product = np.clip(np.dot(vec_before_norm, vec_after_norm), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot_product))

                # Jeśli kąt przekracza próg, to potencjalna bifurkacja
                if angle > angle_threshold:
                    # Sprawdź odległość od ostatniej bifurkacji
                    if not bifurcations or (i - bifurcations[-1]) > min_distance_pts:
                        bifurcations.append(i)

        return bifurcations

    @staticmethod
    def get_bifurcation_regions(
            bifurcation_indices: List[int],
            region_size_mm: float = 8.0,
            spacing: np.ndarray = None,
            path_length: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Zwraca regiony wokół bifurkacji, które należy wykluczyć z analizy stenoz

        Args:
            bifurcation_indices: Indeksy bifurkacji
            region_size_mm: Rozmiar regionu do wykluczenia (w mm)
            spacing: Spacing obrazu
            path_length: Długość ścieżki

        Returns:
            Lista krotek (start_idx, end_idx) regionów do wykluczenia
        """
        if not bifurcation_indices or spacing is None:
            return []

        mean_spacing = np.mean(spacing)
        region_size_pts = max(2, int(region_size_mm / mean_spacing))

        excluded_regions = []

        for bif_idx in bifurcation_indices:
            start_idx = max(0, bif_idx - region_size_pts // 2)
            end_idx = min(path_length - 1, bif_idx + region_size_pts // 2)
            excluded_regions.append((start_idx, end_idx))

        # Połącz nakładające się regiony
        if len(excluded_regions) > 1:
            merged_regions = []
            current_start, current_end = excluded_regions[0]

            for start, end in excluded_regions[1:]:
                if start <= current_end:  # Nakładające się regiony
                    current_end = max(current_end, end)
                else:  # Nowy region
                    merged_regions.append((current_start, current_end))
                    current_start, current_end = start, end

            merged_regions.append((current_start, current_end))
            excluded_regions = merged_regions

        return excluded_regions


class StenosisAnalyzer:
    """Klasa do analizy stenoz z wykluczeniem bifurkacji"""

    @staticmethod
    def rolling_average(arr: np.ndarray, window: int) -> np.ndarray:
        """Wygładzanie z lepszą obsługą brzegów"""
        if window < 2 or len(arr) < window:
            return arr

        # Użyj padding dla lepszej obsługi brzegów
        padded = np.pad(arr, window // 2, mode='edge')
        return np.convolve(padded, np.ones(window) / window, mode='valid')

    @staticmethod
    def detect_local_stenosis(
            diameters: List[float],
            spacing: np.ndarray,
            window_mm: float = 10,
            min_stenosis: float = 30,
            min_length_pts: int = 3,
            severity_thresholds: Dict[str, float] = None,
            graph: nx.Graph = None,
            path_indices: List[int] = None,
            path_points: np.ndarray = None
    ) -> Tuple[List[StenosisRegion], List[float]]:
        """Ulepszona detekcja stenoz z wykluczeniem bifurkacji"""

        if severity_thresholds is None:
            severity_thresholds = {
                'mild': 30,  # 30-49%
                'moderate': 50,  # 50-69%
                'severe': 70,  # 70-89%
                'critical': 90  # 90%+
            }

        diameters = np.array(diameters)
        n = len(diameters)

        if n < 5:  # Za mało punktów
            return [], [0] * n

        # Wykryj bifurkacje
        excluded_regions = []

        if graph is not None and path_indices is not None:
            # Wykrywanie bifurkacji na podstawie grafu
            bifurcations_graph = BifurcationDetector.detect_bifurcations(
                graph, path_indices, None, spacing
            )

            if bifurcations_graph:
                logger.info(f"Detected {len(bifurcations_graph)} bifurcations from graph topology")
                excluded_regions.extend(
                    BifurcationDetector.get_bifurcation_regions(
                        bifurcations_graph, region_size_mm=8.0, spacing=spacing, path_length=n
                    )
                )

        if path_points is not None:
            # Wykrywanie bifurkacji na podstawie geometrii
            bifurcations_geom = BifurcationDetector.detect_bifurcations_geometric(
                path_points, spacing, angle_threshold=35.0, distance_threshold_mm=6.0
            )

            if bifurcations_geom:
                logger.info(f"Detected {len(bifurcations_geom)} bifurcations from geometry")
                excluded_regions.extend(
                    BifurcationDetector.get_bifurcation_regions(
                        bifurcations_geom, region_size_mm=6.0, spacing=spacing, path_length=n
                    )
                )

        # Usuń duplikaty i posortuj regiony wykluczenia
        if excluded_regions:
            excluded_regions = list(set(excluded_regions))
            excluded_regions.sort()
            logger.info(f"Excluding {len(excluded_regions)} bifurcation regions from stenosis analysis")

        mean_spacing = np.mean(spacing)
        window_pts = max(2, int(window_mm / mean_spacing))

        stenosis_values = []

        # Oblicz stenozę dla każdego punktu
        for i in range(n):
            # Sprawdź, czy punkt jest w regionie bifurkacji
            in_excluded_region = any(start <= i <= end for start, end in excluded_regions)

            if in_excluded_region:
                stenosis_values.append(0)  # Nie analizuj stenoz w bifurkacjach
                continue

            # Referencyjny region - preferuj proksymalny, ale unikaj bifurkacji
            ref_indices = []

            # Zbierz indeksy referencyjne, pomijając bifurkacje
            for j in range(max(0, i - window_pts), min(n, i + window_pts + 1)):
                if j != i:  # Wyklucz bieżący punkt
                    # Sprawdź, czy punkt referencyjny nie jest w bifurkacji
                    in_ref_excluded = any(start <= j <= end for start, end in excluded_regions)
                    if not in_ref_excluded:
                        ref_indices.append(j)

            if len(ref_indices) < 2:
                stenosis_values.append(0)
                continue

            ref_diameters = diameters[ref_indices]

            # Użyj percentyla dla stabilności, preferuj większe średnice (proksymalnie)
            ref_diam = np.percentile(ref_diameters, 75)

            if ref_diam > 0:
                stenosis = max(0, (1 - diameters[i] / ref_diam) * 100)
            else:
                stenosis = 0

            stenosis_values.append(stenosis)

        # Wygładź wartości stenoz (tylko poza bifurkacjami)
        stenosis_smooth = np.array(stenosis_values).copy()

        # Wygładzaj tylko segmenty poza bifurkacjami
        if excluded_regions:
            # Podziel na segmenty poza bifurkacjami
            segments = []
            last_end = 0

            for start, end in excluded_regions:
                if start > last_end:
                    segments.append((last_end, start))
                last_end = max(last_end, end + 1)

            if last_end < n:
                segments.append((last_end, n))

            # Wygładź każdy segment osobno
            for seg_start, seg_end in segments:
                if seg_end - seg_start > 3:
                    segment = stenosis_smooth[seg_start:seg_end]
                    smooth_segment = StenosisAnalyzer.rolling_average(segment, 3)
                    stenosis_smooth[seg_start:seg_end] = smooth_segment
        else:
            stenosis_smooth = StenosisAnalyzer.rolling_average(stenosis_smooth, 3)

        # Znajdź regiony stenoz (poza bifurkacjami)
        regions = []
        current_region = []

        for idx, sten in enumerate(stenosis_smooth):
            # Sprawdź, czy punkt jest w regionie bifurkacji
            in_excluded_region = any(start <= idx <= end for start, end in excluded_regions)

            if sten >= min_stenosis and not in_excluded_region:
                current_region.append((idx, sten))
            else:
                if len(current_region) >= min_length_pts:
                    regions.append(StenosisAnalyzer._create_stenosis_region(
                        current_region, diameters, mean_spacing
                    ))
                current_region = []

        # Sprawdź ostatni region
        if len(current_region) >= min_length_pts:
            regions.append(StenosisAnalyzer._create_stenosis_region(
                current_region, diameters, mean_spacing
            ))

        return regions, stenosis_smooth.tolist()

    @staticmethod
    def _create_stenosis_region(
            region_data: List[Tuple[int, float]],
            diameters: np.ndarray,
            spacing: float
    ) -> StenosisRegion:
        """Tworzenie obiektu StenosisRegion"""
        indices, stenosis_vals = zip(*region_data)
        max_sten_idx = max(region_data, key=lambda x: x[1])

        return StenosisRegion(
            start_idx=indices[0],
            end_idx=indices[-1],
            length=len(region_data),
            max_stenosis=max_sten_idx[1],
            max_stenosis_idx=max_sten_idx[0],
            mean_stenosis=np.mean(stenosis_vals),
            location_mm=max_sten_idx[0] * spacing
        )


class VisualizationManager:
    """Klasa zarządzająca wizualizacją"""

    def __init__(self):
        self.plt = None
        self.selected_points_left = []
        self.selected_points_right = []
        self.visual_objects_left = []
        self.visual_objects_right = []
        self.arteries = {}

    def initialize_visualization(self, left_artery: ArteryData, right_artery: ArteryData):
        """Inicjalizacja wizualizacji"""
        self.arteries['left'] = left_artery
        self.arteries['right'] = right_artery

        self.plt = vedo.Plotter(
            title="Analiza tętnic - Kliknij punkty: start, koniec1, [koniec2]\n[L]-lewa [P]-prawa [R]-reset",
            axes=1, bg='white', size=(1200, 900)
        )

        objects = self._create_initial_objects()
        self.plt.show(objects, resetcam=True, interactive=False)

        # Dodaj callbacki
        self.plt.add_callback('LeftButtonPress', self._handle_click)
        self.plt.add_callback("KeyPress", self._handle_keypress)

    def _create_initial_objects(self) -> List:
        """Tworzenie początkowych obiektów wizualizacji"""
        objects = []

        # Lewa tętnica
        if self.arteries['left'].mask is not None:
            left_mesh = vedo.Volume(self.arteries['left'].mask).isosurface().c('lightgreen').alpha(0.2)
            left_skel = vedo.Points(self.arteries['left'].points, r=3, c='darkgreen')
            objects.extend([left_mesh, left_skel])

        # Prawa tętnica
        if (self.arteries['right'].mask is not None and
                np.any(self.arteries['right'].mask)):
            right_mesh = vedo.Volume(self.arteries['right'].mask).isosurface().c('lightblue').alpha(0.2)
            right_skel = vedo.Points(self.arteries['right'].points, r=3, c='darkblue')
            objects.extend([right_mesh, right_skel])

        return objects

    def _handle_click(self, event):
        """Obsługa kliknięć"""
        if not event.actor or event.keypress:
            return

        clicked_pos = event.picked3d
        if clicked_pos is None:
            return

        # Określ, która tętnica jest bliżej
        artery_name = self._determine_closest_artery(clicked_pos)
        if artery_name is None:
            return

        self._process_point_selection(artery_name, clicked_pos)

    def _determine_closest_artery(self, clicked_pos: np.ndarray) -> Optional[str]:
        """Określa, która tętnica jest najbliżej klikniętego punktu"""
        distances = {}

        for name, artery in self.arteries.items():
            if artery.points is not None and len(artery.points) > 0:
                dists = np.linalg.norm(artery.points - clicked_pos, axis=1)
                distances[name] = np.min(dists)

        if not distances:
            return None

        return min(distances.items(), key=lambda x: x[1])[0]

    def _process_point_selection(self, artery_name: str, clicked_pos: np.ndarray):
        """Przetwarzanie selekcji punktu"""
        artery = self.arteries[artery_name]
        selected_points = getattr(self, f'selected_points_{artery_name}')
        visual_objects = getattr(self, f'visual_objects_{artery_name}')

        max_points = 3 if artery_name == 'left' else 2

        if len(selected_points) >= max_points:
            logger.info(f"Already selected {max_points} points for {artery_name} artery.")
            return

        # Znajdź najbliższy punkt
        distances = np.linalg.norm(artery.points - clicked_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = artery.points[closest_idx]

        selected_points.append(closest_idx)

        # Dodaj wizualizację punktu
        colors = ['green', 'red', 'orange'] if artery_name == 'left' else ['green', 'red']
        point_color = colors[len(selected_points) - 1]
        point_sphere = vedo.Sphere(pos=closest_point, r=1.5, c=point_color)

        self.plt.add(point_sphere)
        visual_objects.append(point_sphere)

        logger.info(f"[{artery_name.upper()}] Selected point {len(selected_points)}: index {closest_idx}")

        # Sprawdź, czy można znaleźć ścieżki
        if (artery_name == 'left' and len(selected_points) == 3) or \
                (artery_name == 'right' and len(selected_points) == 2):
            self._find_and_visualize_paths(artery_name)

    def _find_and_visualize_paths(self, artery_name: str):
        """Znajdowanie i wizualizacja ścieżek"""
        artery = self.arteries[artery_name]
        selected_points = getattr(self, f'selected_points_{artery_name}')
        visual_objects = getattr(self, f'visual_objects_{artery_name}')

        if artery.graph is None:
            logger.error(f"No graph available for {artery_name} artery")
            return

        try:
            paths = self._calculate_paths(artery, selected_points)
            self._visualize_paths(artery_name, artery, paths, visual_objects)

        except Exception as e:
            logger.error(f"Error finding paths for {artery_name}: {e}")

    def _calculate_paths(self, artery: ArteryData, selected_points: List[int]) -> List[List[int]]:
        """Obliczanie ścieżek między wybranymi punktami"""
        paths = []
        start_idx = selected_points[0]

        for end_idx in selected_points[1:]:
            try:
                path = nx.shortest_path(artery.graph, start_idx, end_idx)
                paths.append(path)
            except nx.NetworkXNoPath:
                logger.warning(f"No path found between {start_idx} and {end_idx}")

        return paths

    def _visualize_paths(self, artery_name: str, artery: ArteryData, paths: List[List[int]], visual_objects: List):
        """Wizualizacja ścieżek z analizą stenoz i wykluczeniem bifurkacji"""

        for i, path in enumerate(paths, 1):
            if not path:
                continue

            # Oblicz średnice
            diameters = []
            for idx in path:
                point = artery.points[idx]
                diameter = DiameterCalculator.adaptive_diameter_calculation(
                    point, artery.dist_map, artery.skeleton, artery.points, artery.spacing
                )
                diameters.append(diameter)

            # Wizualizuj ścieżkę
            path_points = artery.points[path]
            line = vedo.Line(path_points)
            line.cmap('jet', diameters)
            line.lw(8)

            # Dodaj skalę kolorów
            pos_x = 0.0 if artery_name == 'left' else 0.85
            line.add_scalarbar(
                title=f"{artery_name.capitalize()} - Path {i} [mm]",
                pos=((pos_x, 0.05), (pos_x + 0.1, 0.35))
            )

            self.plt.add(line)
            visual_objects.append(line)

            # Analiza stenoz z wykluczeniem bifurkacji
            regions, stenosis_values = StenosisAnalyzer.detect_local_stenosis(
                diameters,
                artery.spacing,
                window_mm=8,
                min_stenosis=25,
                min_length_pts=3,
                graph=artery.graph,
                path_indices=path,
                path_points=path_points
            )

            # Raportowanie
            self._report_path_analysis(artery_name, i, diameters, regions, stenosis_values)

            # Wizualizuj stenozy (tylko te poza bifurkacjami)
            self._visualize_stenosis_regions(artery, path, regions, visual_objects)

            # Wizualizuj bifurkacje osobno
            self._visualize_bifurcations(artery_name, artery, path, path_points, visual_objects)

    def _visualize_bifurcations(self, artery_name: str, artery: ArteryData, path: List[int],
                                path_points: np.ndarray, visual_objects: List):
        """Wizualizuj bifurkacje jako niebieskie kule"""
        # Wykryj bifurkacje na podstawie grafu
        if artery.graph is not None:
            bifurcations_graph = BifurcationDetector.detect_bifurcations(
                artery.graph, path, artery.points, artery.spacing
            )

            for bif_idx in bifurcations_graph:
                if 0 <= bif_idx < len(path):
                    bif_point = artery.points[path[bif_idx]]
                    bif_sphere = vedo.Sphere(pos=bif_point, r=1.2, c='lightblue').alpha(0.7)
                    self.plt.add(bif_sphere)
                    visual_objects.append(bif_sphere)

        # Wykryj bifurkacje geometryczne
        bifurcations_geom = BifurcationDetector.detect_bifurcations_geometric(
            path_points, artery.spacing, angle_threshold=35.0
        )

        for bif_idx in bifurcations_geom:
            if 0 <= bif_idx < len(path_points):
                bif_point = path_points[bif_idx]
                bif_sphere = vedo.Sphere(pos=bif_point, r=1.0, c='cyan').alpha(0.8)
                self.plt.add(bif_sphere)
                visual_objects.append(bif_sphere)

    def _report_path_analysis(self, artery_name: str, path_num: int, diameters: List[float],
                              regions: List[StenosisRegion], stenosis_values: List[float]):
        """Szczegółowe raportowanie analizy ścieżki z informacją o bifurkacjach"""
        print(f"\n=== ANALIZA ŚCIEŻKI {path_num} - {artery_name.upper()} ===")
        print(f"Długość ścieżki: {len(diameters)} punktów")
        print(f"Średnia średnica: {np.mean(diameters):.2f} ± {np.std(diameters):.2f} mm")
        print(f"Minimalna średnica: {np.min(diameters):.2f} mm")
        print(f"Maksymalna średnica: {np.max(diameters):.2f} mm")

        # Policz punkty wykluczonych bifurkacji
        excluded_points = sum(1 for s in stenosis_values if s == 0)
        if excluded_points > 0:
            print(f"🔵 Wykluczono {excluded_points} punktów jako bifurkacje/rozgałęzienia")

        if regions:
            print(f"\n🔍 Znaleziono {len(regions)} regionów zwężeń (poza bifurkacjami):")
            for i, region in enumerate(regions, 1):
                severity = self._classify_stenosis_severity(region.max_stenosis)
                print(f"  Region {i}: {region.max_stenosis:.1f}% ({severity})")
                print(f"    Lokalizacja: {region.location_mm:.1f} mm od początku")
                print(f"    Długość: {region.length} punktów")
                print(f"    Średnia stenoza: {region.mean_stenosis:.1f}%")
        else:
            print("✅ Nie wykryto istotnych zwężeń (poza naturalnymi bifurkacjami)")

        # Dodatkowe ostrzeżenia
        severe_regions = [r for r in regions if r.max_stenosis >= 70]
        if severe_regions:
            print(f"\n⚠️  UWAGA: {len(severe_regions)} ciężkich zwężeń wymaga szczególnej uwagi!")

        # Informacja o metodologii
        print(f"\n📊 Metodologia:")
        print(f"   - Wykluczono naturalne bifurkacje z analizy stenoz")
        print(f"   - Użyto 75. percentyla jako referencję średnicy")
        print(f"   - Minimalna długość zwężenia: 3 punkty")
        print(f"   - Próg istotności: 25% zwężenia średnicy")

    @staticmethod
    def _classify_stenosis_severity(stenosis_percent: float) -> str:
        """Klasyfikacja ciężkości stenozy"""
        if stenosis_percent >= 90:
            return "KRYTYCZNA"
        elif stenosis_percent >= 70:
            return "CIĘŻKA"
        elif stenosis_percent >= 50:
            return "UMIARKOWANA"
        elif stenosis_percent >= 30:
            return "ŁAGODNA"
        else:
            return "MINIMALNA"

    def _visualize_stenosis_regions(self, artery: ArteryData, path: List[int],
                                    regions: List[StenosisRegion], visual_objects: List):
        """Wizualizacja regionów stenoz"""
        for region in regions:
            if 0 <= region.max_stenosis_idx < len(path):
                sten_point = artery.points[path[region.max_stenosis_idx]]

                # Kolor w zależności od ciężkości
                if region.max_stenosis >= 70:
                    color = 'red'
                    radius = 2.0
                elif region.max_stenosis >= 50:
                    color = 'orange'
                    radius = 1.8
                else:
                    color = 'yellow'
                    radius = 1.5

                sten_sphere = vedo.Sphere(pos=sten_point, r=radius, c=color)
                self.plt.add(sten_sphere)
                visual_objects.append(sten_sphere)

    def _handle_keypress(self, event):
        """Obsługa klawiszy"""
        key = event.keypress.lower()
        if key == 'r':
            self._reset_selection('all')
        elif key == 'l':
            self._reset_selection('left')
        elif key == 'p':
            self._reset_selection('right')

    def _reset_selection(self, artery: str = 'all'):
        """Reset selekcji i wizualizacji"""
        if artery in ['left', 'all']:
            for obj in self.visual_objects_left:
                self.plt.remove(obj)
            self.selected_points_left.clear()
            self.visual_objects_left.clear()
            logger.info("[LEFT] Selections and paths reset.")

        if artery in ['right', 'all']:
            for obj in self.visual_objects_right:
                self.plt.remove(obj)
            self.selected_points_right.clear()
            self.visual_objects_right.clear()
            logger.info("[RIGHT] Selections and paths reset.")

        if artery == 'all':
            self.plt.clear()
            objects = self._create_initial_objects()
            self.plt.add(objects)
            logger.info("Scene fully reset. You can select points again.")

        self.plt.render()


class ArteryAnalysisApp:
    """Główna klasa aplikacji"""

    def __init__(self, left_path: str, right_path: str):
        self.left_path = left_path
        self.right_path = right_path
        self.left_artery = None
        self.right_artery = None
        self.visualization_manager = None

    def run(self):
        """Uruchomienie pełnej analizy"""
        try:
            # Ładowanie danych
            self._load_arteries()

            # Budowanie grafów
            self._build_graphs()

            # Inicjalizacja wizualizacji
            self._initialize_visualization()

            # Uruchomienie interaktywnej sesji
            self._run_interactive_session()

        except Exception as e:
            logger.error(f"Error in application: {e}")
            raise

    def _load_arteries(self):
        """Ładowanie danych tętnic"""
        logger.info("Loading artery data...")

        self.left_artery = ArteryLoader.load_artery(self.left_path, "Left")
        self.right_artery = ArteryLoader.load_artery(self.right_path, "Right")

        # Sprawdzenie poprawności ładowania
        if self.left_artery.mask is None and self.right_artery.mask is None:
            raise ValueError("No valid artery data loaded")

    def _build_graphs(self):
        """Budowanie grafów dla tętnic"""
        logger.info("Building graphs...")

        if self.left_artery.mask is not None:
            GraphBuilder.build_graph(self.left_artery)

        if self.right_artery.mask is not None:
            GraphBuilder.build_graph(self.right_artery)

    def _initialize_visualization(self):
        """Inicjalizacja menedżera wizualizacji"""
        logger.info("Initializing visualization...")

        self.visualization_manager = VisualizationManager()
        self.visualization_manager.initialize_visualization(
            self.left_artery, self.right_artery
        )

    def _run_interactive_session(self):
        """Uruchomienie sesji interaktywnej"""
        logger.info("Starting interactive session...")
        logger.info("Instructions:")
        logger.info("- Click on artery points to select analysis path")
        logger.info("- Left artery: select 3 points (start, end1, end2)")
        logger.info("- Right artery: select 2 points (start, end)")
        logger.info("- Press 'R' to reset all, 'L' for left only, 'P' for right only")

        self.visualization_manager.plt.interactive().close()


# Funkcje pomocnicze dla kompatybilności wstecznej
def load_artery(filepath: str, artery_name: str):
    """Wrapper dla kompatybilności wstecznej"""
    artery_data = ArteryLoader.load_artery(filepath, artery_name)
    return (artery_data.mask, artery_data.skeleton, artery_data.points,
            artery_data.dist_map, artery_data.spacing)


def build_graph(points, skeleton, dist_map, spacing, artery_name, radius_voxels=1.5):
    """Wrapper dla kompatybilności wstecznej"""
    artery_data = ArteryData(
        name=artery_name,
        points=points,
        skeleton=skeleton,
        dist_map=dist_map,
        spacing=spacing
    )
    return GraphBuilder.build_graph(artery_data, radius_voxels)


# Główna funkcja uruchamiająca
def main():
    """Główna funkcja aplikacji"""
    # Ścieżki do plików - dostosuj według potrzeb
    left_path = r"C:\Users\student.VIRMED\Desktop\Slicer_JM\CADRADS 3\Segmentation_right.nrrd"
    right_path = r"C:\Users\student.VIRMED\Desktop\Slicer_JM\CADRADS 3\Segmentation_left.nrrd"

    try:
        app = ArteryAnalysisApp(left_path, right_path)
        app.run()

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print("Sprawdź ścieżki do plików NRRD")

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Błąd aplikacji: {e}")


if __name__ == "__main__":
    main()

# Przykład użycia w starym stylu (dla kompatybilności)
"""
# Stary sposób użycia - nadal działa
left_mask, left_skeleton, left_points, left_dist_map, left_spacing = load_artery(
    r"path_to_left_artery.nrrd", "Left")
right_mask, right_skeleton, right_points, right_dist_map, right_spacing = load_artery(
    r"path_to_right_artery.nrrd", "Right")

G_left, left_point_to_id = build_graph(left_points, left_skeleton, left_dist_map, left_spacing, "Left")
G_right, right_point_to_id = build_graph(right_points, right_skeleton, right_dist_map, right_spacing, "Right")
"""