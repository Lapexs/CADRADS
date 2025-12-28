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
import os
import matplotlib
matplotlib.use('Agg') # Zapobiega wyskakiwaniu okienek wykresów

# =================================================================
# 1. KONFIGURACJA OSTATECZNA (FINAL STABLE)
# Parametry dobrane na podstawie pełnej walidacji.
# =================================================================

UPSAMPLE_FACTOR = 8
BIFURCATION_MERGE_RADIUS_MM = 2.0
BIFURCATION_MIN_BRANCH_LENGTH_MM = 5.0
BIFURCATION_BUFFER_MM = 10 # Bufor bezpieczeństwa przy rozwidleniach

# Parametry detekcji zwężeń (Strategy: High Specificity)
STENOSIS_WINDOW_MM = 15      # Stabilne tło
STENOSIS_MIN_PCT = 25.0      # Próg kliniczny (eliminuje szum <25%)
STENOSIS_MIN_LEN_MM = 3.0    # Kluczowy filtr szumu
GRADIENT_THRESH = -0.15      # Bezpiecznik morfologiczny

# NOWY PARAMETR: Ignorowanie zbyt małych naczyń
STENOSIS_IGNORE_DIAMETER_MM = 1.5 
# =================================================================

# --- FUNKCJE POMOCNICZE ---

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
    return np.array([0.5, 0.5, 0.5])

def adaptive_diameter_calculation(p, dist_map, skeleton, points, spacing, base_window=7):
    z, y, x = int(p[0]), int(p[1]), int(p[2])
    if not (0 <= z < dist_map.shape[0] and 0 <= y < dist_map.shape[1] and 0 <= x < dist_map.shape[2]):
        return 0.0
    initial_radius = dist_map[z, y, x]
    if initial_radius <= 0.5: return 2 * initial_radius
    adaptive_window = max(3, min(base_window, int(initial_radius / min(spacing) + 1)))
    z_s, z_e = max(0, z - adaptive_window), min(dist_map.shape[0], z + adaptive_window + 1)
    y_s, y_e = max(0, y - adaptive_window), min(dist_map.shape[1], y + adaptive_window + 1)
    x_s, x_e = max(0, x - adaptive_window), min(dist_map.shape[2], x + adaptive_window + 1)
    skel_region = skeleton[z_s:z_e, y_s:y_e, x_s:x_e]
    skel_points_local = np.argwhere(skel_region > 0)
    if len(skel_points_local) == 0: return 2 * dist_map[z, y, x]
    skel_points_abs = skel_points_local + np.array([z_s, y_s, x_s])
    current_point = np.array([z, y, x])
    distances = np.linalg.norm((skel_points_abs - current_point) * spacing, axis=1)
    nearby_mask = distances <= 2.0
    if np.sum(nearby_mask) < 3: nearby_mask = distances <= 4.0
    nearby_pts = skel_points_abs[nearby_mask]
    if len(nearby_pts) == 0: return 2 * dist_map[z, y, x]
    radii = dist_map[nearby_pts[:,0], nearby_pts[:,1], nearby_pts[:,2]]
    return np.median(radii) * 2

def linear_upsample_phys_points(phys_pts, factor=8):
    if len(phys_pts) < 2: return phys_pts
    segs = np.linalg.norm(np.diff(phys_pts, axis=0), axis=1)
    total = np.sum(segs)
    if total <= 0: return phys_pts
    total_samples = max(100, len(phys_pts) * factor)
    fine_pts = []
    for i in range(len(phys_pts)-1):
        a, b = phys_pts[i], phys_pts[i+1]
        nseg = max(2, int(round(total_samples * (segs[i]/total))))
        ts = np.linspace(0, 1, nseg, endpoint=False)
        for t in ts: fine_pts.append(a + (b - a) * t)
    fine_pts.append(phys_pts[-1])
    return np.vstack(fine_pts)

def compute_upsampled_path(path_pts, spacing, upsample_factor=UPSAMPLE_FACTOR):
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
    mapping_idx = np.array([int(tree.query(p)[1]) for p in phys_pts], dtype=int)
    return phys_pts, cum_orig, fine_pts, cum_fine, mapping_idx

def plot_graph_thread(diams, cum_orig, diag, regions, title, save_path):
    fig, ax1 = mplt.subplots(figsize=(10,4))
    ax1.plot(cum_orig, diams, label='Diameter', color='blue')
    ax1.plot(cum_orig, diag['d_smooth'], color='cyan', alpha=0.6, label='Smooth')
    ax1.set_ylabel('Diameter [mm]', color='blue')
    ax1.set_xlabel('Distance [mm]')
    
    ax2 = ax1.twinx()
    ax2.plot(cum_orig, diag['grad'], color='red', linestyle='--', label='Gradient')
    ax2.set_ylabel('Gradient [mm/mm]', color='red')
    
    for reg in regions:
        s = cum_orig[reg['start_idx']]
        e = cum_orig[reg['end_idx']]
        color = 'red' 
        ax1.axvspan(s, e, color=color, alpha=0.2)

    ax1.set_title(title)
    fig.tight_layout()
    mplt.savefig(save_path, dpi=150)
    mplt.close(fig)
    print(f"[PLOT] Saved to {save_path}")

# =================================================================
# 2. FUNKCJA DETEKCJI
# =================================================================
def detect_local_stenosis_with_grad(diameters, spacing, 
                                  window_mm=15, 
                                  min_stenosis=20, 
                                  min_length_mm=3.0,  # Parametr w MM
                                  use_gradient=False, 
                                  grad_thresh=-0.15, 
                                  x_positions=None):
    
    diams = np.array(diameters, dtype=float)
    n = len(diams)
    mean_spacing = float(np.mean(spacing))
    window_pts = max(1, int(round(window_mm / mean_spacing)))
    
    d_smooth = gaussian_filter1d(diams, sigma=1.0, mode='nearest')
    
    if x_positions is not None and len(x_positions) == n:
        grad = np.gradient(d_smooth, x_positions)
    else:
        grad = np.gradient(d_smooth) / mean_spacing

    pct_drop = np.zeros(n)
    for i in range(n):
        left = max(0, i - window_pts)
        right = min(n, i + window_pts + 1)
        ref_region = diams[left:right]
        if len(ref_region) > 0:
            ref_val = np.median(ref_region)
            pct_drop[i] = (1 - diams[i]/ref_val)*100 if ref_val > 0 else 0

    cond_pct = pct_drop >= min_stenosis
    regions = []
    
    if np.any(cond_pct):
        diff = np.diff(np.concatenate(([0], cond_pct.view(np.int8), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        for s, e in zip(starts, ends):
            if x_positions is not None:
                real_length_mm = x_positions[e] - x_positions[s]
            else:
                real_length_mm = (e - s + 1) * mean_spacing

            if real_length_mm >= min_length_mm:
                is_valid = True
                if use_gradient:
                    search_start = max(0, s - 3) 
                    search_end = min(n, e + 1)
                    local_grads = grad[search_start:search_end]
                    if len(local_grads) > 0:
                        steepest_drop = np.min(local_grads)
                        if steepest_drop > grad_thresh:
                            is_valid = False
                    else:
                        is_valid = False

                if is_valid:
                    sub_drop = pct_drop[s:e+1]
                    max_idx_local = np.argmax(sub_drop)
                    
                    # Obliczamy średnią średnicę w tym zwężeniu (dla weryfikacji <1.5mm)
                    avg_diameter_in_lesion = np.mean(diams[s:e+1])
                    
                    regions.append({
                        'start_idx': int(s),
                        'end_idx': int(e),
                        'length_mm': float(real_length_mm),
                        'max_stenosis': float(np.max(sub_drop)),
                        'max_stenosis_idx': int(s + max_idx_local),
                        'avg_diameter': float(avg_diameter_in_lesion) # Dodajemy info o średnicy
                    })
                
    diagnostics = {'d_smooth': d_smooth, 'grad': grad, 'pct_drop': pct_drop}
    return regions, diagnostics

# ================= KLASY =================

class Artery:
    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
        self.mask = None
        self.skeleton = None
        self.points = np.array([])
        self.dist_map = None
        self.spacing = np.array([0.5, 0.5, 0.5])
        self.graph = nx.Graph()
        self.tree = None
        self.bifurcations = ([], np.empty((0,3)))
        self.mesh = None
        self.skel_actor = None
        self.loaded = False

    def load(self):
        print(f"Loading {self.name} from {self.filepath}...")
        try:
            data, header = nrrd.read(self.filepath)
            self.spacing = extract_spacing(header)
            mask_raw = (data > 0).astype(np.uint8)
            labeled = label(mask_raw)
            props = regionprops(labeled)
            if props:
                largest = max(props, key=lambda x: x.area)
                self.mask = (labeled == largest.label).astype(np.uint8)
            else:
                self.mask = mask_raw

            mask_smooth = gaussian(self.mask.astype(float), sigma=0.5) > 0.5
            self.skeleton = skeletonize(mask_smooth).astype(np.uint8)
            self.points = np.argwhere(self.skeleton > 0)
            self.dist_map = ndimage.distance_transform_edt(self.mask.astype(float), sampling=self.spacing)
            
            if len(self.points) > 0:
                self.tree = cKDTree(self.points)
            
            self.loaded = True
            print(f" -> {self.name}: {len(self.points)} points, spacing {self.spacing}")
            
            self._build_graph()
            self._detect_bifurcations()
            
        except Exception as e:
            print(f"ERROR loading {self.name}: {e}")
            self.loaded = False

    def _build_graph(self):
        if len(self.points) == 0: return
        G = nx.Graph()
        for i in range(len(self.points)):
            G.add_node(i, pos=self.points[i])
        
        radius_search_vox = 2.5 
        neighbors_list = self.tree.query_ball_tree(self.tree, r=radius_search_vox)
        
        scaled_pts = self.points * self.spacing
        for i, neighbors in enumerate(neighbors_list):
            for j in neighbors:
                if i < j:
                    dist = np.linalg.norm(scaled_pts[i] - scaled_pts[j])
                    G.add_edge(i, j, weight=dist)
        self.graph = G

    def _detect_bifurcations(self):
        if len(self.points) == 0: return
        pts_set = {tuple(p) for p in self.points}
        offsets = [(dz, dy, dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1) if not (dz==0 and dy==0 and dx==0)]
        candidates = []
        for idx, p in enumerate(self.points):
            z,y,x = int(p[0]), int(p[1]), int(p[2])
            neigh = 0
            for off in offsets:
                if (z+off[0], y+off[1], x+off[2]) in pts_set: neigh += 1
            if neigh >= 3: candidates.append(idx)
        if not candidates: return
        cand_phys = self.points[candidates] * self.spacing
        tree = cKDTree(cand_phys)
        groups = tree.query_ball_tree(tree, r=BIFURCATION_MERGE_RADIUS_MM)
        visited = set()
        bif_indices = []
        bif_locs = []
        for i in range(len(candidates)):
            if i in visited: continue
            cluster = []
            stack = [i]
            while stack:
                curr = stack.pop()
                if curr in visited: continue
                visited.add(curr)
                cluster.append(curr)
                for n in groups[curr]:
                    if n not in visited: stack.append(n)
            glob_indices = [candidates[x] for x in cluster]
            phys_cluster = self.points[glob_indices] * self.spacing
            centroid = np.mean(phys_cluster, axis=0)
            dists = np.linalg.norm(phys_cluster - centroid, axis=1)
            rep_idx = glob_indices[np.argmin(dists)]
            bif_indices.append(rep_idx)
            bif_locs.append(centroid)
        self.bifurcations = (bif_indices, np.vstack(bif_locs) if bif_locs else np.empty((0,3)))

    def get_vedos(self, color_mesh, color_skel):
        if not self.loaded: return []
        if self.mesh is None:
            self.mesh = vedo.Volume(self.mask).isosurface().c(color_mesh).alpha(0.2)
        if self.skel_actor is None:
            self.skel_actor = vedo.Points(self.points, r=3, c=color_skel)
        return [self.mesh, self.skel_actor]

class AnalyzerApp:
    def __init__(self, left_path, right_path):
        self.left = Artery("Left", left_path)
        self.right = Artery("Right", right_path)
        
        self.selected_pts = {'left': [], 'right': []} 
        self.visuals = {'left': [], 'right': [], 'stenosis': []}
        self.stenosis_results = []
        
        self.cursor_sphere = None
        self.cursor_text = None
        
        self.plt = vedo.Plotter(title="Analiza Tetnic | [R] Reset", 
                                axes=1, bg='white', size=(1200, 900))
        
        self.left.load()
        self.right.load()
        
        actors = []
        actors.extend(self.left.get_vedos('lightgreen', 'darkgreen'))
        actors.extend(self.right.get_vedos('lightblue', 'darkblue'))
        self.plt.add(actors)
        
        self.status_text = vedo.Text2D("Wybierz punkty:\nLewa (L): 3 pkt\nPrawa (P): 2 pkt", 
                                     pos='top-left', s=0.8, bg='yellow', alpha=0.5)
        self.plt.add(self.status_text)

        self.plt.add_callback('LeftButtonPress', self.on_click)
        self.plt.add_callback('MouseMove', self.on_move)
        self.plt.add_callback('KeyPress', self.on_key)

    def run(self):
        self.plt.show(resetcam=True, interactive=True)

    def reset(self, side='all'):
        print(f"Resetting {side}...")
        sides = ['left', 'right'] if side == 'all' else [side]
        for s in sides:
            self.plt.remove(self.visuals[s])
            self.visuals[s] = []
            self.selected_pts[s] = []
        
        if side == 'all':
            self.plt.remove(self.visuals['stenosis'])
            self.visuals['stenosis'] = []
            self.stenosis_results = []
            self.status_text.text("Zresetowano. Wybierz punkty.")
            
        self.plt.render()

    def determine_segment_name(self, current_dist, sorted_bifurcation_dists):
        if not sorted_bifurcation_dists: return "Segment"
        tolerance = 2.0 
        if current_dist < (sorted_bifurcation_dists[0] + tolerance): return "Proximal"
        if len(sorted_bifurcation_dists) > 1:
            if current_dist < (sorted_bifurcation_dists[1] + tolerance): return "Mid"
            else: return "Distal"
        return "Distal"

    def calculate_paths(self, artery_obj, side_key):
        pts_indices = self.selected_pts[side_key]
        start_node = pts_indices[0]
        end_nodes = pts_indices[1:]
        
        print(f"\n--- ANALIZA: {artery_obj.name} ---")
        self.status_text.text(f"Przetwarzanie {artery_obj.name}...")
        self.plt.render()
        
        path_counter = 1
        for end_node in end_nodes:
            try:
                path_idxs = nx.shortest_path(artery_obj.graph, start_node, end_node)
            except nx.NetworkXNoPath:
                print(f"Brak ścieżki w grafie {artery_obj.name}!")
                continue

            path_pts = artery_obj.points[path_idxs]
            
            # Obliczamy średnice
            diameters = []
            for p in path_pts:
                d = adaptive_diameter_calculation(p, artery_obj.dist_map, artery_obj.skeleton, 
                                                artery_obj.points, artery_obj.spacing)
                diameters.append(d)
            
            # Wygładzamy średnice, żeby "szum" nie wyznaczył błędnego punktu odcięcia
            diameters_smooth = gaussian_filter1d(diameters, sigma=2.0)

            _, cum_orig, _, _, _ = compute_upsampled_path(path_pts, artery_obj.spacing)
            
            # =========================================================
            # 1. WYZNACZANIE PUNKTU ODCIĘCIA (CUT-OFF) 1.5 mm
            # =========================================================
            # Szukamy od końca punktu, gdzie naczynie jest jeszcze "grube" (>1.5mm)
            valid_end_idx = len(diameters) - 1 # Domyślnie cała ścieżka ważna
            
            # Idziemy od tyłu
            for i in range(len(diameters) - 1, -1, -1):
                if diameters_smooth[i] >= STENOSIS_IGNORE_DIAMETER_MM:
                    valid_end_idx = i
                    break # Znaleźliśmy punkt, gdzie naczynie robi się "istotne"
            
            valid_end_dist = cum_orig[valid_end_idx]
            print(f" -> Punkt odcięcia dystalnego (<{STENOSIS_IGNORE_DIAMETER_MM}mm): {valid_end_dist:.1f} mm")
            
            # Zaznaczamy ten punkt na wykresie 3D (opcjonalnie - np. szara kulka)
            cutoff_pos = path_pts[valid_end_idx]
            cutoff_sph = vedo.Sphere(cutoff_pos, r=2.0, c='gray').alpha(0.5)
            self.plt.add(cutoff_sph)
            self.visuals[side_key].append(cutoff_sph)
            # =========================================================

            regions, diag = detect_local_stenosis_with_grad(
                diameters, artery_obj.spacing,
                window_mm=STENOSIS_WINDOW_MM,
                min_stenosis=STENOSIS_MIN_PCT,
                min_length_mm=STENOSIS_MIN_LEN_MM,
                use_gradient=True,
                grad_thresh=GRADIENT_THRESH,
                x_positions=cum_orig
            )
            
            line = vedo.Line(path_pts).lw(6)
            line.cmap('jet', diameters) 
            
            if path_counter == 1:
                if side_key == 'left':
                    pos_coords = ((0.05, 0.05), (0.20, 0.25))
                else:
                    pos_coords = ((0.80, 0.05), (0.95, 0.25))
                line.add_scalarbar(title=f"Srednica {artery_obj.name} [mm]", pos=pos_coords)
            
            self.plt.add(line)
            self.visuals[side_key].append(line)
            
            path_bifs_dist = []
            all_bif_indices, _ = artery_obj.bifurcations
            node_to_dist = {node: dist for node, dist in zip(path_idxs, cum_orig)}
            
            for bif_idx in all_bif_indices:
                if bif_idx in node_to_dist:
                    path_bifs_dist.append(node_to_dist[bif_idx])
            path_bifs_dist.sort()
            
            print(f"\n[Sciezka {path_counter}] Dlugosc: {cum_orig[-1]:.1f} mm")

            for reg in regions:
                start_idx = reg['start_idx']
                start_dist = cum_orig[start_idx]
                
                ignored = False
                
                # 1. Filtr Bifurkacji
                for bd in path_bifs_dist:
                    if bd <= start_dist and (start_dist - bd) <= BIFURCATION_BUFFER_MM:
                        ignored = True; break
                
                # 2. Filtr Dystalnego Odcięcia (NOWY)
                if not ignored:
                    # Jeśli początek zwężenia jest DALEJ niż punkt odcięcia -> Ignoruj
                    if start_idx > valid_end_idx:
                        print(f" -> [IGNOROWANE] Zwężenie w końcówce dystalnej (Poz: {start_dist:.1f}mm > Cutoff: {valid_end_dist:.1f}mm)")
                        ignored = True
                
                if not ignored:
                    sten_val = reg['max_stenosis']
                    seg_name = self.determine_segment_name(start_dist, path_bifs_dist)
                    
                    print(f" -> ZWEZENIE ({seg_name}): {sten_val:.1f}% | Poz: {start_dist:.1f}mm")

                    idx_local = reg['max_stenosis_idx']
                    pos_3d = path_pts[idx_local]
                    sph = vedo.Sphere(pos_3d, r=3.5, c='red').alpha(0.8)
                    self.plt.add(sph)
                    self.visuals['stenosis'].append(sph)

                    self.stenosis_results.append({
                        'artery': artery_obj.name,
                        'path': path_counter,
                        'side': 'L' if side_key=='left' else 'P',
                        'segment': seg_name,
                        'stenosis': sten_val
                    }) 

            save_dir = "gradient_analysis"
            os.makedirs(save_dir, exist_ok=True)
            plot_graph_thread(diameters, cum_orig, diag, regions, 
                              f"{artery_obj.name} Path {path_counter}", 
                              f"{save_dir}/{artery_obj.name}_path{path_counter}.png")
            
            path_counter += 1
        
        print(f"Zakonczono analize {artery_obj.name}.")
        self.status_text.text(f"Gotowe: {artery_obj.name}. Wyniki w konsoli.")
        self.cadrads_calc()
        self.plt.render()

    def cadrads_calc(self):
        if not self.stenosis_results: return
        
        print("\n" + "="*40)
        print("     RAPORT ZWĘŻEŃ (Wsparcie Decyzji)")
        print("="*40)
        print(f"{'Naczynie':<10} | {'Segment':<10} | {'Zwężenie [%]':<12} | {'Klasa'}")
        print("-" * 50)

        for res in self.stenosis_results:
            s = res['stenosis']
            cls_suggestion = "1-2"
            if s >= 70: cls_suggestion = "4A (High)"
            elif s >= 50: cls_suggestion = "3 (Mod)"
            elif s >= 25: cls_suggestion = "2 (Mild)"
            
            print(f"{res['artery']:<10} | {res['segment']:<10} | {s:5.1f}%       | {cls_suggestion}")

        print("-" * 50)
        print("\n[!!!] UWAGA KLINICZNA / DISCLAIMER:")
        print("Algorytm analizuje wyłącznie geometrię drożnego światła naczynia.")
        print("Wykryte zwężenia (np. 50-60%) mogą towarzyszyć całkowitej OKLUZJI (100%)")
        print("w innym segmencie, której program geometryczny może nie wykryć.")
        print("\nZALECENIE: Weryfikacja wizualna pod kątem CAD-RADS 5 (Total Occlusion).")
        print("="*40 + "\n")

    def on_click(self, event):
        if not event.actor or event.keypress: return
        click_pos = event.picked3d
        if click_pos is None: return

        if self.visuals['stenosis']:
            st_pos = [np.array(s.pos()) for s in self.visuals['stenosis']]
            if not st_pos: return
            tree = cKDTree(st_pos)
            d, idx = tree.query(click_pos)
            if d < 8.0: 
                removed = self.visuals['stenosis'].pop(idx)
                self.plt.remove(removed)
                print("Usunięto manualnie znacznik zwężenia.")
                self.plt.render()
                return

        dist_l = np.inf
        dist_r = np.inf
        idx_l, idx_r = -1, -1
        
        if self.left.loaded:
            dist_l, idx_l = self.left.tree.query(click_pos)
        if self.right.loaded:
            dist_r, idx_r = self.right.tree.query(click_pos)
        
        if min(dist_l, dist_r) > 15.0: return 

        target = self.left if dist_l < dist_r else self.right
        side_key = 'left' if dist_l < dist_r else 'right'
        max_pts = 3 if side_key == 'left' else 2
        
        final_idx = idx_l if dist_l < dist_r else idx_r

        if len(self.selected_pts[side_key]) >= max_pts:
            print(f"Limit punktów! Resetuj.")
            return

        if final_idx in self.selected_pts[side_key]: return
        
        self.selected_pts[side_key].append(final_idx)
        pt_coord = target.points[final_idx]
        
        colors = ['green', 'yellow', 'orange']
        col = colors[len(self.selected_pts[side_key])-1]
        sph = vedo.Sphere(pt_coord, r=2.5, c=col)
        self.plt.add(sph)
        self.visuals[side_key].append(sph)
        
        print(f"[{target.name}] Punkt {len(self.selected_pts[side_key])}/{max_pts}")

        if len(self.selected_pts[side_key]) == max_pts:
            self.calculate_paths(target, side_key)
            
        self.plt.render()

    def on_move(self, event):
        if event.picked3d is None:
            if self.cursor_sphere: 
                self.plt.remove(self.cursor_sphere)
                self.cursor_sphere = None
            if self.cursor_text: 
                self.plt.remove(self.cursor_text)
                self.cursor_text = None
            self.plt.render()
            return

        pos = event.picked3d
        
        d_left, idx_left = self.left.tree.query(pos) if self.left.tree else (np.inf, -1)
        d_right, idx_right = self.right.tree.query(pos) if self.right.tree else (np.inf, -1)
        
        if min(d_left, d_right) > 10.0:
            if self.cursor_sphere: 
                self.plt.remove(self.cursor_sphere)
                self.cursor_sphere = None
            if self.cursor_text:
                self.plt.remove(self.cursor_text)
                self.cursor_text = None
            self.plt.render()
            return

        target_artery = self.left if d_left < d_right else self.right
        target_idx = idx_left if d_left < d_right else idx_right
        
        voxel_pt = target_artery.points[target_idx]
        z,y,x = int(voxel_pt[0]), int(voxel_pt[1]), int(voxel_pt[2])
        diam = 0.0
        if (0 <= z < target_artery.dist_map.shape[0] and 
            0 <= y < target_artery.dist_map.shape[1] and 
            0 <= x < target_artery.dist_map.shape[2]):
            diam = target_artery.dist_map[z,y,x] * 2.0

        if self.cursor_sphere: 
            self.plt.remove(self.cursor_sphere)
        
        self.cursor_sphere = vedo.Sphere(pos=voxel_pt, r=1.5, c='white', alpha=0.5)
        self.plt.add(self.cursor_sphere)
        
        if self.cursor_text: 
            self.plt.remove(self.cursor_text)
        
        self.cursor_text = vedo.Text2D(f"{target_artery.name}\nD={diam:.2f} mm", pos='top-left', c='black', bg='white')
        self.plt.add(self.cursor_text)
        
        self.plt.render()

    def on_key(self, event):
        if event.keypress == 'r': self.reset('all')
        elif event.keypress == 'l': self.reset('left')
        elif event.keypress == 'p': self.reset('right')

# ================= MAIN =================
if __name__ == "__main__":
    # ZMIEŃ ŚCIEŻKI DO SWOICH PLIKÓW
    LEFT_FILE = r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_left.nrrd"
    RIGHT_FILE = r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_right.nrrd"

    if not os.path.exists(LEFT_FILE): print(f"UWAGA: Brak {LEFT_FILE}")
    if not os.path.exists(RIGHT_FILE): print(f"UWAGA: Brak {RIGHT_FILE}")

    app = AnalyzerApp(LEFT_FILE, RIGHT_FILE)
    app.run()