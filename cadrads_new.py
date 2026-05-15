import numpy as np
import networkx as nx
import os
import matplotlib
matplotlib.use('Agg') # Generowanie w tle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# 1. IMPORTY
# =============================================================================

from gradient2 import Artery, compute_upsampled_path, adaptive_diameter_calculation, detect_local_stenosis_with_grad


# =============================================================================
# 2. TWÓJ "ZŁOTY STANDARD" (PUNKT PRACY)
# To są parametry, które wybrałeś jako najlepsze.
# Wszystkie wykresy będą pokazywać odchylenia od TEGO punktu.
# =============================================================================
BEST_WINDOW = 11.0
BEST_LEN_MM = 3.0
BEST_PCT    = 24.0
BEST_GRAD   = -0.19

TOLERANCE_MM = 10.0

# =============================================================================
# 3. DANE PACJENTÓW (WKLEJ SWOJĄ LISTĘ)
# =============================================================================
PATIENTS = {
    "P1": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_right.nrrd", "start_node": 329, "end_node": 595, "true_stenoses_mm": []},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_left.nrrd", "start_node": 0, "end_node": 328, "true_stenoses_mm": []},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_left.nrrd", "start_node": 0, "end_node": 163, "true_stenoses_mm": []}
    ],
    "P2": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_right.nrrd", "start_node": 398, "end_node": 705, "true_stenoses_mm": []},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_left.nrrd", "start_node": 10, "end_node": 779, "true_stenoses_mm": []},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_left.nrrd", "start_node": 0, "end_node": 1014, "true_stenoses_mm": []}
    ],
    "P3": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_right.nrrd", "start_node": 275, "end_node": 463, "true_stenoses_mm": []},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_left.nrrd", "start_node": 1, "end_node": 239, "true_stenoses_mm": []},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_left.nrrd", "start_node": 1, "end_node": 612, "true_stenoses_mm": []}
    ],
    "P4": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_right.nrrd", "start_node": 323, "end_node": 515, "true_stenoses_mm": []},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_left.nrrd", "start_node": 0, "end_node": 669, "true_stenoses_mm": [55.6]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_left.nrrd", "start_node": 0, "end_node": 775, "true_stenoses_mm": [90.6]}
    ],
    "P5": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_right.nrrd", "start_node": 298, "end_node": 462, "true_stenoses_mm": [0.0]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_left.nrrd", "start_node": 0, "end_node": 460, "true_stenoses_mm": [10.7]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_left.nrrd", "start_node": 0, "end_node": 184, "true_stenoses_mm": []}
    ],
    "P6": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_right.nrrd", "start_node": 325, "end_node": 647, "true_stenoses_mm": [46.9]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_left.nrrd", "start_node": 0, "end_node": 499, "true_stenoses_mm": [45.7, 83.4]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_left.nrrd", "start_node": 0, "end_node": 265, "true_stenoses_mm": []}
    ],
    "P7": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_right.nrrd", "start_node": 314, "end_node": 610, "true_stenoses_mm": [1.3]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_left.nrrd", "start_node": 0, "end_node": 625, "true_stenoses_mm": [18.6, 56.9]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_left.nrrd", "start_node": 0, "end_node": 990, "true_stenoses_mm": [29.6,39.7]}
    ],
    "P8": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_right.nrrd", "start_node": 183, "end_node": 360, "true_stenoses_mm": [5.0,30.5,90.7]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_left.nrrd", "start_node": 0, "end_node": 527, "true_stenoses_mm": [8.4, 41.6,88.2]}, 
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_left.nrrd", "start_node": 0, "end_node": 357, "true_stenoses_mm": []}
    ],
    "P9": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_right.nrrd", "start_node": 210, "end_node": 409, "true_stenoses_mm": [32.4,124.7]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_left.nrrd", "start_node": 59, "end_node": 594, "true_stenoses_mm": [98.7]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_left.nrrd", "start_node": 59, "end_node": 857, "true_stenoses_mm": []}
    ],
    "P10": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_right.nrrd", "start_node": 599, "end_node": 868, "true_stenoses_mm": [21.6,36.4,89,110.7]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_left.nrrd", "start_node": 0, "end_node": 185, "true_stenoses_mm": [48,84.7]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_left.nrrd", "start_node": 0, "end_node": 518, "true_stenoses_mm": []}
    ],
                "P11": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0v2\Segmentation_right.nrrd", 
            "start_node": 194, 
            "end_node": 231, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0v2\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 310, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0v2\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 323, 
            "true_stenoses_mm": [] 
        }
    ],
            "P12": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0v3\Segmentation_right.nrrd", 
            "start_node": 252, 
            "end_node": 29, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0v3\Segmentation_left.nrrd", 
            "start_node": 32, 
            "end_node": 357, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0v3\Segmentation_left.nrrd", 
            "start_node": 32, 
            "end_node": 0, 
            "true_stenoses_mm": [] 
        }
    ],
                "P13": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1v2\Segmentation_right.nrrd", 
            "start_node": 297, 
            "end_node": 737, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1v2\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 395, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1v2\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 328, 
            "true_stenoses_mm": [] 
        }
    ],
                "P14": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1v3\Segmentation_right.nrrd", 
            "start_node": 441, 
            "end_node": 793, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1v3\Segmentation_left.nrrd", 
            "start_node": 1, 
            "end_node": 652, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1v3\Segmentation_left.nrrd", 
            "start_node": 1, 
            "end_node": 388, 
            "true_stenoses_mm": [] 
        }
    ],
                "P15": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2v2\Segmentation_right.nrrd", 
            "start_node": 312, 
            "end_node": 402, 
            "true_stenoses_mm": [59.9] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2v2\Segmentation_left.nrrd", 
            "start_node": 2, 
            "end_node": 403, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2v2\Segmentation_left.nrrd", 
            "start_node": 2, 
            "end_node": 244, 
            "true_stenoses_mm": [] 
        }
    ],
                "P16": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2v3\Segmentation_right.nrrd", 
            "start_node": 588, 
            "end_node": 812, 
            "true_stenoses_mm": [96.8] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2v3\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 820, 
            "true_stenoses_mm": [35.0,70.6] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2v3\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 686, 
            "true_stenoses_mm": [2.8] 
        }
    ],
                "P17": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v3\Segmentation_right.nrrd", 
            "start_node": 543, 
            "end_node": 26, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v3\Segmentation_left.nrrd", 
            "start_node": 163, 
            "end_node": 715, 
            "true_stenoses_mm": [36.2] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v3\Segmentation_left.nrrd", 
            "start_node": 163, 
            "end_node": 0, 
            "true_stenoses_mm": [] 
        }
    ],
                "P18": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v4\Segmentation_right.nrrd", 
            "start_node": 479, 
            "end_node": 972, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v4\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 963, 
            "true_stenoses_mm": [52.9,81.5] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v4\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 874, 
            "true_stenoses_mm": [56.3,70.5] 
        }
    ],
                "P19": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v3\Segmentation_right.nrrd", 
            "start_node": 540, 
            "end_node": 672, 
            "true_stenoses_mm": [38.1,106.2,131.9] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v3\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 721, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v3\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 956, 
            "true_stenoses_mm": [] 
        }
    ],
                "P20": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v3\Segmentation_right.nrrd", 
            "start_node": 450, 
            "end_node": 529, 
            "true_stenoses_mm": [62.7] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v3\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 758, 
            "true_stenoses_mm": [32] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v3\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 925, 
            "true_stenoses_mm": [19.7,36.7] 
        }
    ],
}

# =============================================================================
# 4. FUNKCJA OBLICZENIOWA (POPRAWIONA LOGIKA F1/ZDROWI)
# =============================================================================
def calculate_f1_scores_list(data_by_patient, win, pct, length, grad):
    """Zwraca listę wyników F1 dla każdego pacjenta osobno."""
    scores = []
    
    for pid, arteries_data in data_by_patient.items():
        tp_total, fp_total, fn_total = 0, 0, 0
        total_true_stenoses_count = 0
        
        for artery in arteries_data:
            total_true_stenoses_count += len(artery['true_stenoses'])
            
            # Detekcja (z rozpakowaniem krotki)
            regions, _ = detect_local_stenosis_with_grad(
                artery['diameters'], artery['spacing'],
                window_mm=win, min_stenosis=pct, min_length_mm=length,
                use_gradient=True, grad_thresh=grad, x_positions=artery['cum_orig']
            )
            
            detected_pos = [artery['cum_orig'][r['max_stenosis_idx']] for r in regions]
            true_rem = artery['true_stenoses'].copy()
            
            local_tp, local_fp = 0, 0
            for det in detected_pos:
                match = False
                for true_pos in true_rem:
                    if abs(det - true_pos) <= TOLERANCE_MM:
                        match = True; true_rem.remove(true_pos); break
                if match: local_tp += 1
                else: local_fp += 1
            
            tp_total += local_tp
            fp_total += local_fp
            fn_total += len(true_rem)
            
        # Ocena pacjenta
        if total_true_stenoses_count == 0:
            # Zdrowy: 1.0 jak nic nie wykryto, 0.0 jak fałszywy alarm
            score = 1.0 if fp_total == 0 else 0.0
        else:
            # Chory: klasyczne F1
            prec = tp_total/(tp_total+fp_total) if (tp_total+fp_total) > 0 else 0
            rec = tp_total/(tp_total+fn_total) if (tp_total+fn_total) > 0 else 0
            score = 2*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
            
        scores.append(score)
        
    return scores

# =============================================================================
# 5. FUNKCJA RYSOWANIA (ERROR BARS + SCATTER)
# =============================================================================
def make_plot(param_name, x_values, all_scores, filename, x_label, best_val_x):
    # Statystyki
    means = [np.mean(s) for s in all_scores]
    stds  = [np.std(s) for s in all_scores]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 1. Rysowanie punktów pacjentów (Scatter)
    for i, x in enumerate(x_values):
        y_vals = all_scores[i]
        # Jitter X (rozrzut)
        jitter = np.random.normal(0, (max(x_values)-min(x_values))*0.015, size=len(y_vals))
        ax.scatter(x + jitter, y_vals, color='gray', alpha=0.35, s=25, zorder=2)

    # 2. Rysowanie średniej i odchylenia (Error Bar)
    ax.errorbar(x_values, means, yerr=stds, fmt='o-', 
                color='#0056b3', ecolor='#d9534f', elinewidth=2, capsize=4, 
                markersize=6, zorder=3, label='Średnia ± Std Dev')

    # 3. Zaznaczenie wybranego OPTIMUM (czerwona kropka)
    # Znajdujemy indeks wartości najbliższej naszemu BEST_VAL
    try:
        best_idx = np.argmin(np.abs(np.array(x_values) - best_val_x))
        ax.scatter([x_values[best_idx]], [means[best_idx]], s=150, facecolors='none', edgecolors='red', linewidth=2, zorder=4, label='Wybrany Punkt Pracy')
    except:
        pass

    # Stylizacja
    ax.set_title(f'Wpływ parametru: {param_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Skuteczność (F1 / Swoistość)', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"[OK] Wykres zapisany: {filename}")
    plt.close()

# =============================================================================
# 6. GŁÓWNA PĘTLA
# =============================================================================
def prepare_data_per_patient():
    # ... (Tu wklejam tę samą funkcję wczytywania co w poprzednim kodzie, dla porządku skrócę wklejanie, ale musi tu być) ...
    # UŻYJ TEJ SAMEJ FUNKCJI WCZYTYWANIA CO WCZEŚNIEJ
    # DLA PEWNOŚCI WKLEJAM JĄ JESZCZE RAZ NA DOLE
    print("-> Ładowanie danych pacjentów...")
    data_by_patient = {}
    for patient_id, arteries in PATIENTS.items():
        print(f"   Wczytuję {patient_id}...")
        p_data = []
        for conf in arteries:
            if not os.path.exists(conf['file_path']): continue
            try:
                a = Artery("X", conf['file_path'])
                a.load()
                if conf['start_node'] in a.graph and conf['end_node'] in a.graph:
                    path_idxs = nx.shortest_path(a.graph, conf['start_node'], conf['end_node'])
                    path_pts = a.points[path_idxs]
                    diams = [adaptive_diameter_calculation(p, a.dist_map, a.skeleton, a.points, a.spacing) for p in path_pts]
                    _, cum_orig, _, _, _ = compute_upsampled_path(path_pts, a.spacing)
                    p_data.append({
                        "diameters": diams, "spacing": a.spacing, "cum_orig": cum_orig,
                        "true_stenoses": conf['true_stenoses_mm']
                    })
            except: pass
        data_by_patient[patient_id] = p_data
    return data_by_patient

if __name__ == "__main__":
    data_db = prepare_data_per_patient()
    if not data_db: exit()
    
    print("\nGenerowanie wykresów do pracy inżynierskiej...")

    # --- 1. WINDOW (Szerokość okna) ---
    print("-> Analiza Window...")
    # Zakres testowy: od 5 do 30 co 2.5
    rng_win = np.arange(7, 15, 1)
    res_win = []
    for val in rng_win:
        # Zmieniamy TYLKO Window, reszta BEST
        res_win.append(calculate_f1_scores_list(data_db, val, BEST_PCT, BEST_LEN_MM, BEST_GRAD))
    make_plot("Szerokość Okna", rng_win, res_win, "wykres_window.png", "Window [mm]", BEST_WINDOW)

    # --- 2. MIN LENGTH (Długość zwężenia) ---
    print("-> Analiza Length...")
    # Zakres: od 1.0 do 6.0 co 0.5
    rng_len = np.arange(2.0, 5.0, 0.1)
    res_len = []
    for val in rng_len:
        # Zmieniamy TYLKO Length
        res_len.append(calculate_f1_scores_list(data_db, BEST_WINDOW, BEST_PCT, val, BEST_GRAD))
    make_plot("Minimalna Długość", rng_len, res_len, "wykres_length.png", "Length [mm]", BEST_LEN_MM)

    # --- 3. PERCENTAGE (Próg zwężenia) ---
    print("-> Analiza Percentage...")
    # Zakres: od 15 do 45 co 2.5
    rng_pct = np.arange(15, 30, 1)
    res_pct = []
    for val in rng_pct:
        res_pct.append(calculate_f1_scores_list(data_db, BEST_WINDOW, val, BEST_LEN_MM, BEST_GRAD))
    make_plot("Próg Zwężenia", rng_pct, res_pct, "wykres_pct.png", "Stenosis [%]", BEST_PCT)

    # --- 4. GRADIENT ---
    print("-> Analiza Gradient...")
    # Zakres: od -0.05 do -0.30 co -0.02
    rng_grad = np.arange(-0.10, -0.25, -0.01)
    # Sortujemy malejąco (dla ładnego wykresu, chociaż osie i tak się ustawią)
    res_grad = []
    for val in rng_grad:
        res_grad.append(calculate_f1_scores_list(data_db, BEST_WINDOW, BEST_PCT, BEST_LEN_MM, val))
    make_plot("Próg Gradientu", rng_grad, res_grad, "wykres_gradient.png", "Gradient [mm/mm]", BEST_GRAD)

    print("\nGotowe! Wykresy w folderze.")