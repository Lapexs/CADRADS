import numpy as np
import itertools
import time
import os
import sys
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================
# IMPORTY Z PLIKU GŁÓWNEGO
# =============================================================================
from gradient2 import Artery, compute_upsampled_path, adaptive_diameter_calculation, detect_local_stenosis_with_grad


# =============================================================================
# 1. ZAKRESY POSZUKIWAŃ (BARDZO GĘSTE - TAK JAK CHCIAŁEŚ)
# Uwaga: Liczba kombinacji to iloczyn długości tych tablic!
# =============================================================================
SEARCH_SPACE = {
    # Window: od 10 do 20 co 1.0 (10, 11, ... 20) -> 11 wartości
    'WINDOW':  np.arange(7, 20, 1.0), 
    
    # Length: od 2.0 do 4.0 co 0.2 (2.0, 2.2, ... 4.0) -> 11 wartości
    'LEN_MM':  np.arange(2.0, 4.0, 0.1), 
    
    # Percent: od 20 do 30 co 2.5 (20, 22.5, 25, 27.5, 30) -> 5 wartości
    'PCT':     np.arange(15, 30, 1),   
    
    # Gradient: od -0.10 do -0.20 co 0.02 -> 6 wartości
    'GRAD':    np.arange(-0.10, -0.20, -0.01) 
}

# Razem: 11 * 11 * 5 * 6 = ~3,630 kombinacji.
# To jest wykonalne (powinno zająć ok. 30-60 minut na dobrym PC).

# =============================================================================
# 2. DANE PACJENTÓW (WKLEJ SWOJE DANE TUTAJ)
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
# 3. PRZYGOTOWANIE DANYCH (CACHE)
# =============================================================================
def prepare_data_per_patient():
    print("-> [CACHE] Ładowanie danych do pamięci...")
    data_by_patient = {}
    
    for patient_id, arteries in PATIENTS.items():
        print(f"   Wczytuję: {patient_id}")
        patient_data = []
        for conf in arteries:
            if not os.path.exists(conf['file_path']): continue
            try:
                a = Artery("X", conf['file_path'])
                a.load()
                if conf['start_node'] in a.graph and conf['end_node'] in a.graph:
                    path_idxs = nx.shortest_path(a.graph, conf['start_node'], conf['end_node'])
                    path_pts = a.points[path_idxs]
                    # Pre-obliczanie średnic, żeby nie robić tego w pętli
                    diams = [adaptive_diameter_calculation(p, a.dist_map, a.skeleton, a.points, a.spacing) for p in path_pts]
                    _, cum_orig, _, _, _ = compute_upsampled_path(path_pts, a.spacing)
                    
                    patient_data.append({
                        "diameters": diams,
                        "spacing": a.spacing,
                        "cum_orig": cum_orig,
                        "true_stenoses": conf['true_stenoses_mm']
                    })
            except Exception as e:
                print(f"Błąd u {patient_id}: {e}")
        data_by_patient[patient_id] = patient_data
    return data_by_patient

# =============================================================================
# 4. FUNKCJA OCENY (LOGIKA HYBRYDOWA)
# =============================================================================
TOLERANCE_MM = 10.0

def evaluate_combination(data_by_patient, win, length, pct, grad):
    f1_scores = []
    
    for pid, arteries_data in data_by_patient.items():
        tp, fp, fn = 0, 0, 0
        total_true = 0
        
        for artery in arteries_data:
            total_true += len(artery['true_stenoses'])
            
            # Detekcja (Rozpakowanie krotki!)
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
            
            tp += local_tp
            fp += local_fp
            fn += len(true_rem)
            
        # Ocena hybrydowa
        if total_true == 0: 
            score = 1.0 if fp == 0 else 0.0
        else:
            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
            rec = tp/(tp+fn) if (tp+fn) > 0 else 0
            score = 2*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
            
        f1_scores.append(score)
        
    return np.mean(f1_scores)

def plot_heatmap(data_results, param_x, param_y, fixed_params, filename):
    print(f"   -> Generuję mapę: {param_x} vs {param_y}...")
    filtered = []
    for score, params in data_results:
        match = True
        for k, v in fixed_params.items():
            if abs(params[k] - v) > 0.001: match = False; break
        if match: filtered.append((params[param_x], params[param_y], score))

    if not filtered:
        print(f"      [OSTRZEŻENIE] Brak danych dla mapy! Sprawdź fixed_params.")
        return

    vals_x = sorted(list(set([f[0] for f in filtered])))
    vals_y = sorted(list(set([f[1] for f in filtered])))
    if param_x == 'GRAD': vals_x.sort(reverse=True)
    if param_y == 'GRAD': vals_y.sort(reverse=True)

    matrix = np.zeros((len(vals_y), len(vals_x)))
    for x_val, y_val, score in filtered:
        try:
            ix = vals_x.index(x_val); iy = vals_y.index(y_val)
            matrix[iy, ix] = score
        except: continue

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='inferno', origin='lower', aspect='auto') 
    
    ax.set_xticks(np.arange(len(vals_x)))
    ax.set_yticks(np.arange(len(vals_y)))
    ax.set_xticklabels([f"{v:.2f}" for v in vals_x], rotation=45)
    ax.set_yticklabels([f"{v:.2f}" for v in vals_y])
    ax.set_xlabel(param_x, fontsize=12); ax.set_ylabel(param_y, fontsize=12)
    
    fixed_str = ", ".join([f"{k}={v}" for k,v in fixed_params.items()])
    ax.set_title(f"Mapa Stabilności: {param_x} vs {param_y}\n(Przy stałych: {fixed_str})", fontsize=14)
    plt.colorbar(im, ax=ax).ax.set_ylabel("F1-Score", rotation=-90, va="bottom")

    if len(vals_x) < 20:
        for i in range(len(vals_y)):
            for j in range(len(vals_x)):
                val = matrix[i, j]
                text_color = "black" if val > 0.7 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close()
    print(f"   [OK] Zapisano: {filename}")

def plot_parallel_coordinates(results, filename):
    print(f"   -> Generuję wykres Parallel Coordinates: {filename}...")
    data = []
    for score, params in results:
        row = params.copy(); row['F1'] = score; data.append(row)
    
    df = pd.DataFrame(data)
    threshold_f1 = df['F1'].quantile(0.80) 
    df_filtered = df[df['F1'] >= threshold_f1].copy().sort_values(by='F1')

    cols = ['WINDOW', 'LEN_MM', 'PCT', 'GRAD', 'F1']
    min_max = {c: (df[c].min(), df[c].max()) for c in cols}
    df_norm = df_filtered.copy()
    for c in cols:
        if min_max[c][1] != min_max[c][0]:
            df_norm[c] = (df_filtered[c] - min_max[c][0]) / (min_max[c][1] - min_max[c][0])
        else: df_norm[c] = 0

    fig, ax = plt.subplots(figsize=(15, 8))
    for i in range(len(df_norm)):
        row = df_norm.iloc[i]
        ax.plot(range(len(cols)), row, color=plt.cm.plasma(row['F1']), alpha=0.5)

    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=12, fontweight='bold')
    for i, col in enumerate(cols):
        ax.axvline(i, color='black', linewidth=1)
        ax.text(i, -0.05, f"{min_max[col][0]:.2f}", ha='center', va='top')
        ax.text(i, 1.05, f"{min_max[col][1]:.2f}", ha='center', va='bottom')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=df_filtered['F1'].min(), vmax=df_filtered['F1'].max()))
    plt.colorbar(sm, ax=ax).set_label('F1-Score', rotation=270, labelpad=15)
    plt.title(f"Wykres Współrzędnych Równoległych (TOP 20% Wyników)", fontsize=16)
    
    ax.set_yticks([]); ax.spines['top'].set_visible(False); ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close()
    print(f"   [OK] Zapisano: {filename}")


# =============================================================================
# 5. GŁÓWNA PĘTLA GRID SEARCH
# =============================================================================
if __name__ == "__main__":
    # 1. Ładowanie danych
    data_by_patient = prepare_data_per_patient()
    
    if not data_by_patient:
        print("Brak danych!")
        sys.exit()

    # 2. Generowanie siatki
    keys = sorted(SEARCH_SPACE.keys())
    # itertools.product tworzy wszystkie możliwe kombinacje list
    combinations = list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
    total = len(combinations)
    
    print("\n" + "="*60)
    print(f"START GRID SEARCH")
    print(f"Liczba kombinacji: {total}")
    print(f"Parametry: {keys}")
    print("="*60)
    
    results = []
    start_time = time.time()
    
    # 3. Pętla (Mielenie)
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        score = evaluate_combination(
            data_by_patient, 
            win=params['WINDOW'], 
            length=params['LEN_MM'], 
            pct=params['PCT'], 
            grad=params['GRAD']
        )
        
        results.append((score, params))
        
        # Pasek postępu co 5%
        if i % max(1, int(total/20)) == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = (total - i) * avg_time
            print(f"Postęp: {i}/{total} ({i/total*100:.1f}%) | ETA: {eta/60:.1f} min | Ost. F1: {score:.3f}")

    total_time = time.time() - start_time
    print(f"\n-> Zakończono w {total_time:.1f} sekund.")
    
    # 4. Wyniki
    # Sortujemy malejąco po wyniku F1
    results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n" + "="*60)
    print("   TOP 10 NAJLEPSZYCH KOMBINACJI")
    print("="*60)
    for i in range(min(10, len(results))):
        score, p = results[i]
        print(f"{i+1}. F1={score:.4f} | Win={p['WINDOW']:.1f}, Len={p['LEN_MM']:.1f}, Pct={p['PCT']:.1f}, Grad={p['GRAD']:.2f}")
    
    # Zapis do pliku txt
    with open("wyniki_grid_search.txt", "w") as f:
        for i, (score, p) in enumerate(results):
            f.write(f"{i+1}. F1={score:.4f} | {p}\n")
    print("\nPełne wyniki zapisano w 'wyniki_grid_search.txt'")

    print("\nGENEROWANIE WYKRESÓW...")
    
    # Przykładowe punkty "stałe" do map ciepła - musisz wybrać jedną wartość z zakresu, żeby przekrój zadziałał
    # Wybieramy wartości "środkowe" lub Twoje ulubione, o ile istnieją w siatce
    # Najbezpieczniej wziąć te z najlepszego wyniku
    best_params = results[0][1]
    
    # 1. HEATMAPA: WINDOW vs GRAD (przy najlepszym Len i Pct)
    plot_heatmap(results, 'WINDOW', 'GRAD', 
                 {'LEN_MM': best_params['LEN_MM'], 'PCT': best_params['PCT']}, 
                 'gs_heatmap_win_grad.png')
    
    # 2. HEATMAPA: LEN vs PCT (przy najlepszym Win i Grad)
    plot_heatmap(results, 'LEN_MM', 'PCT', 
                 {'WINDOW': best_params['WINDOW'], 'GRAD': best_params['GRAD']}, 
                 'gs_heatmap_len_pct.png')
    
    # 3. PARALLEL COORDINATES
    try:
        plot_parallel_coordinates(results, 'gs_parallel_coords.png')
    except NameError:
        print("Brak pandas - pomijam wykres równoległy.")
    except Exception as e:
        print(f"Błąd rysowania Parallel Coords: {e}")