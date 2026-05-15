import numpy as np
import networkx as nx
import os
import matplotlib
matplotlib.use('Agg') # Tryb bezokienkowy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# IMPORTY POMOCNICZE
from gradient2 import Artery, compute_upsampled_path, adaptive_diameter_calculation

# =============================================================================
# 1. KONFIGURACJA PACJENTÓW (STRUKTURA DRZEWA)
# =============================================================================

PATIENTS = {
    # --- PACJENT 1 (Zdefiniuj ID, np. "P1") ---
    "P1": [
        { 
            "name": "RCA",
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_right.nrrd", 
            "start_node": 329, 
            "end_node": 595, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 328, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 163, 
            "true_stenoses_mm": [] 
        }
    ],
    
    "P2": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_right.nrrd", 
            "start_node": 398, 
            "end_node": 705, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_left.nrrd", 
            "start_node": 10, 
            "end_node": 779, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 1014, 
            "true_stenoses_mm": [] 
        }
    ],
    
    "P3": [
        { 
            "name": "RCA",
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_right.nrrd", 
            "start_node": 275, 
            "end_node": 463, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_left.nrrd", 
            "start_node": 1, 
            "end_node": 239, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_left.nrrd", 
            "start_node": 1, 
            "end_node": 612, 
            "true_stenoses_mm": [] 
        }
    ],

    "P4": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_right.nrrd", 
            "start_node": 323, 
            "end_node": 515, 
            "true_stenoses_mm": [] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 669, 
            "true_stenoses_mm": [55.6] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 775, 
            "true_stenoses_mm": [90.6] 
        }
    ],

    "P5": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_right.nrrd", 
            "start_node": 298, 
            "end_node": 462, 
            "true_stenoses_mm": [0.0] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 460, 
            "true_stenoses_mm": [10.7] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 184, 
            "true_stenoses_mm": [] 
        }
    ],

    "P6": [
        { 
            "name": "RCA",
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_right.nrrd", 
            "start_node": 325, 
            "end_node": 647, 
            "true_stenoses_mm": [46.9] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 499, 
            "true_stenoses_mm": [45.7, 83.4] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 265, 
            "true_stenoses_mm": [] 
        }
    ],
    "P7": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_right.nrrd", 
            "start_node": 314, 
            "end_node": 610, 
            "true_stenoses_mm": [1.3] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 625, 
            "true_stenoses_mm": [18.6, 56.9] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 990, 
            "true_stenoses_mm": [29.6,39.7] 
        }
    ],
    "P8": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_right.nrrd", 
            "start_node": 183, 
            "end_node": 360, 
            "true_stenoses_mm": [5.0,30.5,90.7] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 527, 
            "true_stenoses_mm": [8.4, 41.6,88.2,133.9] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 357, 
            "true_stenoses_mm": [] 
        }
    ],
    "P9": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_right.nrrd", 
            "start_node": 210, 
            "end_node": 409, 
            "true_stenoses_mm": [32.4,124.7] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_left.nrrd", 
            "start_node": 59, 
            "end_node": 594, 
            "true_stenoses_mm": [98.7] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_left.nrrd", 
            "start_node": 59, 
            "end_node": 857, 
            "true_stenoses_mm": [] 
        }
    ],
        "P10": [
        { 
            "name": "RCA", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_right.nrrd", 
            "start_node": 599, 
            "end_node": 868, 
            "true_stenoses_mm": [21.6,36.4,89,110.7] 
        },
        { 
            "name": "LAD", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 185, 
            "true_stenoses_mm": [48,84.7] 
        },
        { 
            "name": "LCx", 
            "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_left.nrrd", 
            "start_node": 0, 
            "end_node": 518, 
            "true_stenoses_mm": [] 
        }
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


RUN_ONLY = [] 

# ============================================================
# 3. FUNKCJA DETEKCJI (WERSJA MM)
# ============================================================
def detect_local_stenosis_with_grad(diameters, spacing, window_mm=15, min_stenosis=20, 
                                  min_length_mm=3.0, use_gradient=False, grad_thresh=-0.15, 
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
                    regions.append({'max_stenosis_idx': int(s + max_idx_local)})
    return regions

# ================= PARAMETRY BAZOWE =================
TOLERANCE_MM = 10.0 
BASE_WINDOW = 10    
BASE_LEN_MM = 3.0   
BASE_PCT = 24.0     
BASE_GRAD = -0.19   

# ================= SILNIK TESTOWY =================
def prepare_data():
    print("-> Ładowanie danych pacjentów...")
    prepared = []
    
    # Iterujemy po słowniku PATIENTS
    for patient_id, arteries in PATIENTS.items():
        
        # Filtracja (Biała Lista)
        if RUN_ONLY and patient_id not in RUN_ONLY:
            continue 

        print(f"   [PACJENT: {patient_id}]")
        
        for artery_conf in arteries:
            fpath = artery_conf['file_path']
            if not os.path.exists(fpath):
                print(f"      BŁĄD: Brak pliku {fpath}")
                continue

            try:
                print(f"      -> Wczytuje tętnicę: {artery_conf['name']}")
                artery = Artery("Test", fpath)
                artery.load()
                
                start, end = artery_conf['start_node'], artery_conf['end_node']
                
                if start in artery.graph and end in artery.graph:
                    path_idxs = nx.shortest_path(artery.graph, start, end)
                    path_pts = artery.points[path_idxs]
                    diams = [adaptive_diameter_calculation(p, artery.dist_map, artery.skeleton, artery.points, artery.spacing) for p in path_pts]
                    _, cum_orig, _, _, _ = compute_upsampled_path(path_pts, artery.spacing)
                    
                    prepared.append({
                        "patient": patient_id,
                        "artery": artery_conf['name'],
                        "diameters": diams, 
                        "spacing": artery.spacing, 
                        "cum_orig": cum_orig, 
                        "true_stenoses": artery_conf['true_stenoses_mm']
                    })
                else:
                    print(f"      BŁĄD: Węzły {start}->{end} nie istnieją!")
            except Exception as e:
                print(f"      BŁĄD KRYTYCZNY: {e}")
            
    return prepared

def run_test_param(param_name, range_values, data_list):
    print(f"\n=== TESTOWANIE PARAMETRU: {param_name} ===")
    results = {'x': [], 'F1': [], 'TP': [], 'FP': []}
    
    for val in range_values:
        curr_win    = val if param_name == 'WINDOW' else BASE_WINDOW
        curr_len_mm = val if param_name == 'LEN_MM' else BASE_LEN_MM
        curr_pct    = val if param_name == 'PCT'    else BASE_PCT
        curr_grad   = val if param_name == 'GRAD'   else BASE_GRAD
        
        global_tp, global_fp, global_fn = 0, 0, 0
        
        for data in data_list:
            regions = detect_local_stenosis_with_grad(
                data['diameters'], data['spacing'],
                window_mm=curr_win, min_stenosis=curr_pct, min_length_mm=curr_len_mm,
                use_gradient=True, grad_thresh=curr_grad, x_positions=data['cum_orig']
            )
            
            detected_pos = [data['cum_orig'][r['max_stenosis_idx']] for r in regions]
            
            # Logika zliczania TP/FP/FN
            local_tp = 0
            local_fp = 0
            true_rem = data['true_stenoses'].copy()
            
            for det in detected_pos:
                match = False
                for true_pos in true_rem:
                    if abs(det - true_pos) <= TOLERANCE_MM:
                        match = True; true_rem.remove(true_pos); break
                if match: local_tp += 1
                else: local_fp += 1
            
            global_tp += local_tp
            global_fp += local_fp
            global_fn += len(true_rem)
            
        prec = global_tp/(global_tp+global_fp) if (global_tp+global_fp) > 0 else 0
        rec = global_tp/(global_tp+global_fn) if (global_tp+global_fn) > 0 else 0
        f1 = 2*(prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
        
        print(f"Val: {val:6.3f} | TP:{global_tp} FP:{global_fp} | F1: {f1:.3f}")
        results['x'].append(val); results['F1'].append(f1); results['TP'].append(global_tp); results['FP'].append(global_fp)
        
    return results

def plot_param_result(res, param_name, filename, unit):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel(f'{param_name} [{unit}]'); ax1.set_ylabel('Liczba detekcji', color='black')
    ax1.plot(res['x'], res['TP'], 'g-o', label='TP'); ax1.plot(res['x'], res['FP'], 'r-x', label='FP')
    ax1.legend(loc='center left'); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx(); ax2.set_ylabel('F1 Score', color='blue')
    ax2.plot(res['x'], res['F1'], 'b--', linewidth=2, label='F1 Score')
    best_idx = np.argmax(res['F1'])
    plt.title(f'Optymalizacja: {param_name}\nMax F1={res["F1"][best_idx]:.2f} dla {res["x"][best_idx]:.2f} {unit}')
    plt.tight_layout(); plt.savefig(filename)
    print(f"Zapisano wykres: {filename}")

if __name__ == "__main__":
    all_data = prepare_data()
    
    if all_data:
        # 1. TEST GRADIENTU
        res_grad = run_test_param('GRAD', np.arange(-0.01, -0.30, -0.01), all_data)
        plot_param_result(res_grad, 'Próg Gradientu', 'final_opt_gradient.png', '-')
        
        # 2. TEST OKNA
        res_win = run_test_param('WINDOW', np.arange(3, 30, 1), all_data)
        plot_param_result(res_win, 'Szerokość Okna', 'final_opt_window.png', 'mm')
        
        # 3. TEST DŁUGOŚCI
        res_len = run_test_param('LEN_MM', np.arange(0.5, 5, 0.1), all_data)
        plot_param_result(res_len, 'Minimalna Długość', 'final_opt_len_mm.png', 'mm')
        
        # 4. TEST PROCENTÓW
        res_pct = run_test_param('PCT', np.arange(5, 30, 1), all_data)
        plot_param_result(res_pct, 'Próg Zwężenia', 'final_opt_pct.png', '%')
    else:
        print("Brak danych do testów. Sprawdź RUN_ONLY i ścieżki.")