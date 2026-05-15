import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# IMPORTY Z PLIKU GŁÓWNEGO
# =============================================================================

from gradient2 import Artery, compute_upsampled_path, adaptive_diameter_calculation, detect_local_stenosis_with_grad


# =============================================================================
# 1. KONFIGURACJA OSTATECZNA (TWOJE ZŁOTE PARAMETRY)
# =============================================================================
FINAL_WINDOW = 11.0
FINAL_LEN    = 3.0
FINAL_PCT    = 24.0
FINAL_GRAD   = -0.19

TOLERANCE_MM = 10.0 # Margines błędu lokalizacji (np. +/- 1 cm)

# =============================================================================
# 2. DANE PACJENTÓW (KOMPLETNE)
# =============================================================================
PATIENTS = {
    "P1 (CAD 0)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_right.nrrd", "start_node": 329, "end_node": 595, "true_stenoses_mm": []},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_left.nrrd", "start_node": 0, "end_node": 328, "true_stenoses_mm": []},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 0\Segmentation_left.nrrd", "start_node": 0, "end_node": 163, "true_stenoses_mm": []}
    ],
    "P2 (CAD 1)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_right.nrrd", "start_node": 398, "end_node": 705, "true_stenoses_mm": []},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_left.nrrd", "start_node": 10, "end_node": 779, "true_stenoses_mm": []},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 1\Segmentation_left.nrrd", "start_node": 0, "end_node": 1014, "true_stenoses_mm": []}
    ],
    "P3 (CAD 2)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_right.nrrd", "start_node": 275, "end_node": 463, "true_stenoses_mm": []},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_left.nrrd", "start_node": 1, "end_node": 239, "true_stenoses_mm": []},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 2\Segmentation_left.nrrd", "start_node": 1, "end_node": 612, "true_stenoses_mm": []}
    ],
    "P4 (CAD 3)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_right.nrrd", "start_node": 323, "end_node": 515, "true_stenoses_mm": []},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_left.nrrd", "start_node": 0, "end_node": 669, "true_stenoses_mm": [55.6]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3\Segmentation_left.nrrd", "start_node": 0, "end_node": 775, "true_stenoses_mm": [90.6]}
    ],
    "P5 (CAD 4)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_right.nrrd", "start_node": 298, "end_node": 462, "true_stenoses_mm": [0.0,117.8,152.4]}, # Okluzja
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_left.nrrd", "start_node": 0, "end_node": 460, "true_stenoses_mm": [10.7]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4\Segmentation_left.nrrd", "start_node": 0, "end_node": 184, "true_stenoses_mm": []}
    ],
    "P6 (CAD 5)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_right.nrrd", "start_node": 325, "end_node": 647, "true_stenoses_mm": [46.9]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_left.nrrd", "start_node": 0, "end_node": 499, "true_stenoses_mm": [45.7, 83.4]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5\Segmentation_left.nrrd", "start_node": 0, "end_node": 265, "true_stenoses_mm": []}
    ],
    "P7 (CAD 3v2)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_right.nrrd", "start_node": 314, "end_node": 610, "true_stenoses_mm": [1.3]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_left.nrrd", "start_node": 0, "end_node": 625, "true_stenoses_mm": [18.6, 56.9]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3v2\Segmentation_left.nrrd", "start_node": 0, "end_node": 990, "true_stenoses_mm": [29.6,39.7]}
    ],
    "P8 (CAD 4v2)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_right.nrrd", "start_node": 183, "end_node": 360, "true_stenoses_mm": [5.0, 30.5, 90.7]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_left.nrrd", "start_node": 0, "end_node": 527, "true_stenoses_mm": [8.4, 41.6, 88.2]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 4v2\Segmentation_left.nrrd", "start_node": 0, "end_node": 357, "true_stenoses_mm": []}
    ],
    "P9 (CAD 5v2)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_right.nrrd", "start_node": 210, "end_node": 409, "true_stenoses_mm": [32.4, 124.7]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_left.nrrd", "start_node": 59, "end_node": 594, "true_stenoses_mm": [98.7]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 5v2\Segmentation_left.nrrd", "start_node": 59, "end_node": 857, "true_stenoses_mm": []}
    ],
    "P10 (CAD 3_4)": [
        {"name": "RCA", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_right.nrrd", "start_node": 599, "end_node": 868, "true_stenoses_mm": [21.6, 36.4, 89.0, 110.7]},
        {"name": "LAD", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_left.nrrd", "start_node": 0, "end_node": 185, "true_stenoses_mm": [48.0, 84.7]},
        {"name": "LCx", "file_path": r"C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\CADRADS 3_4\Segmentation_left.nrrd", "start_node": 0, "end_node": 518, "true_stenoses_mm": []}
    ],
                "P11 (CAD 0v2)": [
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
            "P12 (CAD 0v3)": [
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
                "P13 (CAD 1v2)": [
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
                "P14 (CAD 1v3)": [
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
                "P15 (CAD 2v2)": [
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
                "P16 (CAD 2v3)": [
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
                "P17 (CAD 3v3)": [
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
                "P18 (CAD 3v4)": [
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
                "P19 (CAD 4v3)": [
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
                "P20 (CAD 5v3)": [
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
# 3. SILNIK WALIDACJI
# =============================================================================
def validate_all_patients():
    stats = []
    print("=== ROZPOCZYNAM WALIDACJĘ KOŃCOWĄ ===")
    print(f"Parametry: Win={FINAL_WINDOW}, Len={FINAL_LEN}, Pct={FINAL_PCT}, Grad={FINAL_GRAD}\n")

    for pid, arteries in PATIENTS.items():
        print(f"-> Analiza pacjenta: {pid}")
        
        # Statystyki per pacjent
        p_tp, p_fp, p_fn = 0, 0, 0
        total_ground_truth = 0
        
        for conf in arteries:
            if not os.path.exists(conf['file_path']):
                print(f"   [!] Brak pliku: {conf['name']}")
                continue
                
            try:
                # 1. Obliczenia
                a = Artery("X", conf['file_path'])
                a.load()
                if not (conf['start_node'] in a.graph and conf['end_node'] in a.graph): continue
                
                path_idxs = nx.shortest_path(a.graph, conf['start_node'], conf['end_node'])
                path_pts = a.points[path_idxs]
                diams = [adaptive_diameter_calculation(p, a.dist_map, a.skeleton, a.points, a.spacing) for p in path_pts]
                _, cum_orig, _, _, _ = compute_upsampled_path(path_pts, a.spacing)
                
                # 2. Detekcja
                regions, _ = detect_local_stenosis_with_grad(
                    diams, a.spacing,
                    window_mm=FINAL_WINDOW,
                    min_stenosis=FINAL_PCT,
                    min_length_mm=FINAL_LEN,
                    use_gradient=True,
                    grad_thresh=FINAL_GRAD,
                    x_positions=cum_orig
                )
                
                # 3. Porównanie z Ground Truth
                detected_mm = [cum_orig[r['max_stenosis_idx']] for r in regions]
                true_mm = conf['true_stenoses_mm'].copy()
                total_ground_truth += len(true_mm)
                
                # Matching
                local_tp = 0
                local_fp = 0
                
                for det_pos in detected_mm:
                    match_found = False
                    for t_pos in true_mm:
                        if abs(det_pos - t_pos) <= TOLERANCE_MM:
                            match_found = True
                            true_mm.remove(t_pos) # Usunięcie trafionego
                            break
                    
                    if match_found: local_tp += 1
                    else: local_fp += 1
                
                local_fn = len(true_mm) # To co zostało, to przeoczenia
                
                p_tp += local_tp
                p_fp += local_fp
                p_fn += local_fn
                
            except Exception as e:
                print(f"   [!] Błąd w {conf['name']}: {e}")

        # Obliczenie metryk dla pacjenta
        precision = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0 else 0
        recall = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Obsługa pacjentów zdrowych (dzielenie przez zero)
        if total_ground_truth == 0:
            status = "HEALTHY"
            # Jeśli zdrowy i FP=0 -> F1=1.0 (Sukces)
            # Jeśli zdrowy i FP>0 -> F1=0.0 (Fałszywy alarm)
            f1 = 1.0 if p_fp == 0 else 0.0
        else:
            status = "DISEASED"

        stats.append({
            "Patient": pid,
            "Type": status,
            "GT (Prawda)": total_ground_truth,
            "TP (Trafienia)": p_tp,
            "FP (Fałszywe)": p_fp,
            "FN (Przeoczone)": p_fn,
            "F1-Score": round(f1, 2)
        })

    return pd.DataFrame(stats)

# =============================================================================
# 4. RYSOWANIE I RAPORT
# =============================================================================
if __name__ == "__main__":
    df = validate_all_patients()
    
    print("\n" + "="*80)
    print("SZCZEGÓŁOWY RAPORT WALIDACJI")
    print("="*80)
    print(df.to_string(index=False))
    
    # Obliczenie średnich
    avg_f1 = df["F1-Score"].mean()
    total_tp = df["TP (Trafienia)"].sum()
    total_fp = df["FP (Fałszywe)"].sum()
    total_fn = df["FN (Przeoczone)"].sum()
    
    # Global Precision/Recall
    gl_prec = total_tp / (total_tp + total_fp) if (total_tp+total_fp) > 0 else 0
    gl_rec = total_tp / (total_tp + total_fn) if (total_tp+total_fn) > 0 else 0
    gl_f1 = 2 * (gl_prec * gl_rec) / (gl_prec + gl_rec) if (gl_prec+gl_rec) > 0 else 0

    print("\n" + "-"*80)
    print(f"ŚREDNI F1-SCORE (Macro-Average): {avg_f1:.4f}")
    print(f"GLOBALNY F1-SCORE (Micro-Average): {gl_f1:.4f}")
    print(f"SUMA: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print("="*80)
    
    # --- WYKRES SŁUPKOWY ---
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    
    # Rysujemy Rzeczywiste (jako tło) i Wykryte (TP+FP)
    plt.bar(x - width/2, df["GT (Prawda)"], width, label='Ground Truth (Opis Lekarza)', color='lightgray', edgecolor='black')
    plt.bar(x + width/2, df["TP (Trafienia)"], width, label='Poprawne Detekcje (TP)', color='green')
    # FP rysujemy na górze słupka TP (stacked) lub obok - tu zrobimy obok dla czytelności
    # Ale żeby nie zaciemniać, narysujmy po prostu TP i FP obok siebie
    
    plt.clf() # Czyścimy
    
    # Nowy, lepszy wykres: TP, FP, FN
    width = 0.25
    plt.bar(x - width, df["TP (Trafienia)"], width, label='Trafienia (TP)', color='#2ca02c') # Zielony
    plt.bar(x,        df["FP (Fałszywe)"],  width, label='Fałszywe Alarmy (FP)', color='#d62728') # Czerwony
    plt.bar(x + width, df["FN (Przeoczone)"], width, label='Przeoczone (FN)', color='#ff7f0e') # Pomarańczowy
    
    plt.xlabel('Pacjent')
    plt.ylabel('Liczba Zwężeń')
    plt.title('Skuteczność detekcji dla poszczególnych przypadków testowych')
    plt.xticks(x, df["Patient"], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig("final_validation_chart.png", dpi=300)
    print("\nWykres zapisano jako 'final_validation_chart.png'")