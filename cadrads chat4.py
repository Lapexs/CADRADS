import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian

def visualize_smoothing_justification():
    # 1. Tworzymy "schodkowate" koło (mała rozdzielczość)
    size = 20
    Y, X = np.ogrid[:size, :size]
    center = (10, 10)
    radius = 6.5 # Połówkowa wartość żeby wymusić aliasing
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    binary_mask = dist <= radius
    
    # 2. Wygładzamy
    sigma = 1
    smoothed = gaussian(binary_mask.astype(float), sigma=sigma)
    re_thresholded = smoothed > 0.5

    # 3. Rysowanie
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Oryginał
    axes[0].imshow(binary_mask, cmap='gray', interpolation='nearest')
    axes[0].set_title("A. Surowa Maska Binarna\n(Efekt schodkowania)", fontsize=11)
    # Zaznaczamy "schodki"
    axes[0].annotate('Ostre rogi\n(Aliasing)', xy=(15, 6), xytext=(18, 2),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='red')

    # Gauss
    axes[1].imshow(smoothed, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f"B. Filtr Gaussa (sigma={sigma})\n(Wartości ciągłe)", fontsize=11)

    # Wynik
    axes[2].imshow(re_thresholded, cmap='gray', interpolation='nearest')
    axes[2].set_title("C. Maska po regularyzacji\n(Gładsze krawędzie)", fontsize=11)
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("uzasadnienie_gaussa.png", dpi=150)
    print("Wygenerowano: uzasadnienie_gaussa.png")

if __name__ == "__main__":
    visualize_smoothing_justification()