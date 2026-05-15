# CAD-RADS - analiza zwezen tetnic w danych NRRD

Repozytorium zawiera eksperymentalne skrypty Python do analizy segmentacji tetnic
wiencowych zapisanych w formacie NRRD. Kod buduje szkielet naczynia, wyznacza
sciezki w grafie, szacuje srednice naczynia i wykrywa lokalne zwezenia na
podstawie spadku srednicy oraz gradientu.

Projekt jest w obecnym stanie badawczo-prototypowy: sciezki do danych i listy
pacjentow sa wpisane bezposrednio w skryptach, a dane NRRD nie sa czescia repo.

## Najwazniejsze pliki

| Plik | Rola |
| --- | --- |
| `gradient2.py` | Glowny modul analityczny i aplikacja interaktywna z wizualizacja 3D w `vedo`. Definiuje klase `Artery`, klase `AnalyzerApp` oraz funkcje do obliczania srednic i wykrywania zwezen. |
| `Split.py` | Walidacja finalnego zestawu parametrow na wielu pacjentach. Generuje raport w konsoli i wykres `final_validation_chart.png`. |
| `test_gradientu_multipath.py` | Pelny grid search dla parametrow detekcji (`WINDOW`, `LEN_MM`, `PCT`, `GRAD`). Zapisuje ranking do `wyniki_grid_search.txt` i generuje wykresy stabilnosci. |
| `test_wszystkich_parametrow.py` | Jednowymiarowe testy wplywu poszczegolnych parametrow na F1/TP/FP. Generuje wykresy `final_opt_*.png`. |
| `cadrads_new.py` | Generowanie wykresow porownujacych wplyw parametrow wokol wybranego punktu pracy. |
| `wyniki_grid_search.txt` | Przykladowy zapis wynikow grid search. |

## Dane wejsciowe

Skrypty oczekuja segmentacji tetnic w formacie `.nrrd`, zwykle w parach:

- `Segmentation_left.nrrd`
- `Segmentation_right.nrrd`

W konfiguracjach pacjentow uzywane sa trzy glowne naczynia:

- `RCA`
- `LAD`
- `LCx`

Dla kazdego przypadku konfiguracja zawiera:

- `file_path` - sciezke do pliku NRRD,
- `start_node` i `end_node` - indeksy wezlow szkieletu/grafu wyznaczajace analizowana sciezke,
- `true_stenoses_mm` - pozycje referencyjnych zwezen w milimetrach, uzywane do walidacji.

Uwaga: obecnie w plikach sa lokalne sciezki Windows, np.
`C:\Users\PC\Desktop\INZYNIERKA\Slicer_JM\...`. Przed uruchomieniem trzeba je
zmienic na sciezki dostepne na danym komputerze.

## Wymagania

Repozytorium nie ma jeszcze pliku `requirements.txt`, ale z importow w kodzie
wynikaja nastepujace zaleznosci:

- Python 3
- `numpy`
- `scipy`
- `scikit-image`
- `networkx`
- `matplotlib`
- `pandas`
- `vedo`
- `pynrrd`

Przykladowe przygotowanie srodowiska:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-image networkx matplotlib pandas vedo pynrrd
```

Jesli na danym systemie interpreter jest dostepny jako `python`, mozna uzyc
`python` zamiast `python3`.

## Uruchomienie aplikacji interaktywnej

1. Otworz `gradient2.py`.
2. Na dole pliku ustaw poprawne sciezki:

   ```python
   LEFT_FILE = r".../Segmentation_left.nrrd"
   RIGHT_FILE = r".../Segmentation_right.nrrd"
   ```

3. Uruchom:

   ```bash
   python3 gradient2.py
   ```

4. W oknie 3D wybierz punkty na naczyniach:

   - lewa tetnica: 3 punkty,
   - prawa tetnica: 2 punkty,
   - `r` - reset calej analizy,
   - `l` - reset lewej tetnicy,
   - `p` - reset prawej tetnicy.

Po analizie program wypisuje raport zwezen w konsoli i zapisuje wykresy srednic
do katalogu `gradient_analysis/`.

## Uruchomienie walidacji i testow parametrow

Przed uruchomieniem skryptow walidacyjnych popraw sciezki `file_path` w slowniku
`PATIENTS` oraz upewnij sie, ze indeksy `start_node` i `end_node` odpowiadaja
aktualnym szkieletom naczyn.

### Walidacja finalnych parametrow

```bash
python3 Split.py
```

Wyniki:

- raport TP/FP/FN/F1 w konsoli,
- `final_validation_chart.png`.

### Grid search wielu parametrow

```bash
python3 test_gradientu_multipath.py
```

Wyniki:

- `wyniki_grid_search.txt`,
- `gs_heatmap_win_grad.png`,
- `gs_heatmap_len_pct.png`,
- `gs_parallel_coords.png`.

### Test pojedynczych parametrow

```bash
python3 test_wszystkich_parametrow.py
```

Wyniki:

- `final_opt_gradient.png`,
- `final_opt_window.png`,
- `final_opt_len_mm.png`,
- `final_opt_pct.png`.

### Wykresy do porownania punktu pracy

```bash
python3 cadrads_new.py
```

Wyniki:

- `wykres_window.png`,
- `wykres_length.png`,
- `wykres_pct.png`,
- `wykres_gradient.png`.

## Aktualny punkt pracy algorytmu

W kilku skryptach powtarza sie wybrany zestaw parametrow:

```python
WINDOW = 11.0
LEN_MM = 3.0
PCT = 24.0
GRAD = -0.19
```

Wyniki grid search w `wyniki_grid_search.txt` wskazuja najlepsze kombinacje w
okolicy:

```text
F1 ~= 0.816
WINDOW = 11.0
LEN_MM = 3.3
PCT = 22-23
GRAD = -0.19
```

## Jak dziala analiza

W uproszczeniu pipeline wyglada tak:

1. Wczytanie maski `.nrrd`.
2. Wybor najwiekszej spojnej skladowej segmentacji.
3. Wygenerowanie szkieletu 3D naczynia.
4. Budowa grafu na punktach szkieletu.
5. Wyznaczenie najkrotszej sciezki miedzy wskazanymi wezlami.
6. Obliczenie srednicy lokalnej z mapy odleglosci.
7. Wygladzenie profilu srednicy.
8. Wykrycie lokalnych spadkow srednicy przekraczajacych prog procentowy,
   minimalna dlugosc i prog gradientu.
9. Odfiltrowanie czesci przypadkow przy bifurkacjach i bardzo dystalnych
   fragmentach naczynia.
10. Przypisanie orientacyjnej klasy CAD-RADS na podstawie maksymalnego zwezenia.

## Disclaimer

Projekt sluzy do eksperymentow badawczych i wsparcia analizy obrazowej. Wyniki
nie powinny byc traktowane jako samodzielna diagnoza kliniczna.
