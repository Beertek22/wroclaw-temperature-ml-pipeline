# Douczanie modelu predykcji temperatury (praca magisterska)

Potok ML do prognozowania temperatury powietrza we Wrocławiu z wyprzedzeniem 24 h.
Dane godzinowe pobierane są z Open-Meteo Historical API, zapisywane w SQLite,
a model XGBoost jest trenowany, douczany na nowych danych i porównywany w eksperymentach.
Przebieg uruchomień rejestrowany jest w MLflow.

## Struktura projektu

```
wroclaw-temperature-ml-pipeline/
├── collect_data.py            # Moduł 1: pobieranie danych z Open-Meteo i zapis do bazy
├── train.py                   # Moduł 2: trenowanie modelu bazowego, funkcje wspólne
├── retrain.py                 # Moduł 3: mechanizm douczania i wdrażania modelu
├── evaluate.py                # Moduł 4: eksperymenty badawcze (P1–P3)
├── calculate_statistics.py    # Skrypt pomocniczy: statystyki opisowe danych
├── requirements.txt           # Lista zależności (wersje bibliotek)
├── README.md                  # Opis projektu i instrukcja uruchomienia
├── .gitignore                 # Pliki pomijane w repozytorium
├── data/
│   └── weather.db             # Baza SQLite z danymi wykorzystanymi w pracy
├── results/                   # Wyniki eksperymentów (.csv)
│   ├── experiment_drift.csv                # P1: dryft sezonowy
│   ├── experiment_P2_windows.csv           # P2: podsumowanie wariantów okna
│   ├── experiment_P2_windows_weekly.csv    # P2: wyniki tygodniowe
│   ├── experiment_P2_windows_seasons.csv   # P2: średnie MAE wg pory roku
│   ├── experiment_P3_frequency.csv         # P3: wyniki tygodniowe
│   ├── experiment_P3_summary.csv           # P3: podsumowanie strategii
│   └── experiment_P3_by_year.csv           # P3: średnie MAE wg roku
│
│   Tworzone przez skrypty (poza repozytorium, patrz .gitignore):
├── models/                    # Zapisane modele: model_<data>.pkl, model_retrain_<data>_<okno>.pkl
├── logs/                      # Logi modułów: collect.log, train.log, retrain.log, evaluate.log
├── mlflow.db                  # Baza MLflow (parametry i metryki uruchomień)
└── mlruns/                    # Artefakty MLflow (pliki modeli)
```

Katalogi `data/`, `models/`, `logs/`, `results/` oraz pliki MLflow tworzone są automatycznie
przy pierwszym uruchomieniu skryptów. Repozytorium zawiera kod, dane wykorzystane w pracy
(`data/weather.db`) i wyniki eksperymentów (`results/`). Wytrenowane modele, logi i historia
MLflow nie są wersjonowane – powstają po uruchomieniu skryptów.

## Wymagania

- Python 3.12
- biblioteki z `requirements.txt` (requests, pandas, numpy, scikit-learn, xgboost, mlflow)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Wszystkie polecenia należy uruchamiać z katalogu `wroclaw-temperature-ml-pipeline/`.

## Uruchomienie krok po kroku

### 1. Pobranie danych

#### Dane wykorzystane w pracy

| | |
|---|---|
| Źródło | Open-Meteo Historical Weather API (`archive-api.open-meteo.com/v1/archive`), domyślny tryb Best Match |
| Lokalizacja | Wrocław, 51,11° N, 17,04° E |
| Okres | 2024-10-06 00:00 – 2026-10-06 23:00 (UTC) |
| Liczba rekordów | 17 544 obserwacje godzinowe (731 dni × 24 h), 0 odrzuconych przez walidację |
| Zmienne | `temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m`, `wind_direction_10m`, `pressure_msl`, `cloud_cover` |
| Sposób pobrania | jedno uruchomienie `--mode history` w dniu 6 października 2026 (24 zapytania po ok. 30 dni, 0 nieudanych) |

Dokładny zbiór danych użyty w pracy znajduje się w pliku `data/weather.db`.

#### Odtworzenie zbioru danych z pracy

Tryb `--mode history` pobiera ostatnie 730 dni **licząc od dnia uruchomienia**, więc uruchomiony
w innym dniu pobierze inny okres. Aby pobrać dokładnie ten sam zakres co w pracy, użyj trybu `range`
z nową, pustą bazą:

```bash
# 1. przenieś dotychczasową bazę (jeśli istnieje)
move data\weather.db data\weather_backup.db        # Linux/macOS: mv data/weather.db data/weather_backup.db

# 2. pobierz okres z pracy (daty UTC, oba końce włącznie)
python collect_data.py --mode range --start 2024-10-06 --end 2026-10-06
```

Pusta baza jest potrzebna, bo rekordy już zapisane nie są nadpisywane (z wyjątkiem rekordów
wcześniej odrzuconych przez walidację), więc pobranie do istniejącej bazy nie zastąpi jej zawartości.

Po zakończeniu skrypt wyświetla podsumowanie, które powinno wyglądać tak:

```
  Rekordów (valid): 17,544
  Zakres (UTC):  2024-10-06T00:00  ->  2026-10-06T23:00
```

Zgodność z danymi z pracy można dodatkowo sprawdzić skryptem `calculate_statistics.py`
(punkt 2) – wyniki powinny odpowiadać tabeli statystyk opisowych w rozdziale 2
(np. temperatura: min −11,1 °C, max 37,7 °C, średnia 10,33 °C).

**Uwaga:** trenowanie i eksperymenty wykorzystują wszystkie dane zapisane w bazie. Dopisanie
nowych dni (np. przez `--mode daily`) zmieni zakres danych, a tym samym punkty startowe symulacji
w eksperymentach P2 i P3 – wyniki będą się wtedy różnić od przedstawionych w pracy.

#### Pobieranie danych w działaniu systemu

```bash
python collect_data.py --mode history   # pierwsze uruchomienie: ostatnie 730 dni od dziś
python collect_data.py --mode daily     # uzupełnienie danych od ostatniego poprawnego rekordu
python collect_data.py --mode range --start 2023-01-01 --end 2023-12-31   # dowolny zakres (UTC, włącznie)
```

- `history` – dzieli zakres na ok. 30-dniowe zapytania; błąd jednego zapytania nie przerywa pozostałych.
- `daily` – pobiera dane od dnia ostatniego poprawnego rekordu, zawsze co najmniej 7 dni wstecz
  (zapas na opóźnienia w publikacji danych w archiwum i na dni bez uruchomienia).
- `range` – pobiera wskazany zakres dat, np. starsze dane albo dokładny okres z pracy.

Dane godzinowe (czas UTC) zapisywane są w tabeli `weather_observations`, każdy rekord
przechodzi walidację zakresów wartości (kolumna `is_valid`). Przed wyznaczeniem cech szereg
jest wyrównywany do pełnych godzin, więc ewentualne luki w danych nie przesuwają cech opóźnionych.

### 2. Statystyki opisowe (opcjonalnie)

```bash
python calculate_statistics.py
```

Wyświetla średnie miesięczne temperatury oraz statystyki opisowe zmiennych.

### 3. Trenowanie modelu bazowego

```bash
python train.py                          # wszystkie dostępne dane
python train.py --window-days 365        # tylko ostatnie 365 dni
python train.py --exclude-last-days 30   # pomija ostatnie 30 dni (zostawia dane dla retrain.py)
```

Jakość szacowana jest na chronologicznym podziale 80/20 z 24-godzinną przerwą między
zbiorem treningowym a testowym (brak wycieku informacji), po czym model trenowany jest
ponownie na 100% danych, zapisywany w `models/` i oznaczany jako aktywny w tabeli `model_versions`.

### 4. Douczanie modelu

```bash
python retrain.py                    # douczanie na pełnej historii (domyślnie)
python retrain.py --window-days 90   # douczanie tylko na ostatnich 90 dniach
python retrain.py --force-deploy     # wdrożenie bez względu na wynik (do testów)
```

Działanie:

1. Wczytuje aktywny model i koniec jego danych treningowych `T`.
2. Wymaga co najmniej 168 nowych godzin danych po `T` (inaczej kończy pracę).
3. Nowe dane dzieli w połowie (punkt `S`). Kandydat trenowany jest na danych do `T` oraz na
   pierwszej połowie nowych danych (domyślnie na całej historii), a druga połowa jest zbiorem
   testowym. Między treningiem a testem zachowana jest 24-godzinna przerwa
   (zapobiega przeciekowi zmiennej docelowej).
4. Porównuje MAE starego modelu i kandydata na tym samym zbiorze testowym.
5. Jeśli kandydat jest nie gorszy, trenuje go ponownie na wszystkich danych
   (lub na oknie `--window-days`) i wdraża jako nowy aktywny model. Model zapisywany jest
   zawsze (także odrzucony, jako nieaktywny), więc historia prób jest widoczna w MLflow
   i tabeli `model_versions`.

`retrain.py` wymaga aktywnego modelu z pliku `.pkl` w katalogu `models/`. Modele nie są
przechowywane w repozytorium, dlatego po jego sklonowaniu należy najpierw uruchomić `train.py`.

### 5. Eksperymenty badawcze

```bash
python evaluate.py --experiment drift       # P1: dryft sezonowy (trening na jednej porze roku, test na innej)
python evaluate.py --experiment windows     # P2: okno danych (7–180 dni, okno sezonowe, pełna historia)
python evaluate.py --experiment frequency   # P3: częstotliwość douczania, scenariusze „pełna historia” i „zimny start”
```

Eksperymenty P2 i P3 symulują działanie systemu tydzień po tygodniu: w każdym tygodniu
model prognozuje 168 godzin, a douczanie oznacza wytrenowanie nowego modelu od zera na
wybranym zbiorze danych. Każdy wynik porównywany jest z prognozą persystencji
(„temperatura za 24 h = temperatura teraz”); kolumna `skill` to 1 − MAE_model / MAE_persystencji.

Wyniki zapisywane są do `results/*.csv` (podsumowania, wyniki tygodniowe, wyniki wg pory roku)
oraz rejestrowane w MLflow.

### 6. Podgląd wyników w MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

Interfejs MLflow jest dostępny lokalnie pod adresem <http://127.0.0.1:5000>. Na Windowsie
parametr `--workers 1` jest wymagany (bez niego serwer kończy się błędem WinError 10022).
Plik `mlflow.db` nie jest przechowywany w repozytorium, więc po sklonowaniu interfejs jest pusty
do czasu pierwszego uruchomienia `train.py`, `retrain.py` lub `evaluate.py`.


## Automatyzacja (Windows)

Codzienne pobieranie danych można zaplanować w Harmonogramie zadań:

1. Utwórz zadanie podstawowe z wyzwalaczem codziennym (np. 8:00).
2. Akcja: uruchom `C:\ścieżka\do\wroclaw-temperature-ml-pipeline\.venv\Scripts\python.exe`
   z argumentem `collect_data.py --mode daily`, katalog startowy: `C:\ścieżka\do\wroclaw-temperature-ml-pipeline`.

Analogicznie można zaplanować `retrain.py`. Zgodnie z wnioskami z eksperymentu P3 w pierwszym
roku działania systemu douczanie warto uruchamiać co tydzień, a gdy model zna już pełny cykl
roczny, wystarczy co 28 dni.

## Baza danych

Plik `data/weather.db` (SQLite) można przeglądać np. w [DB Browser for SQLite](https://sqlitebrowser.org/).

| Tabela | Zawartość |
|---|---|
| `weather_observations` | godzinowe obserwacje meteorologiczne (UTC) z wynikiem walidacji |
| `model_versions` | historia modeli: ścieżka pliku, zakres danych, metryki, flaga aktywnego modelu |
| `pipeline_logs` | zapis każdego uruchomienia modułów potoku |

## Powtarzalność

Model XGBoost używa `random_state=42`, a podziały danych są chronologiczne, więc wyniki
na tych samych danych i wersjach bibliotek są powtarzalne. Różne platformy lub wersje XGBoost
mogą dawać minimalnie inne wartości metryk.
