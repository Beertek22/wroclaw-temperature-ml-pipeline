"""
evaluate.py
-----------
Eksperymenty badawcze.

Numeracja zgodna z pracą:
  Eksperyment 1 (P1): dryft sezonowy                 --experiment drift
  Eksperyment 2 (P2): wpływ okna danych              --experiment windows
  Eksperyment 3 (P3): wpływ częstotliwości douczania --experiment frequency

Eksperymenty P2 i P3 korzystają z tej samej symulacji tydzień po tygodniu.
Douczanie oznacza wytrenowanie nowego modelu od zera na wybranym zbiorze danych.
Każdy wynik porównywany jest z prognozą persystencji (temperatura za 24 h =
temperatura teraz), a skill score = 1 - MAE_model / MAE_persystencji.

Cechy wyznaczane są raz na całym, ciągłym szeregu godzinowym, a następnie
dzielone na zbiory. Wiersz z chwili t ma wartość docelową z chwili t+24h,
dlatego do treningu trafiają tylko wiersze, których wartość docelowa jest
wcześniejsza niż początek okresu testowego (brak wycieku informacji).

Użycie:
  python evaluate.py --experiment drift
  python evaluate.py --experiment windows
  python evaluate.py --experiment frequency
"""

import argparse
import logging
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from train import (
    load_data, build_features, fit_model, score_model,
    setup_logging, window_label, DB_PATH, HORIZON,
)

# ── Konfiguracja ──────────────────────────────────────────────────────────────

LOG_PATH    = Path("logs/evaluate.log")
RESULTS_DIR = Path("results")

log = logging.getLogger(__name__)


# ── Funkcje pomocnicze ────────────────────────────────────────────────────────

def load_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Wczytuje wszystkie poprawne dane i buduje cechy na całym szeregu."""
    return build_features(load_data(conn))


def train_rows(feat: pd.DataFrame, before) -> pd.Series:
    """
    Maska wierszy, których WARTOŚĆ DOCELOWA (t + 24h) jest wcześniejsza
    niż `before`. Tylko takie wiersze mogą trafić do treningu, jeśli
    od chwili `before` zaczyna się okres oceniany.
    """
    # Wiersz z chwili t ma target z chwili t + 24 h. Jeśli ten target wypada
    # już w okresie ocenianym (>= before), model uczyłby się na wartościach,
    # które potem prognozuje, dlatego takie wiersze są wykluczane.
    # Wynik to maska True/False na każdym wierszu.
    return feat["observed_at"] + HORIZON < before


def persistence_scores(df: pd.DataFrame) -> tuple[float, float]:
    err = df["target"] - df["temperature_2m"]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# Symulacja tydzień po tygodniu (wspólna dla eksperymentów P2 i P3)
# ══════════════════════════════════════════════════════════════════════════════
#
# Okres symulacji dzielony jest na pełne tygodnie. W każdym tygodniu aktualny
# model prognozuje wszystkie 168 godzin i liczone jest MAE tego tygodnia.
# Trenowanie modelu od zera na wybranym zbiorze danych

WEEK = timedelta(days=7)
SEASONAL_HALF_WIDTH_DAYS = 45   # okno sezonowe: ten sam okres rok wcześniej ±45 dni
SEASONAL_RECENT_DAYS     = 30   # ostatnie 30 dni


def make_weeks(feat: pd.DataFrame, start) -> list:
    """Początki pełnych tygodni od `start` do końca danych."""
    end = feat["observed_at"].max()       # początek ostatniej godziny w danych
    weeks, t = [], start                  # t = początek sprawdzanego tygodnia
    # Tydzień to wiersze t, t+1h, ..., t+167h. Jest pełny, jeśli jego ostatni
    # wiersz (t + 167h) istnieje, czyli t + 167h <= end. Po dodaniu 1h do obu
    # stron: t + WEEK <= end + 1h. Niepełny tydzień na końcu danych jest
    # pomijany, żeby każde tygodniowe MAE liczone było z 168 godzin.
    while t + WEEK <= end + timedelta(hours=1):
        weeks.append(t)                   # zapisanie początku tygodnia
        t += WEEK
    return weeks


def select_train(feat: pd.DataFrame, before, window) -> pd.DataFrame:
    """
    Dane do treningu modelu, który ma prognozować od chwili `before`.
    Zawsze tylko wiersze, których wartość docelowa (t + 24 h) jest znana przed
    `before` (24-godzinna przerwa, brak wycieku informacji).
      window = None        -> pełna historia,
      window = N (int)     -> ostatnie N dni,
      window = "sezonowe"  -> ten sam okres rok wcześniej (±45 dni)
                              oraz ostatnie 30 dni.
    """
    # Krok 1 (wspólny dla wszystkich okien): tylko wiersze, których target
    # był znany przed `before`. Ostatni wiersz to before - 25 h.
    known = feat[train_rows(feat, before)]
    if window is None:
        return known                      # pełna historia
    t = known["observed_at"]
    if window == "sezonowe":
        # Dwa rozłączne kawałki połączone operatorem or
        #   1) ten sam okres rok wcześniej: [before - 365 d - 45 d, before - 365 d + 45 d),
        #   2) ostatnie 30 dni przed `before`.
        # Np. dla before = 2026-03-01: styczeń-kwiecień 2025 + luty 2026.
        center = before - timedelta(days=365)
        half = timedelta(days=SEASONAL_HALF_WIDTH_DAYS)
        mask = ((t >= center - half) & (t < center + half)) | \
               (t >= before - timedelta(days=SEASONAL_RECENT_DAYS))
        return known[mask]
    # Okno N dni: ostatnie N dni przed `before`. Po odjęciu 24-godzinnej
    # przerwy zostaje N * 24 - 24 wierszy (np. 30 dni -> 696).
    return known[t >= before - timedelta(days=window)]


def season_name(month: int) -> str:
    # .get(klucz, domyślna): miesiące spoza słownika (9, 10, 11) dają "jesień"
    return {12: "zima", 1: "zima", 2: "zima", 3: "wiosna", 4: "wiosna", 5: "wiosna",
            6: "lato", 7: "lato", 8: "lato"}.get(month, "jesień")


def week_rows(feat: pd.DataFrame, week_start) -> pd.DataFrame:
    # Przedział [week_start, week_start + 7 dni): początek włącznie, koniec
    # bez - dzięki temu kolejne tygodnie stykają się i nie nakładają (168 h).
    return feat[(feat["observed_at"] >= week_start) & (feat["observed_at"] < week_start + WEEK)]


def week_result(i: int, week_start, week: pd.DataFrame, mae: float, **extra) -> dict:
    """Jeden wiersz wyników symulacji: wynik modelu w jednym tygodniu."""
    # **extra zbiera dodatkowe argumenty nazwane w słownik, np.
    # wariant="30 dni", n_train=696 (P2) albo scenario=..., strategy=... (P3),
    # a {**extra, ...} wkleja je do wyniku. Dzięki temu ta sama funkcja
    # obsługuje oba eksperymenty. Pora roku liczona jest wg początku tygodnia.
    return {**extra, "week": i, "date": str(week_start.date()),
            "season": season_name(week_start.month), "mae": mae, "n": len(week)}


def summarize(rows: pd.DataFrame, key: str, order: list) -> pd.DataFrame:
    """Średnie, odchylenie i maksimum tygodniowego MAE oraz skill score
    względem persystencji: 1 - MAE_model / MAE_persystencji."""
    # groupby + agg: jeden wiersz na wariant, trzy statystyki z tygodniowych MAE.
    # reindex(order): kolejność wierszy wg listy zamiast alfabetycznej.
    s = rows.groupby(key)["mae"].agg(["mean", "std", "max"]).reindex(order)
    pers = rows[rows[key] == "persystencja"]["mae"].mean()
    # Skill > 0: model lepszy od persystencji (np. 0.09 = błąd mniejszy o 9%),
    # skill < 0: gorszy. Liczone dla wszystkich wierszy naraz.
    s["skill"] = 1 - s["mean"] / pers
    return s


def print_summary(title: str, s: pd.DataFrame, extra: pd.DataFrame | None = None) -> None:
    print("\n" + "=" * 84)
    print(f"  {title}")
    print("=" * 84)
    print(f"  {'Wariant':<30} {'MAE_śr':>8} {'MAE_std':>8} {'MAE_max':>8} {'Skill':>8}")
    print("-" * 84)
    # Najlepszy wariant szukany jest wśród modeli - wiersz persystencji jest
    # pomijany (errors="ignore": brak takiego wiersza nie powoduje błędu).
    best = s.drop(index="persystencja", errors="ignore")["mean"].min()
    for name, r in s.iterrows():
        mark = " <- NAJLEPSZY" if r["mean"] == best else ""
        # :<30 / :>8.4f - szerokość pola i wyrównanie; :>7.1% - ułamek jako procent
        print(f"  {name:<30} {r['mean']:>8.4f} {r['std']:>8.4f} {r['max']:>8.4f} "
              f"{r['skill']:>7.1%}{mark}")
    if extra is not None:
        print("-" * 84)
        print("  Średnie MAE wg pory roku (początek tygodnia):")
        print(extra.round(4).to_string())
    print("=" * 84 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# EKSPERYMENT 2 (P2): Wpływ okna danych treningowych
# ══════════════════════════════════════════════════════════════════════════════

WINDOWS = [7, 14, 30, 60, 90, 180, "sezonowe", None]   # None = wszystkie dane
INITIAL_SHARE = 0.7      # początek symulacji: 70% szeregu (1 marca 2026)


def experiment_windows(conn: sqlite3.Connection) -> None:
    """
    Douczanie co 7 dni (stała częstotliwość), różne okna danych.
    Przed każdym tygodniem symulacji dla każdego okna trenowany jest nowy
    model na danych z tego okna i prognozuje cały tydzień. Wynik: średnie
    tygodniowe MAE z całego okresu (te same tygodnie co w eksperymencie P3).
    """
    log.info("EKSPERYMENT 2 (P2): Wpływ okna danych treningowych")
    feat = load_features(conn)

    # Początek symulacji: data wiersza na pozycji 70% (.iloc = pozycja, nie etykieta).
    # Wszystko wcześniej to historia znana od początku symulacji.
    start = feat["observed_at"].iloc[int(len(feat) * INITIAL_SHARE)]
    weeks = make_weeks(feat, start)       # lista początków tygodni

    # weeks[-1] to początek ostatniego tygodnia, + WEEK daje koniec symulacji
    log.info("Symulacja: %d tygodni (%s -> %s), douczanie co 7 dni",
             len(weeks), weeks[0].date(), (weeks[-1] + WEEK).date())

    mlflow.set_experiment("experiment_P2_windows")
    rows = []          # wspólna lista wyników: persystencja + wszystkie okna
    # Prognoza odniesienia w każdym tygodniu (bez modelu, więc n_train = 0).
    # i = numer tygodnia, ws = początek tygodnia; [0] wybiera MAE z (MAE, RMSE).
    for i, ws in enumerate(weeks):
        week = week_rows(feat, ws)
        rows.append(week_result(i, ws, week, persistence_scores(week)[0],
                                wariant="persystencja", n_train=0))
    for window in WINDOWS:
        label = window_label(window)
        for i, ws in enumerate(weeks):
            train_df = select_train(feat, ws, window)
            week = week_rows(feat, ws)
            mae, _ = score_model(fit_model(train_df), week)
            rows.append(week_result(i, ws, week, mae, wariant=label, n_train=len(train_df)))
        # Z całej listy wyników tylko MAE bieżącego okna (31 wartości)
        maes = [r["mae"] for r in rows if r["wariant"] == label]
        log.info("[%s] średnie MAE=%.4f", label, np.mean(maes))
        with mlflow.start_run(run_name=f"window_{label}"):
            mlflow.log_param("window_days", window or "all")   # None -> "all"
            mlflow.log_param("retrain_every_days", 7)
            mlflow.log_metric("avg_mae_weekly", float(np.mean(maes)))

    df_r = pd.DataFrame(rows)     # lista słowników -> tabela (279 wierszy x 7 kolumn)
    df_r.to_csv(RESULTS_DIR / "experiment_P2_windows_weekly.csv", index=False)
    # Kolejność wierszy w tabelach: persystencja, potem okna od najkrótszego
    order = ["persystencja"] + [window_label(w) for w in WINDOWS]
    s = summarize(df_r, "wariant", order)
    s["n_train_śr"] = df_r.groupby("wariant")["n_train"].mean().reindex(order).round(0)
    s.to_csv(RESULTS_DIR / "experiment_P2_windows.csv")
    # Tabela przestawna: wiersze = warianty, kolumny = pory roku, komórka = średnie
    # MAE z tygodni danego wariantu w danej porze roku. Kolumny ustawione
    # chronologicznie; zimy brak, bo symulacja trwa od marca do października.
    by_season = (df_r.pivot_table(index="wariant", columns="season", values="mae", aggfunc="mean")
                 .reindex(order)[["wiosna", "lato", "jesień"]])
    by_season.to_csv(RESULTS_DIR / "experiment_P2_windows_seasons.csv")
    print_summary(f"EKSPERYMENT 2 (P2): okno danych, douczanie co 7 dni "
                  f"({len(weeks)} tygodni, {weeks[0].date()} -> {(weeks[-1] + WEEK).date()})",
                  s, by_season)
    print(f"  Wyniki: {RESULTS_DIR / 'experiment_P2_windows.csv'} (+ _weekly, _seasons)\n")


# ══════════════════════════════════════════════════════════════════════════════
# EKSPERYMENT 3 (P3): Wpływ częstotliwości douczania
# ══════════════════════════════════════════════════════════════════════════════

RETRAIN_WINDOW_DAYS = 60

# Dwa scenariusze startowe:
#   pełna historia - model początkowy zna ok. 17 miesięcy (wszystkie pory roku),
#   zimny start    - model początkowy zna tylko ok. 5 miesięcy (jesień i zimę),
#                    jak przy wdrożeniu systemu w nowej lokalizacji.
SCENARIOS = [
    ("pełna historia", INITIAL_SHARE),
    ("zimny start",    0.2),
]

# (etykieta, co ile tygodni douczać, okno douczania w dniach, None = wszystkie dane)
# Decyzja zapada raz na tydzień, dlatego częstotliwości są wielokrotnościami 7 dni.
STRATEGIES = [
    ("model statyczny",              None, None),
    ("co 7 dni, okno 60 dni",        1, RETRAIN_WINDOW_DAYS),
    ("co 14 dni, okno 60 dni",       2, RETRAIN_WINDOW_DAYS),
    ("co 28 dni, okno 60 dni",       4, RETRAIN_WINDOW_DAYS),
    ("co 7 dni, wszystkie dane",     1, None),
    ("co 14 dni, wszystkie dane",    2, None),
    ("co 28 dni, wszystkie dane",    4, None),
]


def simulate_strategy(feat: pd.DataFrame, weeks: list, model,
                      every_weeks: int | None, window_days: int | None) -> tuple[list, int]:
    """
    Jedna strategia douczania: model prognozuje kolejne tygodnie, a co
    `every_weeks` tygodni jest zastępowany nowym modelem wytrenowanym od zera
    na danych znanych na koniec tygodnia (every_weeks = None -> model statyczny).
    Zwraca (lista tygodniowych MAE, liczba douczań).
    """
    maes, n_retrains = [], 0
    for i, ws in enumerate(weeks):
        # Najpierw aktualny model prognozuje tydzień
        maes.append(score_model(model, week_rows(feat, ws))[0])
        # douczanie po tygodniu nr (every_weeks-1), (2*every_weeks-1), ...
        # (bez douczania po ostatnim tygodniu - nowy model nie byłby już oceniany)
        # jeśli wypada termin, model jest zastępowany nowym.
        #   every_weeks = None (model statyczny) -> warunek zawsze fałszywy,
        #   (i + 1) % every_weeks == 0 -> co every_weeks tygodni,
        #   i + 1 < len(weeks)          -> bez douczania po ostatnim tygodniu.
        if every_weeks and (i + 1) % every_weeks == 0 and i + 1 < len(weeks):
            # Dane znane na KONIEC bieżącego tygodnia (= początek następnego)
            known = select_train(feat, ws + WEEK, window_days)
            if len(known) < 200:
                log.warning("Tydzień %d: za mało danych (%d) - bez douczania", i, len(known))
                continue
            model = fit_model(known)
            n_retrains += 1
    return maes, n_retrains


def experiment_frequency(conn: sqlite3.Connection) -> None:
    """
    Symulacja działania produkcyjnego tydzień po tygodniu: model prognozuje
    cały tydzień, po czym - jeśli wypada termin - jest douczany (trenowany od
    nowa) na danych znanych na koniec tego tygodnia. Dwa warianty okna
    oddzielają wpływ CZĘSTOTLIWOŚCI od wpływu ROZMIARU OKNA, a dwa scenariusze
    startowe pokazują, kiedy douczanie przynosi korzyść.
    """
    log.info("EKSPERYMENT 3 (P3): Wpływ częstotliwości douczania")
    feat = load_features(conn)
    mlflow.set_experiment("experiment_P3_frequency")
    rows = []
    summaries = []

    for scenario, share in SCENARIOS:
        start = feat["observed_at"].iloc[int(len(feat) * share)]
        weeks = make_weeks(feat, start)
        # Jeden model początkowy (pełna historia przed startem) dla wszystkich
        # strategii - różnice wyników wynikają więc wyłącznie z douczania.
        initial_model = fit_model(select_train(feat, start, None))   # wspólny start
        log.info("[%s] model początkowy: dane do %s | symulacja: %d tygodni (%s -> %s)",
                 scenario, start, len(weeks), weeks[0].date(), (weeks[-1] + WEEK).date())

        for i, ws in enumerate(weeks):
            week = week_rows(feat, ws)
            rows.append(week_result(i, ws, week, persistence_scores(week)[0],
                                    scenario=scenario, strategy="persystencja"))

        for label, every_weeks, window_days in STRATEGIES:
            maes, n_retrains = simulate_strategy(feat, weeks, initial_model,
                                                 every_weeks, window_days)
            # zip łączy w pary (początek tygodnia, MAE), enumerate dodaje numer;
            # lista składana tworzy wiersz wyników dla każdego tygodnia
            rows += [week_result(i, ws, week_rows(feat, ws), mae,
                                 scenario=scenario, strategy=label)
                     for i, (ws, mae) in enumerate(zip(weeks, maes))]
            log.info("[%s | %s] średnie MAE=%.4f | douczań: %d",
                     scenario, label, np.mean(maes), n_retrains)
            with mlflow.start_run(run_name=f"freq_{scenario}_{label}"):
                mlflow.log_param("scenario", scenario)
                mlflow.log_param("strategy", label)
                mlflow.log_param("retrain_every_days", every_weeks * 7 if every_weeks else "never")
                mlflow.log_param("retrain_window_days", window_days or "all")
                mlflow.log_param("n_retrains", n_retrains)
                mlflow.log_metric("avg_mae_weekly", float(np.mean(maes)))
        summaries.append((scenario, weeks))

    df_r = pd.DataFrame(rows)
    df_r.to_csv(RESULTS_DIR / "experiment_P3_frequency.csv", index=False)
    order = ["persystencja"] + [s[0] for s in STRATEGIES]
    all_sum = []
    for scenario, weeks in summaries:
        part = df_r[df_r["scenario"] == scenario]
        s = summarize(part, "strategy", order)
        s.insert(0, "scenario", scenario)
        all_sum.append(s)
        by_season = (part.pivot_table(index="strategy", columns="season", values="mae",
                                      aggfunc="mean").reindex(order))
        by_season = by_season[[c for c in ["wiosna", "lato", "jesień", "zima"] if c in by_season]]
        print_summary(f"EKSPERYMENT 3 (P3) - scenariusz: {scenario} ({len(weeks)} tygodni, "
                      f"{weeks[0].date()} -> {(weeks[-1] + WEEK).date()})",
                      s.drop(columns="scenario"), by_season)
    pd.concat(all_sum).to_csv(RESULTS_DIR / "experiment_P3_summary.csv")

    # Średnie tygodniowe MAE wg roku rozpoczęcia tygodnia
    # (w zimnym starcie: pierwszy rok, gdy model poznaje pory roku, i drugi rok)
    df_r["year"] = df_r["date"].str[:4]          # "2026-03-01" -> "2026"
    by_year = (df_r.pivot_table(index=["scenario", "strategy"], columns="year",
                                values="mae", aggfunc="mean")
               .reindex(order, level="strategy"))
    by_year.to_csv(RESULTS_DIR / "experiment_P3_by_year.csv")
    print("  Średnie tygodniowe MAE wg roku rozpoczęcia tygodnia:")
    print(by_year.round(4).to_string())
    print(f"\n  Wyniki tygodniowe: {RESULTS_DIR / 'experiment_P3_frequency.csv'}, "
          f"podsumowanie: experiment_P3_summary.csv, experiment_P3_by_year.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# EKSPERYMENT 1 (P1): Dryft sezonowy
# ══════════════════════════════════════════════════════════════════════════════

SUMMER = [6, 7, 8]
WINTER = [12, 1, 2]
MIN_SEASON_HOURS = 85 * 24   # pełny sezon ma 90-92 dni


def season_year(observed_at: pd.Series) -> pd.Series:
    """Rok sezonu: grudzień należy do zimy następnego roku (zima 2024/25 -> 2025)."""
    # (month == 12) daje True/False, astype(int) zamienia na 1/0: grudzień
    # dostaje rok + 1, pozostałe miesiące bez zmian. Cała zima (gru-lut)
    # ma dzięki temu jeden numer roku, mimo że przechodzi przez Nowy Rok.
    return observed_at.dt.year + (observed_at.dt.month == 12).astype(int)


def winter_label(year: int) -> str:
    """2026 -> '2025/26'"""
    return f"{year - 1}/{str(year)[2:]}"        


def complete_seasons(all_df: pd.DataFrame, months: list[int]) -> list[int]:
    """Lata sezonów, dla których baza zawiera (prawie) pełne dane."""
    in_season = all_df["month"].isin(months)    # maska: czy miesiąc należy do sezonu
    # Rok sezonu dla każdej godziny sezonu, value_counts() liczy godziny
    # w każdym roku, np. {2025: 2160, 2026: 2160}.
    counts = season_year(all_df.loc[in_season, "observed_at"]).value_counts()
    # Zwracane są same lata (y); liczba godzin (n) służy tylko do odfiltrowania
    # niepełnych sezonów na brzegach danych. Wynik np. [2025, 2026].
    return sorted(int(y) for y, n in counts.items() if n >= MIN_SEASON_HOURS)


def season_data(all_df: pd.DataFrame, months: list[int], year: int) -> pd.DataFrame:
    """Dane jednego sezonu z jednego roku (ciągły fragment szeregu)."""
    mask = all_df["month"].isin(months) & (season_year(all_df["observed_at"]) == year)
    return all_df[mask].copy()


def experiment_drift(conn: sqlite3.Connection) -> None:
    """
    Model trenowany na jednym sezonie, oceniany na tym samym sezonie
    kolejnego roku (kontrola) i na sezonie przeciwnym (dryft).
    Każdy sezon przetwarzany jest oddzielnie jako ciągły fragment szeregu.
    """
    log.info("EKSPERYMENT 1 (P1): Dryft sezonowy")
    all_df = load_data(conn)
    all_df["observed_at"] = pd.to_datetime(all_df["observed_at"])
    all_df["month"] = all_df["observed_at"].dt.month

    summers = complete_seasons(all_df, SUMMER)
    winters = complete_seasons(all_df, WINTER)
    if len(summers) < 2 or len(winters) < 2:
        log.error("Potrzebne są po 2 pełne sezony letnie i zimowe "
                  "(lato: %s, zima: %s)", summers, winters)
        return

    # Dwa ostatnie pełne sezony: wcześniejszy do treningu, późniejszy do testu
    # (indeksy ujemne liczą od końca listy: -1 ostatni, -2 przedostatni).
    s_train, s_test = summers[-2], summers[-1]
    w_train, w_test = winters[-2], winters[-1]
    # Cechy liczone osobno dla każdego sezonu (ciągły fragment szeregu),
    # więc model nie widzi nawet przez lagi żadnej wartości spoza sezonu.
    seasons = {
        "summer_train": build_features(season_data(all_df, SUMMER, s_train)),
        "summer_test":  build_features(season_data(all_df, SUMMER, s_test)),
        "winter_train": build_features(season_data(all_df, WINTER, w_train)),
        "winter_test":  build_features(season_data(all_df, WINTER, w_test)),
    }
    # Słownik składany: dwa modele (letni i zimowy), każdy użyty w dwóch scenariuszach
    models = {k: fit_model(seasons[k]) for k in ("summer_train", "winter_train")}

    # Układ 2 x 2: (etykieta, model, zbiór testowy, typ). Każdy model i każdy
    # zbiór testowy występuje raz w scenariuszu kontrolnym i raz w dryfcie.
    scenarios = [
        (f"Lato {s_train} -> lato {s_test}", "summer_train", "summer_test", "kontrolny"),
        (f"Zima {winter_label(w_train)} -> zima {winter_label(w_test)}", "winter_train", "winter_test", "kontrolny"),
        (f"Lato {s_train} -> zima {winter_label(w_test)}", "summer_train", "winter_test", "dryft"),
        (f"Zima {winter_label(w_train)} -> lato {s_test}", "winter_train", "summer_test", "dryft"),
    ]

    mlflow.set_experiment("experiment_drift")
    results = []
    for label, tr, te, typ in scenarios:
        # Nazwy z krotki -> obiekty: models["summer_train"], seasons["winter_test"] itd.
        mae, rmse = score_model(models[tr], seasons[te])
        log.info("[%s] MAE=%.4f | RMSE=%.4f", label, mae, rmse)
        results.append({"scenariusz": label, "typ": typ, "mae": mae, "rmse": rmse,
                        "n_train": len(seasons[tr]), "n_test": len(seasons[te])})
        with mlflow.start_run(run_name=label):
            mlflow.log_param("scenario", label)
            mlflow.log_param("type", typ)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)

    # Prognoza odniesienia: persystencja (temperatura za 24 h = temperatura teraz)
    for te, label in (("summer_test", f"Persystencja - lato {s_test}"),
                      ("winter_test", f"Persystencja - zima {winter_label(w_test)}")):
        mae, rmse = persistence_scores(seasons[te])
        results.append({"scenariusz": label, "typ": "odniesienie", "mae": mae, "rmse": rmse,
                        "n_train": 0, "n_test": len(seasons[te])})

    df_r = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "experiment_drift.csv"
    df_r.to_csv(csv_path, index=False)

    # Średnie z par scenariuszy (lato + zima w każdej grupie), więc wynik nie
    # zależy od tego, która pora roku jest łatwiejsza do przewidzenia.
    # Wiersze persystencji nie wchodzą do żadnej średniej.
    control = df_r[df_r["typ"] == "kontrolny"]["mae"].mean()
    drift   = df_r[df_r["typ"] == "dryft"]["mae"].mean()

    print("\n" + "=" * 72)
    print("  EKSPERYMENT 1 (P1): Dryft sezonowy")
    print("=" * 72)
    print(f"  {'Scenariusz':<36} {'Typ':<12} {'MAE':>8} {'RMSE':>8}")
    print("-" * 72)
    for _, row in df_r.iterrows():
        print(f"  {row['scenariusz']:<36} {row['typ']:<12} "
              f"{row['mae']:>8.4f} {row['rmse']:>8.4f}")
    print("-" * 72)
    print(f"  Średnie MAE (kontrolne): {control:.4f} °C")
    print(f"  Średnie MAE (dryft):     {drift:.4f} °C")
    print(f"  Wzrost błędu:            {drift / control:.1f}x "
          f"(+{(drift / control - 1) * 100:.1f}%)")
    print(f"  Wyniki zapisane: {csv_path}")
    print("=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "drift":     experiment_drift,
    "windows":   experiment_windows,
    "frequency": experiment_frequency,
}


def main():
    parser = argparse.ArgumentParser(description="Eksperymenty badawcze - ewaluacja modeli")
    parser.add_argument(
        "--experiment", choices=list(EXPERIMENTS), required=True,
        help=("drift = Eksperyment 1 (P1) | windows = Eksperyment 2 (P2) | "
              "frequency = Eksperyment 3 (P3)"),
    )
    args = parser.parse_args()

    setup_logging(LOG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with closing(sqlite3.connect(DB_PATH)) as conn:
        # Pobranie funkcji ze słownika po nazwie i jej wywołanie z połączeniem
        EXPERIMENTS[args.experiment](conn)


if __name__ == "__main__":
    main()
