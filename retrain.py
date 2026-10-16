"""
retrain.py
----------
Mechanizm douczania modelu na nowych danych z automatyczną decyzją o wdrożeniu.

Rzetelne porównanie modeli:
  T = koniec danych treningowych aktywnego modelu (training_end w model_versions).
  Nowe dane (po T) dzielone są chronologicznie w punkcie S na dwie połowy:

    |---- dane do T ----|-- nowe, cz. 1 --|24h|-- nowe, cz. 2 --|
     stary model                               zbiór testowy
     kandydat ------------------------------>  (dla obu modeli)

  1. Kandydat: model trenowany na danych, których wartość docelowa (t + 24 h)
     poprzedza S - czyli na danych do T oraz na pierwszej połowie nowych danych
     (domyślnie na całej historii, opcjonalnie tylko na ostatnich N dniach).
  2. Zbiór testowy: dane od S do końca - nie widział ich żaden z modeli.
  3. Jeśli kandydat nie jest gorszy, wdrażany jest model trenowany na wszystkich
     dostępnych danych (lub na ostatnich N dniach), łącznie z najnowszymi.
  Wymagane jest co najmniej MIN_NEW_HOURS godzin nowych danych.

Użycie:
  python retrain.py                    # douczanie na pełnej historii (zalecane)
  python retrain.py --window-days 90   # douczanie na ostatnich 90 dniach
"""

import argparse
import logging
import pickle
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import pandas as pd

from train import (
    load_data, build_features, fit_model, score_model, setup_logging, window_label,
    FEATURE_COLS, DB_PATH, MODELS_DIR, MODEL_PARAMS, HORIZON,
)

# ── Konfiguracja ──────────────────────────────────────────────────────────────

LOG_PATH = Path("logs/retrain.log")

MIN_NEW_HOURS  = 168   # min. liczba godzin nowych danych (7 dni): douczanie + test
                       # (zbiór testowy to połowa nowych danych, czyli co najmniej 84 h)
MIN_TRAIN_ROWS = 100   # min. liczba rekordów do wytrenowania kandydata

log = logging.getLogger(__name__)


# ── Wczytanie aktywnego modelu ────────────────────────────────────────────────

def load_active_model(conn: sqlite3.Connection) -> tuple[object | None, dict | None]:
    """
    Wczytuje aktualnie aktywny model z bazy i pliku .pkl.
    Zwraca (model, metadane) lub (None, None), jeśli brak.
    """
    # aktywny model; gdyby aktywnych było kilka - najnowszy
    row = conn.execute("""
        SELECT id, model_path, trained_at, training_end
        FROM model_versions
        WHERE is_active = 1
        ORDER BY trained_at DESC
        LIMIT 1
    """).fetchone()

    if not row:
        log.warning("Brak aktywnego modelu w bazie. Uruchom najpierw: python train.py")
        return None, None

    model_id, model_path, trained_at, training_end = row   # rozpakowanie wiersza wyniku

    path = Path(model_path)
    if not path.exists():
        log.error("Plik modelu nie istnieje: %s", path)
        return None, None

    with open(path, "rb") as f:
        saved = pickle.load(f)                   # słownik: model + lista cech

    log.info("Wczytano aktywny model [id=%d] | trenowany: %s | dane do: %s",
             model_id, trained_at, training_end)

    return saved["model"], {
        "id": model_id,
        "trained_at": trained_at,
        "training_end": pd.to_datetime(training_end),
        # lista cech zapisana razem z modelem - stary model oceniany jest
        # na tych cechach, na których był trenowany
        "feature_cols": saved.get("feature_cols", FEATURE_COLS),
    }


# ── Zapis nowego modelu ───────────────────────────────────────────────────────

def save_new_model(conn: sqlite3.Connection, model, train_df: pd.DataFrame,
                   window_days: int | None, scores: dict, is_better: bool) -> Path:
    """
    Zapisuje nowy model do pliku, MLflow i tabeli model_versions.
    Aktywuje go tylko wtedy, gdy wygrał porównanie (lub wymuszono wdrożenie).
    """
    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"model_retrain_{timestamp}_{window_tag(window_days)}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": FEATURE_COLS,
            "trained_at":   timestamp,
            "window_days":  window_days,
        }, f)

    training_start = str(train_df["observed_at"].min())
    training_end   = str(train_df["observed_at"].max())

    # ── MLflow ──
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("weather_prediction")

    with mlflow.start_run(run_name=f"retrain_{window_tag(window_days)}") as run:
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("run_type",       "retrain")
        mlflow.log_param("window_days",    window_days or "all")
        mlflow.log_param("training_start", training_start)
        mlflow.log_param("training_end",   training_end)
        mlflow.log_param("split_at",       str(scores["split_at"]))
        mlflow.log_param("n_new_train",    scores["n_new_train"])
        mlflow.log_param("n_test",         scores["n_test"])
        mlflow.log_param("deployed",       str(is_better))
        mlflow.log_metric("mae",             scores["new_mae"])
        mlflow.log_metric("rmse",            scores["new_rmse"])
        mlflow.log_metric("mae_old_model",   scores["old_mae"])
        mlflow.log_metric("mae_improvement", scores["old_mae"] - scores["new_mae"])   # > 0: nowy model lepszy
        mlflow.log_artifact(str(model_path))
        run_id = run.info.run_id

    # ── Baza danych ──
    if is_better:
        conn.execute("UPDATE model_versions SET is_active = 0")   # dezaktywacja poprzedniego modelu

    # nowy model trafia do rejestru zawsze, także odrzucony (wtedy is_active = 0)
    conn.execute("""
        INSERT INTO model_versions
            (trained_at, training_start, training_end,
             n_samples, mae, rmse, model_path, is_active, mlflow_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        training_start,
        training_end,
        len(train_df),
        scores["new_mae"],
        scores["new_rmse"],
        str(model_path),
        1 if is_better else 0,                   # aktywny tylko po wygranej
        run_id,
    ))
    conn.commit()

    log.info("Model zapisany: %s | aktywny: %s", model_path, is_better)
    return model_path


# ── Raport ────────────────────────────────────────────────────────────────────

def print_report(scores: dict, window_days: int | None, is_better: bool,
                 T, test_start, test_end) -> None:
    improvement = scores["old_mae"] - scores["new_mae"]
    arrow = "▼" if improvement > 0 else "▲"     # ▼ = błąd spadł
    verdict = "WDROZONY ✓" if is_better else "ODRZUCONY ✗ (stary model dalej aktywny)"

    print("\n" + "="*62)
    print("  WYNIKI DOUCZANIA")
    print("="*62)
    print(f"  Okno treningowe:   {window_label(window_days)}")
    print(f"  Stary model:       dane do {T}")
    print(f"  Kandydat:          dane do {T} + {scores['n_new_train']} godzin nowych danych")
    print(f"  Zbior testowy:     {test_start}  →  {test_end}")
    print(f"                     ({scores['n_test']} godzin, niewidzianych przez oba modele)")
    print("-"*62)
    print(f"  Stary model MAE:   {scores['old_mae']:.4f} °C")
    print(f"  Nowy  model MAE:   {scores['new_mae']:.4f} °C  "
          f"{arrow} {abs(improvement):.4f} °C")
    print("-"*62)
    print(f"  Decyzja:  {verdict}")
    print("="*62)
    print("\n  Historia eksperymentow: http://localhost:5000\n")


def window_tag(window_days: int | None) -> str:
    """Znacznik okna w nazwach plikow i runow MLflow."""
    return f"w{window_days}d" if window_days else "wall"   # np. "w90d" albo "wall" (wszystkie dane)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Douczanie modelu predykcji pogody")
    parser.add_argument(
        "--window-days", type=int, default=None,
        help="Okno czasowe danych do douczania w dniach (domyslnie: wszystkie dane)",
    )
    parser.add_argument(
        "--force-deploy", action="store_true",   # przełącznik bez wartości: podany -> True
        help="Wymus wdrozenie nawet jesli nowy model jest gorszy (do testow)",
    )
    args = parser.parse_args()
    window = timedelta(days=args.window_days) if args.window_days else None   # dni -> timedelta, None = wszystkie dane

    setup_logging(LOG_PATH)
    log.info("START DOUCZANIA | okno: %s", window_label(args.window_days))

    with closing(sqlite3.connect(DB_PATH)) as conn:

        # 1. Wczytaj aktywny model
        old_model, old_meta = load_active_model(conn)
        if old_model is None:
            return
        T = old_meta["training_end"]             # koniec danych aktywnego modelu

        # 2. Wczytaj wszystkie dane i zbuduj cechy
        df = build_features(load_data(conn))

        # 3. Nowe dane (po T) dzielone są w punkcie S na dwie połowy:
        #    pierwsza służy do douczenia kandydata, druga jest zbiorem testowym.
        new_df = df[df["observed_at"] > T]       # dane, których stary model nie widział
        if len(new_df) < MIN_NEW_HOURS:
            log.warning("Za malo nowych danych od %s: %d godzin (minimum %d). "
                        "Douczanie pominiete.", T, len(new_df), MIN_NEW_HOURS)
            return
        S = new_df["observed_at"].iloc[len(new_df) // 2]   # punkt podziału: środek nowych danych

        # Zbiór testowy: od S do końca - nie widział go żaden z modeli.
        test_df = df[df["observed_at"] >= S]
        # Kandydat: wiersze, których wartość docelowa (t + 24 h) poprzedza S,
        # dzięki czemu nie ma wycieku informacji z okresu testowego.
        candidate_df = df[df["observed_at"] + HORIZON < S]
        if window:
            candidate_df = candidate_df[candidate_df["observed_at"] >= S - window]   # ostatnie N dni przed S
        n_new_train = int((candidate_df["observed_at"] > T).sum())   # nowe godziny w danych kandydata

        if len(candidate_df) < MIN_TRAIN_ROWS:
            log.warning("Za malo danych (%s) przed %s: %d rekordow "
                        "(minimum %d). Douczanie pominiete.",
                        window_label(args.window_days), S, len(candidate_df), MIN_TRAIN_ROWS)
            return
        log.info("T=%s | S=%s | kandydat: %d rek. (w tym %d nowych) | test: %d godz.",
                 T, S, len(candidate_df), n_new_train, len(test_df))

        # 4. Porównanie na wspólnym, niewidzianym zbiorze testowym
        candidate = fit_model(candidate_df)      # kandydat douczony na części nowych danych
        old_mae, _        = score_model(old_model, test_df, old_meta["feature_cols"])   # stary model na swoich cechach
        new_mae, new_rmse = score_model(candidate, test_df)   # kandydat na tym samym zbiorze testowym
        scores = {"old_mae": old_mae, "new_mae": new_mae, "new_rmse": new_rmse,
                  "n_test": len(test_df), "n_new_train": n_new_train, "split_at": S}

        log.info("[STARY MODEL] MAE=%.4f °C | [KANDYDAT, %s] MAE=%.4f °C | n_test=%d",
                 old_mae, window_label(args.window_days), new_mae, len(test_df))

        # 5. Decyzja o wdrożeniu: kandydat (douczony na nowych danych) nie może
        #    być gorszy od starego modelu. Przy równym błędzie wdrażany jest model
        #    nowszy, bo zna świeższe obserwacje.
        is_better = new_mae <= old_mae or args.force_deploy
        log.info("Decyzja: %s", "wdrazam nowy model" if is_better else "zachowuje stary model")

        # 6. Model produkcyjny: wszystkie dane (lub ostatnie N dni)
        latest = df["observed_at"].max()         # ostatnia obserwacja w danych
        final_df = df[df["observed_at"] > latest - window] if window else df   # wszystkie dane lub ostatnie N dni
        final_model = fit_model(final_df)        # model, który zostanie zapisany (i ewentualnie wdrożony)

        # 7. Zapis (zawsze - do historii w MLflow i model_versions)
        save_new_model(conn, final_model, final_df, args.window_days, scores, is_better)

        # 8. Raport
        print_report(scores, args.window_days, is_better, T,
                     test_df["observed_at"].min(), test_df["observed_at"].max())


if __name__ == "__main__":
    main()
