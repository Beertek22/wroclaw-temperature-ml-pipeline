"""
train.py
--------
Trenuje model predykcji temperatury za 24h na danych z bazy SQLite.
Zapisuje metryki do MLflow i do tabeli model_versions w bazie.

Procedura:
  1. ocena jakości na podziale chronologicznym 80/20 (z 24-godzinną przerwą),
  2. trening modelu finalnego na wszystkich danych (ten model jest zapisywany).

Okna czasowe liczone są od ostatniej obserwacji w bazie.

Użycie:
  python train.py                          # trenuj na wszystkich danych
  python train.py --window-days 30         # trenuj na ostatnich 30 dniach
  python train.py --exclude-last-days 14   # pomiń ostatnie 14 dni (test retrain.py)
"""

import argparse
import logging
import pickle
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ── Konfiguracja ──────────────────────────────────────────────────────────────

DB_PATH     = Path("data/weather.db")
MODELS_DIR  = Path("models")
LOG_PATH    = Path("logs/train.log")
TARGET_HORIZON_H = 24   
HORIZON = timedelta(hours=TARGET_HORIZON_H)


MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
}

log = logging.getLogger(__name__)


def setup_logging(log_path: Path = LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ── Wczytywanie danych ────────────────────────────────────────────────────────

def load_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Wczytuje wszystkie poprawne dane z bazy, posortowane chronologicznie."""
    df = pd.read_sql_query("""
        SELECT * FROM weather_observations
        WHERE is_valid = 1
        ORDER BY observed_at
    """, conn)
    log.info("Wczytano %d rekordów", len(df))
    return df


# ── Inżynieria cech ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "wind_dir_sin", "wind_dir_cos", "pressure_msl", "cloud_cover",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "temp_lag_1h", "temp_lag_2h", "temp_lag_3h",
    "temp_lag_6h", "temp_lag_12h", "temp_lag_24h",
    "temp_rolling_6h", "temp_rolling_24h",
    "temp_diff_1h", "temp_diff_24h",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy cechy predykcyjne z surowych danych meteorologicznych.
    Zmienna docelowa: temperatura za TARGET_HORIZON_H godzin.
    """
    df = df.copy()                               
    df["observed_at"] = pd.to_datetime(df["observed_at"])
    df = df.sort_values("observed_at").reset_index(drop=True)   # chronologicznie, numeracja od 0

    # Wyrównanie szeregu do pełnych godzin: brakująca godzina staje się wierszem
    # z pustymi wartościami. Dzięki temu przesunięcia (cechy opóźnione, średnie,
    # zmienna docelowa) zawsze odnoszą się do właściwych godzin, a wiersze,
    # których cechy obejmowałyby lukę, są usuwane na końcu funkcji.
    df = df.set_index("observed_at").asfreq("h").reset_index()

    # ── Cechy czasowe ──
    df["hour"]        = df["observed_at"].dt.hour
    df["day_of_year"] = df["observed_at"].dt.dayofyear

    # Sezonowość zakodowana jako sinus/cosinus 
    # 2π · godzina / 24 = kąt na okręgu; para (sin, cos) wskazuje godzinę
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Kierunek wiatru zakodowany cyklicznie (sin/cos)
    df["wind_dir_sin"] = np.sin(2 * np.pi * df["wind_direction_10m"] / 360)
    df["wind_dir_cos"] = np.cos(2 * np.pi * df["wind_direction_10m"] / 360)

    # ── Cechy opóźnione (lagged features) ──
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"temp_lag_{lag}h"] = df["temperature_2m"].shift(lag)   

    # Średnie kroczące temperatury
    df["temp_rolling_6h"]  = df["temperature_2m"].rolling(6).mean()    
    df["temp_rolling_24h"] = df["temperature_2m"].rolling(24).mean()

    # Różnice temperatury (kierunek zmiany)
    df["temp_diff_1h"]  = df["temperature_2m"].diff(1)     
    df["temp_diff_24h"] = df["temperature_2m"].diff(24)   

    # ── Zmienna docelowa: temperatura za TARGET_HORIZON_H godzin ──
    df["target"] = df["temperature_2m"].shift(-TARGET_HORIZON_H)   

    # Usuwa wiersze z brakującymi wartościami (początek: lagi, koniec: target).
    # Sprawdzane są tylko kolumny modelu: pozostałe kolumny tabeli
    # (np. validation_reason, zawsze NULL dla poprawnych rekordów)
    # nie powoduje usuwania wierszy.
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    return df



# ── Trenowanie ────────────────────────────────────────────────────────────────

def create_model() -> xgb.XGBRegressor:
    """Tworzy nowy, niewytrenowany model z hiperparametrami MODEL_PARAMS."""
    return xgb.XGBRegressor(**MODEL_PARAMS)      # ** rozpakowuje słownik na argumenty nazwane


def fit_model(df: pd.DataFrame) -> xgb.XGBRegressor:
    """Trenuje nowy model na całym podanym zbiorze."""
    model = create_model()
    model.fit(df[FEATURE_COLS], df["target"])    
    return model


def score_model(model, df: pd.DataFrame,
                feature_cols: list[str] = FEATURE_COLS) -> tuple[float, float]:
    """Zwraca (MAE, RMSE) modelu na podanym zbiorze."""
    y_pred = model.predict(df[feature_cols])
    return (mean_absolute_error(df["target"], y_pred),
            root_mean_squared_error(df["target"], y_pred))


def window_label(window) -> str:
    """Opis okna danych: None = wszystkie dane, N = ostatnie N dni,
    "sezonowe" = okno sezonowe (eksperyment P2)."""
    if window is None:
        return "wszystkie dane"
    if window == "sezonowe":
        return "okno sezonowe"
    return f"{window} dni"


def train_model(df: pd.DataFrame, window_days: int | None,
                refit_full: bool = False) -> dict:
    """
    Trenuje model XGBoost i ocenia go na podziale chronologicznym 80/20.

    refit_full=False: zwracany jest model trenowany na pierwszych 80% danych.
    refit_full=True:  po ocenie model jest trenowany ponownie na 100% danych,
                      żeby znał najnowsze obserwacje. Metryki MAE/RMSE
                      pochodzą z oceny 80/20 i są szacunkiem jakości tego modelu.
    """
    # Podział chronologiczny
    split_idx = int(len(df) * 0.8)               # indeks wiersza, na którym kończy się 80%
    test_df = df.iloc[split_idx:]                # ostatnie 20% - zbiór testowy
    # 24-godzinna przerwa: do treningu trafiają tylko wiersze, których wartość
    # docelowa (t + 24 h) przypada przed początkiem zbioru testowego.
    train_df = df.iloc[:split_idx]
    train_df = train_df[train_df["observed_at"] + HORIZON < test_df["observed_at"].iloc[0]]

    log.info("Zbiór treningowy: %d rekordów | testowy: %d rekordów",
             len(train_df), len(test_df))

    model = fit_model(train_df)
    mae, rmse = score_model(model, test_df)      # ocena na danych niewidzianych

    log.info("MAE:  %.4f °C", mae)
    log.info("RMSE: %.4f °C", rmse)

    if refit_full:
        model = fit_model(df)                    # model finalny na 100% danych
        log.info("Model finalny wytrenowany na pełnym zbiorze: %d rekordów", len(df))

    return {
        "model":          model,
        "mae":            mae,
        "rmse":           rmse,
        "n_samples":      len(df),
        "training_start": str(df["observed_at"].min()),
        "training_end":   str(df["observed_at"].max()),
        "feature_cols":   FEATURE_COLS,
        "window_days":    window_days,
        "refit_full":     refit_full,
    }


# ── Zapisywanie modelu ────────────────────────────────────────────────────────

def save_model(result: dict, conn: sqlite3.Connection) -> Path:
    """
    Zapisuje model do pliku .pkl i rejestruje go w bazie oraz MLflow.
    """
    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")   
    model_path = MODELS_DIR / f"model_{timestamp}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump({                            # model zapisany razem z listą cech
            "model":        result["model"],
            "feature_cols": result["feature_cols"],
            "trained_at":   timestamp,
        }, f)

    log.info("Model zapisany: %s", model_path)

    # ── MLflow ──
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("weather_prediction")

    with mlflow.start_run() as run:
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("window_days",    result["window_days"] or "all")   # None -> "all"
        mlflow.log_param("refit_full",     result["refit_full"])
        mlflow.log_param("training_start", result["training_start"])
        mlflow.log_param("training_end",   result["training_end"])
        mlflow.log_metric("mae",           result["mae"])
        mlflow.log_metric("rmse",          result["rmse"])
        mlflow.log_artifact(str(model_path))     # kopia pliku modelu w MLflow
        run_id = run.info.run_id                 # identyfikator runu -> model_versions

    log.info("MLflow run_id: %s", run_id)

    # ── Baza danych ──
    # Dezaktywuje poprzedni aktywny model i zarejestruje nowy jako aktywny
    conn.execute("UPDATE model_versions SET is_active = 0")
    # ? - kolejne wartości z krotki poniżej; is_active = 1 wpisane na stałe
    conn.execute("""
        INSERT INTO model_versions
            (trained_at, training_start, training_end,
             n_samples, mae, rmse, model_path, is_active, mlflow_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
    """, (
        timestamp,
        result["training_start"],
        result["training_end"],
        result["n_samples"],
        result["mae"],
        result["rmse"],
        str(model_path),
        run_id,
    ))
    conn.commit()

    return model_path


# ── Raport ────────────────────────────────────────────────────────────────────

def print_report(result: dict, model_path: Path) -> None:
    print("\n" + "="*55)
    print("  WYNIKI TRENOWANIA")
    print("="*55)
    print(f"  Rekordów:               {result['n_samples']:,}")
    print(f"  Zakres danych:          {result['training_start'][:10]}"
          f"  →  {result['training_end'][:10]}")   # [:10] - sama data, bez godziny
    print(f"  Okno czasowe:           {window_label(result['window_days'])}")
    print("-"*55)
    print(f"  MAE  (ocena 80/20):  {result['mae']:.4f} °C")
    print(f"  RMSE (ocena 80/20):  {result['rmse']:.4f} °C")
    print(f"  Model finalny trenowany na 100% danych: "
          f"{'tak' if result['refit_full'] else 'nie'}")
    print("-"*55)
    print(f"  Model: {model_path}")
    print("="*55)
    print("\n  Aby zobaczyć eksperymenty w MLflow:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1")
    print("  Następnie otwórz: http://localhost:5000\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trenowanie modelu predykcji pogody")
    parser.add_argument(
        "--window-days", type=int, default=None,
        help="Ile dni danych użyć do trenowania (domyślnie: wszystkie)"
    )
    parser.add_argument(
        "--exclude-last-days", type=int, default=0,
        help=("Pomiń N ostatnich dni danych. Pozwala od razu przetestować "
              "retrain.py: pominięte dni pełnią rolę nowych danych.")
    )
    args = parser.parse_args()

    setup_logging()

    with closing(sqlite3.connect(DB_PATH)) as conn:
        # 1. Wczytywanie danych i zastowowanie okna czasowego
        #    (oba liczone od ostatniej obserwacji w bazie)
        df_raw = load_data(conn)
        observed = pd.to_datetime(df_raw["observed_at"])   # daty jako datetime (do porównań)
        end = observed.max() - timedelta(days=args.exclude_last_days)   # koniec danych (opcjonalnie bez ostatnich N dni)
        mask = observed <= end                   # maska True/False: dane do końca okresu
        if args.window_days:
            mask &= observed > end - timedelta(days=args.window_days)   # tylko ostatnie N dni
        df_raw = df_raw[mask]
        log.info("Dane treningowe: %d rekordów, do %s", len(df_raw), end)

        if len(df_raw) < 200:
            log.error("Za mało danych do trenowania (%d rekordów). "
                      "Minimum: 200.", len(df_raw))
            return

        # 2. Zbudowanie cech
        log.info("Buduje cechy predykcyjne...")
        df = build_features(df_raw)
        log.info("Po inżynierii cech: %d rekordów, %d cech",
                 len(df), len(FEATURE_COLS))

        # 3. Ocena modelu (80/20) i wytrenowanie modelu finalnego na 100% danych
        log.info("Trenuje model XGBoost...")
        result = train_model(df, args.window_days, refit_full=True)

        # 4. Zapisanie model
        model_path = save_model(result, conn)

        # 5. Raport
        print_report(result, model_path)


if __name__ == "__main__":
    main()