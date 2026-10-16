"""
collect_data.py
---------------
Pobiera dane meteorologiczne z Open-Meteo Historical API i zapisuje je do bazy SQLite.

Wszystkie znaczniki czasu (observed_at, inserted_at, run_at) przechowywane są w UTC.

Użycie:
  python collect_data.py --mode history   # pobiera 2 lata danych historycznych
  python collect_data.py --mode daily     # uzupełnia dane od ostatniego poprawnego rekordu
  python collect_data.py --mode range --start 2023-01-01 --end 2023-12-31
                                          # pobiera dane z podanego zakresu (daty UTC, włącznie)


Przykładowa odpowiedź z API (skrócona, tylko 3 godziny):                                                                                    
data = {
    "latitude": 51.1,               # współrzędne dopasowane do siatki modelu
    "longitude": 17.04,
    "generationtime_ms": 0.42,
    "utc_offset_seconds": 0,
    "timezone": "GMT",
    "timezone_abbreviation": "GMT",
    "elevation": 120.0,
    "hourly_units": {               # jednostki – kod ich nie używa
        "time": "iso8601",
        "temperature_2m": "°C",
        "relative_humidity_2m": "%",
        "precipitation": "mm",
        "wind_speed_10m": "km/h",
        "wind_direction_10m": "°",
        "pressure_msl": "hPa",
        "cloud_cover": "%"
    },
    "hourly": {                     # dane godzinowe – kod je przetwarza
        "time":                 ["2025-01-15T00:00", "2025-01-15T01:00", "2025-01-15T02:00"],
        "temperature_2m":       [1.0,    0.9,    0.9],
        "relative_humidity_2m": [89,     95,     95],
        "precipitation":        [0.2,    0.0,    0.0],
        "wind_speed_10m":       [16.2,   17.5,   17.8],
        "wind_direction_10m":   [280,    287,    288],
        "pressure_msl":         [1025.5, 1025.5, 1025.0],
        "cloud_cover":          [100,    100,    100]
    }
}
"""



import argparse
import logging
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

# ── Konfiguracja ──────────────────────────────────────────────────────────────

LATITUDE  = 51.11   # Wrocław
LONGITUDE = 17.04
CITY_NAME = "Wroclaw"

DB_PATH   = Path("data/weather.db")
LOG_PATH  = Path("logs/collect.log")

HISTORY_DAYS        = 730   # zakres trybu history
CHUNK_DAYS          = 30    # długość pojedynczego zapytania do API
DAILY_LOOKBACK_DAYS = 7     # zapas na opóźnienie publikacji danych w archiwum

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
    "cloud_cover",
]

# ── Logging ───────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)


def setup_logging() -> None:
    """Konfiguruje logowanie do pliku i na konsolę (wywoływana w main)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)   
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ── Baza danych ───────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    """Tworzy tabele, jeśli nie istnieją."""
    
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS weather_observations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            city            TEXT    NOT NULL,
            observed_at     TEXT    NOT NULL,          -- UTC, ISO 8601, np. 2024-01-01T00:00
            temperature_2m          REAL,
            relative_humidity_2m    REAL,
            precipitation           REAL,
            wind_speed_10m          REAL,
            wind_direction_10m      REAL,
            pressure_msl            REAL,
            cloud_cover             REAL,
            is_valid        INTEGER DEFAULT 1,         -- 0 jeśli walidacja odrzuciła rekord
            validation_reason TEXT,                    -- powód odrzucenia (NULL dla poprawnych)
            inserted_at     TEXT    DEFAULT (datetime('now')),   -- UTC
            UNIQUE(city, observed_at)                  -- bez duplikatów
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at      TEXT    NOT NULL,
            training_start  TEXT    NOT NULL,          -- początek okna treningowego
            training_end    TEXT    NOT NULL,          -- koniec okna treningowego
            n_samples       INTEGER,
            mae             REAL,
            rmse            REAL,
            model_path      TEXT,
            is_active       INTEGER DEFAULT 0,         -- 1 = aktualnie używany model
            mlflow_run_id   TEXT
        );

        CREATE TABLE IF NOT EXISTS pipeline_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at          TEXT    DEFAULT (datetime('now')),   -- UTC
            stage           TEXT    NOT NULL,          -- 'collect_history', 'collect_daily', 'collect_range'
            status          TEXT    NOT NULL,          -- 'ok', 'warning', 'error'
            rows_processed  INTEGER,
            rows_rejected   INTEGER,
            message         TEXT
        );
    """)
    conn.commit()
    log.info("Baza danych zainicjalizowana: %s", DB_PATH)


# ── Walidacja ─────────────────────────────────────────────────────────────────

VALIDATION_RULES = {
    "temperature_2m":        (-50.0, 60.0),
    "relative_humidity_2m":  (0.0,  100.0),
    "precipitation":         (0.0,  200.0),
    "wind_speed_10m":        (0.0,  300.0),
    "wind_direction_10m":    (0.0,  360.0),
    "pressure_msl":          (850.0, 1100.0),
    "cloud_cover":           (0.0,  100.0),
}


def validate_row(row: dict) -> tuple[bool, str]:
    """
    Sprawdza pojedynczy rekord pod kątem reguł jakości.
    Zwraca (is_valid, reason).
    """
    for field, (min_val, max_val) in VALIDATION_RULES.items():   
        value = row.get(field)
        if value is None:
            return False, f"brak wartości: {field}"
        if not (min_val <= value <= max_val):
            return False, f"{field}={value} poza zakresem [{min_val}, {max_val}]"
    return True, "ok"


# ── Pobieranie danych ─────────────────────────────────────────────────────────

def fetch_weather(date_from: str, date_to: str) -> list[dict]:
    """
    Pobiera dane godzinowe (UTC) z Open-Meteo Historical API.
    date_from, date_to: format YYYY-MM-DD, zakres włącznie.
    Przy błędzie lub braku danych zwraca pustą listę.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": date_from,
        "end_date":   date_to,
        "hourly":     ",".join(VARIABLES),   # "temperature_2m,relative_humidity_2m,..."
        "timezone":   "GMT",                 # czas w UTC
    }

    log.info("Pobieram dane: %s -> %s", date_from, date_to)
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()         
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:   
        log.error("Błąd pobierania: %s", e)
        return []

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    if not times:
        log.warning("API zwróciło puste dane dla zakresu %s – %s", date_from, date_to)
        return []

    # API zwraca kolumny (zmienna -> lista godzin), baza potrzebuje wierszy (godzina -> zmienne)
    rows = []
    for i, ts in enumerate(times):
        row = {"observed_at": ts, "city": CITY_NAME}
        for var in VARIABLES:
            values = hourly.get(var, [])
            row[var] = values[i] if i < len(values) else None   # brak wartości -> None
        rows.append(row)

    log.info("Pobrano %d rekordów", len(rows))
    return rows


# ── Zapis do bazy ─────────────────────────────────────────────────────────────

# Nowy rekord jest wstawiany. Istniejący rekord jest nadpisywany tylko wtedy,
# gdy był niepoprawny, a nowa wersja jest poprawna. Dzięki temu godziny,
# których archiwum wcześniej nie udostępniało, zostaną uzupełnione.
UPSERT_SQL = """
    INSERT INTO weather_observations
        (city, observed_at, temperature_2m, relative_humidity_2m,
         precipitation, wind_speed_10m, wind_direction_10m,
         pressure_msl, cloud_cover, is_valid, validation_reason)
    VALUES
        (:city, :observed_at, :temperature_2m, :relative_humidity_2m,
         :precipitation, :wind_speed_10m, :wind_direction_10m,
         :pressure_msl, :cloud_cover, :is_valid, :validation_reason)
    ON CONFLICT(city, observed_at) DO UPDATE SET
        temperature_2m       = excluded.temperature_2m,
        relative_humidity_2m = excluded.relative_humidity_2m,
        precipitation        = excluded.precipitation,
        wind_speed_10m       = excluded.wind_speed_10m,
        wind_direction_10m   = excluded.wind_direction_10m,
        pressure_msl         = excluded.pressure_msl,
        cloud_cover          = excluded.cloud_cover,
        is_valid             = excluded.is_valid,
        validation_reason    = excluded.validation_reason,
        inserted_at          = datetime('now')
    WHERE weather_observations.is_valid = 0
      AND excluded.is_valid = 1
"""


def save_to_db(conn: sqlite3.Connection, rows: list[dict]) -> tuple[int, int]:
    """
    Zapisuje rekordy do bazy.
    Zwraca (saved, rejected):
      saved    - rekordy wstawione lub naprawione,
      rejected - spośród nich rekordy zapisane z is_valid = 0.
    """
    saved = 0
    rejected = 0

    for row in rows:
        is_valid, reason = validate_row(row)     

        try:
            # rekord + wynik walidacji -> parametry :nazwa w UPSERT_SQL
            cursor = conn.execute(UPSERT_SQL, {
                **row,
                "is_valid": int(is_valid),
                "validation_reason": None if is_valid else reason,
            })
        except sqlite3.Error as e:
            log.error("Błąd zapisu rekordu %s: %s", row.get("observed_at"), e)
            continue

        if cursor.rowcount > 0:         # 0 = rekord już był i nie wymagał zmian
            saved += 1
            if not is_valid:
                rejected += 1

    conn.commit()                                
    return saved, rejected


def log_pipeline_run(conn, stage, status, rows_processed, rows_rejected, message=""):
    conn.execute("""
        INSERT INTO pipeline_logs (stage, status, rows_processed, rows_rejected, message)
        VALUES (?, ?, ?, ?, ?)
    """, (stage, status, rows_processed, rows_rejected, message))
    conn.commit()


# ── Pobieranie zakresu dat ────────────────────────────────────────────────────

def collect_range(conn: sqlite3.Connection, start: date, end: date, stage: str) -> None:
    """
    Pobiera dane z zakresu [start, end] w kawałkach po CHUNK_DAYS dni.
    Błąd jednego zapytania nie przerywa pobierania pozostałych kawałków.
    """
    log.info("Etap: %s | zakres: %s -> %s", stage, start, end)

    total_saved    = 0
    total_rejected = 0
    chunks         = 0
    failed_chunks  = 0

    current = start
    while current <= end:                        # pobieranie w kawałkach po CHUNK_DAYS dni
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end)   # nie dalej niż end
        chunks += 1

        rows = fetch_weather(str(current), str(chunk_end))   # daty jako "YYYY-MM-DD"
        if not rows:
            failed_chunks += 1
        else:
            saved, rejected = save_to_db(conn, rows)
            total_saved    += saved
            total_rejected += rejected

        current = chunk_end + timedelta(days=1)  # następny kawałek od kolejnego dnia

   
    if failed_chunks == 0:
        status = "ok"
    elif failed_chunks < chunks:
        status = "warning"
    else:
        status = "error"

    message = (f"Zakres: {start} -> {end} | nieudane zapytania: "
               f"{failed_chunks}/{chunks}")
    log.info("Etap %s zakończony [%s] | zapisano: %d | niepoprawne: %d | %s",
             stage, status, total_saved, total_rejected, message)
    log_pipeline_run(conn, stage, status, total_saved, total_rejected, message)


# ── Tryby uruchomienia ────────────────────────────────────────────────────────

def run_history(conn: sqlite3.Connection) -> None:
    """Pobiera dane historyczne z ostatnich HISTORY_DAYS dni."""
    end   = datetime.now(timezone.utc).date()
    start = end - timedelta(days=HISTORY_DAYS)   
    collect_range(conn, start, end, "collect_history")


def run_daily(conn: sqlite3.Connection) -> None:
    """
    Uzupełnia dane od dnia ostatniego poprawnego rekordu do dziś,
    zawsze obejmując co najmniej DAILY_LOOKBACK_DAYS dni wstecz.
    Po przerwie w uruchomieniach pobiera cały brakujący okres.
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=DAILY_LOOKBACK_DAYS)

    # data ostatniego poprawnego rekordu (None przy pustej bazie)
    last_valid = conn.execute(
        "SELECT MAX(observed_at) FROM weather_observations WHERE is_valid = 1"
    ).fetchone()[0]                              

    if last_valid:
        start = min(start, datetime.fromisoformat(last_valid).date())   

    collect_range(conn, start, end, "collect_daily")


# ── Raport po uruchomieniu ────────────────────────────────────────────────────

def print_summary(conn: sqlite3.Connection) -> None:
    total, oldest, newest = conn.execute("""
        SELECT COUNT(*), MIN(observed_at), MAX(observed_at)
        FROM weather_observations WHERE is_valid = 1
    """).fetchone()

    print("\n" + "="*50)
    print(f"  Baza danych: {DB_PATH}")
    print(f"  Rekordów (valid): {total:,}")
    print(f"  Zakres (UTC):  {oldest}  ->  {newest}")
    print("="*50 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pobieranie danych pogodowych")
    parser.add_argument(
        "--mode",
        choices=["history", "daily", "range"],
        default="daily",
        help="history = 2 lata wstecz | daily = uzupełnienie od ostatniego poprawnego rekordu"
             " | range = zakres podany w --start i --end",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,         
        help="początek zakresu w formacie YYYY-MM-DD (tylko dla --mode range)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        help="koniec zakresu w formacie YYYY-MM-DD, włącznie (tylko dla --mode range)",
    )
    args = parser.parse_args()                   

    # kontrola poprawności argumentów trybu range
    if args.mode == "range":
        if args.start is None or args.end is None:
            parser.error("--mode range wymaga podania --start i --end")
        if args.start > args.end:
            parser.error("--start nie może być późniejszy niż --end")
    elif args.start is not None or args.end is not None:
        parser.error("--start i --end można podać tylko z --mode range")

    setup_logging()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)    

    # closing() zamyka połączenie z bazą po zakończeniu bloku
    with closing(sqlite3.connect(DB_PATH)) as conn:
        init_db(conn)

        if args.mode == "history":
            run_history(conn)
        elif args.mode == "range":
            collect_range(conn, args.start, args.end, "collect_range")
        else:
            run_daily(conn)

        print_summary(conn)


if __name__ == "__main__":       
    main()
