"""
calculate_statistics.py
-----------------------
Statystyki opisowe zbioru danych (rozdział 2 pracy):
  1. średnie miesięczne temperatury (rozdział 2.5),
  2. statystyki opisowe zmiennych meteorologicznych (tabela 3, rozdział 2.4).

Użycie:
  python calculate_statistics.py
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/weather.db")

df = pd.read_sql_query(
    "SELECT * FROM weather_observations WHERE is_valid = 1", conn
)

df["observed_at"] = pd.to_datetime(df["observed_at"])   
df["month"] = df["observed_at"].dt.month                

# Średnia temperatura per miesiąc
monthly = df.groupby("month")["temperature_2m"].mean().round(1)   # średnia ze wszystkich godzin miesiąca
print("Średnia temperatura per miesiąc:")
print(monthly)

# Najcieplejszy i najchłodniejszy miesiąc
warmest = monthly.idxmax()      
coldest = monthly.idxmin()    
print(f"\nNajcieplejszy miesiąc: {warmest} → {monthly[warmest]}°C")
print(f"Najchłodniejszy miesiąc: {coldest} → {monthly[coldest]}°C")
print(f"Różnica: {(monthly[warmest] - monthly[coldest]):.1f}°C")


# ── Tabela 3: statystyki opisowe ─────────────────────────────────────────────

# (kolumna, etykieta w tabeli), kierunek wiatru pominięty - zmienna kołowa
VARIABLES = [
    ("temperature_2m",       "temperature_2m (C)"),
    ("relative_humidity_2m", "relative_humidity_2m (%)"),
    ("precipitation",        "precipitation (mm)"),
    ("wind_speed_10m",       "wind_speed_10m (km/h)"),
    ("pressure_msl",         "pressure_msl (hPa)"),
    ("cloud_cover",          "cloud_cover (%)"),
]

# Braki danych liczone na wszystkich rekordach w bazie (także odrzuconych
# przez walidacje), pozostałe statystyki - na rekordach poprawnych.
all_rows = pd.read_sql_query("SELECT * FROM weather_observations", conn)

print("\n" + "=" * 86)
print(f"  Tabela 3. Statystyki opisowe | okres: {df['observed_at'].min():%Y-%m-%d} - "
      f"{df['observed_at'].max():%Y-%m-%d} | obserwacji: {len(df)}")
print("=" * 86)
# :<28 / :>8 - szerokość kolumny i wyrównanie (do lewej / do prawej)
print(f"  {'Zmienna':<28} {'Min':>8} {'Max':>8} {'Średnia':>9} {'Odch. stand.':>13} {'Braki danych':>13}")
print("-" * 86)
for col, name in VARIABLES:
    s = df[col]                                   
    missing = all_rows[col].isna().mean() * 100   # % brakujących wartości
    print(f"  {name:<28} {s.min():>8.1f} {s.max():>8.1f} {s.mean():>9.2f} "
          f"{s.std():>13.2f} {missing:>12.1f}%")   
print("=" * 86)

conn.close()
