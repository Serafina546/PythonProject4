import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["font.family"] = "Arial"

### Linear Storage Model with real precipitation data, for the full length of the time series 01.01.1981- 31.12.2020
# CSV-Datei einlesen
script_dir = Path(__file__).resolve().parent
csv_path = (script_dir/ "data"/ "CAMELS_CH_obs_based_2244.csv")
data = pd.read_csv(csv_path)

# Niederschlagsdaten aus der CSV-Datei verwenden [mm/day]
P = data["precipitation(mm/d)"].to_numpy()

# Fehlende Niederschlagswerte behandeln
P = np.nan_to_num(P, nan=0.0)

# Anzahl der Zeitschritte
n_steps = len(P)

### PET: Anwendung der Hargreaves-Samani Methode für die Berechnung der PET
# Datum und Temperaturdaten
data["date"] = pd.to_datetime(data["date"])
T_mean = data["temperature_mean(degC)"].to_numpy(dtype=float)
T_min = data["temperature_min(degC)"].to_numpy(dtype=float)
T_max = data["temperature_max(degC)"].to_numpy(dtype=float)

# Breitengrad des Einzugsgebiets (Talboden)
LATITUDE_DEG = 46.230046

# Kalendertag des Jahres
J = data["date"].dt.day_of_year.to_numpy()

# Breitengrad in Radiant
phi = np.radians(LATITUDE_DEG)

# Extraterrestrische Strahlung berechnen
dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
Gsc = 0.0820  # MJ m-2 min-1
Ra = ((24 * 60 / np.pi)* Gsc* dr* (ws * np.sin(phi) * np.sin(delta)+ np.cos(phi) * np.cos(delta) * np.sin(ws)))

# Hargreaves-Samani: PET in mm/day
temperature_range = np.maximum(T_max - T_min, 0)
PET = (0.0023* (T_mean + 17.8)* np.sqrt(temperature_range)* 0.408* Ra)

# Negative PET-Werte vermeiden
PET = np.maximum(PET, 0)

# Fehlende Temperaturwerte nicht als Null interpretieren
invalid = (~np.isfinite(T_mean)| ~np.isfinite(T_min)| ~np.isfinite(T_max))
PET[invalid] = np.nan
print("Mittlere PET:", np.nanmean(PET), "mm/day")
print("Maximale PET:", np.nanmax(PET), "mm/day")


#### Interception
Icept_threshold = 0.25  # Anpassen für Kalibrierung (in mm/day)
Icept = np.minimum(P, Icept_threshold)
Evap_icept = np.minimum(PET, Icept)
Unloading = Icept - Evap_icept
P_eff = P - Evap_icept

# Wieder wie in Aufgabe 2a
# Parameter des Speichermodells
k = 0.1 #anpassen für die kalibrierung!
S0 = 0.0

# Arrays für Speicher und Abfluss
S = np.zeros(n_steps)
Q = np.zeros(n_steps)
S_prev = S0

# Wasserbilanz berechnen
for t in range(n_steps):
    Q[t] = k * S_prev
    S[t] = S_prev + P[t] - Q[t]
    S_prev = S[t]

# Arrays für Speicher, Abfluss und tatsächliche Verdunstung
S = np.zeros(n_steps)
Q = np.zeros(n_steps)
ET_soil = np.zeros(n_steps)
S_prev = S0

for t in range(n_steps):
    # Abfluss aus dem Speicher zu Beginn des Zeitschritts
    Q[t] = k * S_prev

    # Speicher nach Abfluss und effektivem Niederschlag
    S_available = max(0.0, S_prev - Q[t] + P_eff[t])

    # Verbleibende PET nach Verdunstung der Interzeption
    PET_remaining = max(0.0, PET[t] - Evap_icept[t])

    # Tatsächliche Verdunstung aus dem Speicher
    ET_soil[t] = min(PET_remaining, S_available)

    # Neuer Speicher
    S[t] = S_available - ET_soil[t]
    S_prev = S[t]

# Beobachteten Abfluss aus der CSV laden
Q_obs = data["discharge_spec(mm/d)"].to_numpy(dtype=float)

# Datum für die x-Achse
dates = data["date"]
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates, Q, label="Modelled discharge",linewidth=0.7, color="green")
ax.plot(dates, Q_obs, label="Observed discharge",linewidth=0.7, color="black")
ax.set_xlabel("Date")
ax.set_ylabel("Discharge [mm/day]")
ax.set_title("Modelled vs Observed Discharge")
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Niederschlag auf der rechten y-achse
ax2 = ax.twinx()
ax2.bar(dates, P, width=1.0, color="darkblue",alpha=0.3, label="Precipitation")
ax2.set_ylabel("Precipitation [mm/day]")
ax2.set_ylim(max(1, np.nanmax(P) * 1.2), 0)
fig.tight_layout()
plt.show()







