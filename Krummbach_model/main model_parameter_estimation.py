### main model parameter estimation ###

import matplotlib.pyplot as plt
import pandas as pd

from snow_model_function import run_snow_model
from PET_model_function import calculate_hargreaves_pet
from interception_model_function import run_interception_model
from flowerpot_model_function import run_flowerpot_model

plt.rcParams["font.family"] = "Arial"

# read and merge data
data_sim = pd.read_csv(r"../data/CAMELS_CH_sim_based_2244.csv", parse_dates=["date"])
data_obs = pd.read_csv(r"../data/CAMELS_CH_obs_based_2244.csv", parse_dates=["date"])

data_obs = data_obs.set_index("date")
data_sim = data_sim.set_index("date")
data = data_obs.join(data_sim, how="inner", suffixes=("_obs", "_sim"))

# extract variables
dates = data.index
P = data["precipitation(mm/d)"].values
T_mean = data["temperature_mean(degC)"].to_numpy(dtype=float)
T_min = data["temperature_min(degC)"].to_numpy(dtype=float)
T_max = data["temperature_max(degC)"].to_numpy(dtype=float)
Q_obs = data["discharge_spec(mm/d)"].to_numpy(dtype=float)

P = pd.Series(P).fillna(0.0).to_numpy()

### Model

## 1. Calculate PET
PET = calculate_hargreaves_pet(
    dates=dates,
    T_mean=T_mean,
    T_min=T_min,
    T_max=T_max,
    latitude_deg=46.230046)

## 2. snow model
# input: Precipitation (P) and Temperature (T_mean)
# Output: SWE and Outflow
# Parameters can be changed
SWE_mod, Outflow, P_snow, P_rain, Melt = run_snow_model(
    P=P,
    T=T_mean,
    T_min=1.0,
    T_max=3.0,
    m_f=3.0,
    T_T=0.0,
    SWE0=0.0)

## 3. interception model
# Input: Outflow (from snow model) and PET
# Output: Netto Precipitation (P_NET) and remaining PET for transpiration
# Parameter can be changed
P_net, Evap_icept, PET_transpiration, Unloading = run_interception_model(
    P=Outflow,
    PET=PET,
    icept_threshold=0.5)

## 4. Flowerpot model (soil/storage model)
# Input: effective net precipitation and remaining PET
# Output: Output: Simulated discharge time series (Q), soil water storage (S), soil evaporation (ET_soil)
Q_modelled, S_soil, ET_soil = run_flowerpot_model(
    P_eff=P_net,
    PET_remaining=PET_transpiration,
    k=0.05,
    S0=0.0)

# 1. Zeitraum-Filter definieren
warmup_data = data.loc["1981-01-01":"1982-12-31"]
calib_data = data.loc["1983-01-01":"2009-12-31"]
val_data = data.loc["2010-01-01":"2020-12-31"]

### 2. objective functions implementieren
def compute_nse(q_obs, q_sim):
  # Ungültige Werte (NaNs) filtern falls vorhanden
  mask = ~np.isnan(q_obs) & ~np.isnan(q_sim)
  q_obs, q_sim = q_obs[mask], q_sim[mask]
  return 1 - np.sum((q_obs - q_sim) ** 2) / np.sum((q_obs - np.mean(q_obs)) ** 2)


def compute_log_nse(q_obs, q_sim):
  mask = ~np.isnan(q_obs) & ~np.isnan(q_sim) & (q_obs > 0) & (q_sim > 0)
  q_obs, q_sim = np.log(q_obs[mask]), np.log(q_sim[mask])
  return 1 - np.sum((q_obs - q_sim) ** 2) / np.sum((q_obs - np.mean(q_obs)) ** 2)


def compute_bias(q_obs, q_sim):
  mask = ~np.isnan(q_obs) & ~np.isnan(q_sim)
  q_obs, q_sim = q_obs[mask], q_sim[mask]
  return np.sum(q_sim - q_obs) / np.sum(q_obs) * 100  # in Prozent

### 3. Modellkette für einen Zeitraum bündeln
def run_model(params, df):
  # Parameter entpacken (Beispiel)
  T_min_snow, T_max_snow, m_f, icept_threshold, k = params

  # 1. PET berechnen (oder vorher einmalig für alle Daten)
  PET = calculate_hargreaves_pet(
      dates=df.index,
      T_mean=df["temperature_mean(degC)"].values,
      T_min=df["temperature_min(degC)"].values,
      T_max=df["temperature_max(degC)"].values,
      latitude_deg=46.230046,
  )

  # 2. Schnee
  _, Outflow, _, _, _ = run_snow_model(
      P=df["precipitation(mm/d)"].values,
      T_mean=df["temperature_mean(degC)"].values,
      T_min=T_min_snow,
      T_max=T_max_snow,
      m_f=m_f,
      T_T=0.0,
      SWE0=0.0,
  )

  # 3. Interzeption
  P_net, _, PET_transpiration, _ = run_interception_model(
      P=Outflow, PET=PET, icept_threshold=icept_threshold
  )

  # 4. Boden/Flowerpot-Modell
  Q_modelled, _, _ = run_flowerpot_model(
      P_eff=P_net, PET_remaining=PET_transpiration, k=k, S0=0.0
  )

  return Q_modelled

### 4. Kalibirierung der Parameter
from scipy.optimize import minimize


def objective_function(params):
  Q_sim = run_model(params, calib_data)
  Q_obs = calib_data["discharge_spec(mm/d)"].values
  # Wir minimieren 1 - NSE (weil minimize eine Kostenfunktion minimiert)
  return 1 - compute_nse(Q_obs, Q_sim)


# Startwerte für [T_min, T_max, m_f, icept_threshold, k]
initial_guess = [1.0, 3.0, 3.0, 0.5, 0.05]
bounds = [
    (0.0, 2.0),
    (2.0, 5.0),
    (1.0, 6.0),
    (0.0, 2.0),
    (0.001, 0.5),
]

result = minimize(
    objective_function, initial_guess, bounds=bounds, method="Nelder-Mead"
)
best_params = result.x
print("Beste Parameter:", best_params)

### 5. Validierung und Auswertung
# Simulation für den Validierungszeitraum mit den kalibrierten Parametern
Q_val_sim = run_model(best_params, val_data)
Q_val_obs = val_data["discharge_spec(mm/d)"].values

# Metriken berechnen
nse_val = compute_nse(Q_val_obs, Q_val_sim)
log_nse_val = compute_log_nse(Q_val_obs, Q_val_sim)
bias_val = compute_bias(Q_val_obs, Q_val_sim)

print(f"Validation NSE: {nse_val:.3f}")
print(f"Validation Log-NSE: {log_nse_val:.3f}")
print(f"Validation Bias: {bias_val:.2f}%")