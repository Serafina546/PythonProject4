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

### Plots

## General plot
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(dates, Q_modelled, label="Modelled Discharge", linewidth=0.8, color="red")
ax.plot(dates, Q_obs, label="Observed Discharge", linewidth=0.8, color="black", alpha=0.7)

ax.set_xlabel("Date")
ax.set_ylabel("Discharge [mm/day]")
ax.set_title("Hydrological Model Krummbach: Modelled vs. Observed Discharge")
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Precipitation on y-Axis
ax2 = ax.twinx()
ax2.bar(dates, P, width=1.0, color="darkblue", alpha=0.2, label="Precipitation")
ax2.set_ylabel("Precipitation [mm/day]")
ax2.set_ylim(max(1, pd.Series(P).max() * 1.5), 0)
ax2.legend(loc="upper right")

fig.tight_layout()
plt.show()

## Example Plots

# Choose events
good_start, good_end = "2005-05-05", "2005-05-25"
bad_start, bad_end = "2008-05-01", "2008-06-30"

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharey=False)

# positive Example
mask_good = (dates >= good_start) & (dates <= good_end)
ax = axes[0]
ax.plot(dates[mask_good], Q_modelled[mask_good], label="Modelled Discharge", color="red", linewidth=1.2)
ax.plot(dates[mask_good], Q_obs[mask_good], label="Observed Discharge", color="black", linewidth=1.2, alpha=0.8)
ax.set_ylabel("Discharge [mm/day]")
ax.set_title(f"Positive Example Period: {good_start} to {good_end}")
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Precipitatoin on y-axis
ax2 = ax.twinx()
ax2.bar(dates[mask_good], P[mask_good], width=1.0, color="darkblue", alpha=0.3, label="Precipitation")
ax2.set_ylabel("Precipitation [mm/day]")
# turn the axis
max_p_good = max(1, pd.Series(P[mask_good]).max() * 1.5)
ax2.set_ylim(max_p_good, 0)


# negative example
mask_bad = (dates >= bad_start) & (dates <= bad_end)
ax_b = axes[1]
ax_b.plot(dates[mask_bad], Q_modelled[mask_bad], label="Modelled Discharge", color="red", linewidth=1.2)
ax_b.plot(dates[mask_bad], Q_obs[mask_bad], label="Observed Discharge", color="black", linewidth=1.2, alpha=0.8)
ax_b.set_xlabel("Date")
ax_b.set_ylabel("Discharge [mm/day]")
ax_b.set_title(f"Negative Example Period: {bad_start} to {bad_end}")
ax_b.legend(loc="upper left")
ax_b.grid(axis="y", linestyle="--", alpha=0.4)

# Precipitation
ax_b2 = ax_b.twinx()
ax_b2.bar(dates[mask_bad], P[mask_bad], width=1.0, color="darkblue", alpha=0.3, label="Precipitation")
ax_b2.set_ylabel("Precipitation [mm/day]")
max_p_bad = max(1, pd.Series(P[mask_bad]).max() * 1.5)
ax_b2.set_ylim(max_p_bad, 0)

fig.tight_layout()
plt.show()

## Example Plots 2

# Choose events
good_start, good_end = "2001-01-01", "2001-12-31"
bad_start, bad_end = "2002-01-01", "2002-12-31"

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharey=False)

# positive Example
mask_good = (dates >= good_start) & (dates <= good_end)
ax = axes[0]
ax.plot(dates[mask_good], Q_modelled[mask_good], label="Modelled Discharge", color="red", linewidth=1.2)
ax.plot(dates[mask_good], Q_obs[mask_good], label="Observed Discharge", color="black", linewidth=1.2, alpha=0.8)
ax.set_ylabel("Discharge [mm/day]")
ax.set_title(f"Positive Example Period: {good_start} to {good_end}")
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Precipitatoin on y-axis
ax2 = ax.twinx()
ax2.bar(dates[mask_good], P[mask_good], width=1.0, color="darkblue", alpha=0.3, label="Precipitation")
ax2.set_ylabel("Precipitation [mm/day]")
# turn the axis
max_p_good = max(1, pd.Series(P[mask_good]).max() * 1.5)
ax2.set_ylim(max_p_good, 0)


# negative example
mask_bad = (dates >= bad_start) & (dates <= bad_end)
ax_b = axes[1]
ax_b.plot(dates[mask_bad], Q_modelled[mask_bad], label="Modelled Discharge", color="red", linewidth=1.2)
ax_b.plot(dates[mask_bad], Q_obs[mask_bad], label="Observed Discharge", color="black", linewidth=1.2, alpha=0.8)
ax_b.set_xlabel("Date")
ax_b.set_ylabel("Discharge [mm/day]")
ax_b.set_title(f"Negative Example Period: {bad_start} to {bad_end}")
ax_b.legend(loc="upper left")
ax_b.grid(axis="y", linestyle="--", alpha=0.4)

# Precipitation
ax_b2 = ax_b.twinx()
ax_b2.bar(dates[mask_bad], P[mask_bad], width=1.0, color="darkblue", alpha=0.3, label="Precipitation")
ax_b2.set_ylabel("Precipitation [mm/day]")
max_p_bad = max(1, pd.Series(P[mask_bad]).max() * 1.5)
ax_b2.set_ylim(max_p_bad, 0)

fig.tight_layout()
plt.show()