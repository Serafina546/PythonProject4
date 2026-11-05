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