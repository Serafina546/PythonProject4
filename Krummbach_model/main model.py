import matplotlib.pyplot as plt
import pandas as pd

from snow_model_function import run_snow_model

plt.rcParams["font.family"] = "Arial"

# read and merge data
data_sim = pd.read_csv(r"../data/CAMELS_CH_sim_based_2244.csv", parse_dates=["date"])
data_obs = pd.read_csv(r"../data/CAMELS_CH_obs_based_2244.csv", parse_dates=["date"])

data_obs = data_obs.set_index("date")
data_sim = data_sim.set_index("date")
data = data_obs.join(data_sim, how="inner", suffixes=("_obs", "_sim"))

# extract variables
P = data["precipitation(mm/d)"].values          # Precipitation
T = data["temperature_mean(degC)"].values       # Temperature

# Model

SWE_mod, Outflow, P_snow, P_rain, Melt = run_snow_model(
    P=P,
    T=T,
    T_min=1.0,
    T_max=3.0,
    m_f=2.0,
    T_T=0.0,
    SWE0=0.0)