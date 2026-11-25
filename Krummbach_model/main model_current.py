import matplotlib.pyplot as plt
import pandas as pd

from snow_model_function import run_snow_model
from PET_model_function import calculate_hargreaves_pet
from interception_model_function import run_interception_model
from flowerpot_quick_runoff_model_function import run_flowerpot_model
from groundwater_model_function import run_groundwater_model
from Glacial_melt_function import glacier_melt

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

## 2. interception model
# Input: Outflow (from snow model) and PET
# Output: Netto Precipitation (P_NET) and remaining PET for transpiration
# Parameter can be changed
P_net, Evap_icept, PET_transpiration, Unloading = run_interception_model(
    P=P,
    PET=PET,
    icept_threshold=0.5)

## 3. snow model
# input: Precipitation (P) and Temperature (T_mean)
# Output: SWE and Outflow
# Parameters can be changed
snow_params = dict(T_snow=1.0, T_rain=3.0, m_f=3.0, T_T=0.0, SWE0=0.0) # defined here for later input in glacial melt function
SWE_mod, Outflow, P_snow, P_rain, Melt = run_snow_model(
    P=P_net,
    T=T_mean,
    T_snow=1.0,
    T_rain=3.0,
    m_f=3.0,
    T_T=0.0,
    SWE0=0.0)

## Glacial melt function
SWE_glacier, Icemelt_glacier, Icemelt_catchment = glacier_melt(
    P=P,
    T=T_mean,
    glacier_area=2.0,       # km², noch anpassen
    catchment_area=80.0,    # km², noch anpassen
    ddf_ice=7.0,
    T0=0.0,
    snow_params=snow_params)

# Eisschmelze zum Input des Bodenmodells addieren
P_eff_total = Outflow + Icemelt_catchment

## 4. Flowerpot model (soil/storage model)
# Input: effective net precipitation and remaining PET
# Output: Output: Simulated discharge time series (Q), soil water storage (S), soil evaporation (ET_soil)
Q_fast, Q_interflow, Percolation, S_soil, ET_soil = run_flowerpot_model(
    P_eff=Outflow,
    PET_remaining=PET_transpiration,
    k=0.05,
    perc_max=2.5,
    S_max=150.0,
    beta=1.5,
    S0=0.0)

## 5. Groundwater model (Gesättigte Zone / Grundwasser)
# Berechnet den langsamen Basisabfluss aus der Perkolation
Q_baseflow, S_gw, GW_loss = run_groundwater_model(
    Percolation=Percolation,
    k_gw=0.05,
    alpha=1.0,
    loss_fraction=0.0,
    S_gw0=100.0)

## 6. Gesamtabfluss (Summe aus allen Abflusskomponenten)
Q_modelled = Q_fast + Q_interflow + Q_baseflow + Icemelt_catchment


### Plots
# 1. Hydrologisches Jahr definieren und Tag des Jahres berechnen (falls noch nicht geschehen)
data['hydro_year'] = data.index.map(lambda x: int(x.year + 1) if x.month >= 10 else int(x.year))
def get_hydro_day(row):
    h_year = int(row['hydro_year'])
    start_date = pd.Timestamp(year=h_year - 1, month=10, day=1)
    return (row.name - start_date).days + 1

data['hydro_day'] = data.apply(get_hydro_day, axis=1)

# Dem DataFrame die Variablen zuweisen, falls noch nicht passiert
data['Q_modelled'] = Q_modelled
data['Q_obs'] = Q_obs

# 2. Schleife über jedes einzelne hydrologische Jahr
for hydro_year, group in data.groupby('hydro_year'):
    # Nur Jahre mit nahezu vollständigen Daten plotten (mind. 350 Tage)
    if len(group) > 350:
        plt.figure(figsize=(10, 4))

        # Beobachteter Abfluss (Q_obs)
        plt.plot(group['hydro_day'], group['Q_obs'], label='Beobachtet ($Q_{obs}$)', color='black', linewidth=1.5,
                 alpha=0.8)

        # Modellierter Abfluss (Q_modelled)
        plt.plot(group['hydro_day'], group['Q_modelled'], label='Modelliert ($Q_{modelled}$)', color='red',
                 linewidth=1.5)

        # Beschriftungen und Layout
        plt.xlabel("Tag des hydrologischen Jahres (1. Okt = Tag 1)", fontsize=10)
        plt.ylabel("Abfluss (mm/d)", fontsize=10)
        plt.title(f"Hydrologisches Jahr {hydro_year} (1. Okt {hydro_year - 1} – 30. Sep {hydro_year})", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(1, 366)
        plt.tight_layout()

        plt.show()

        comp = pd.DataFrame({
            "Q_obs": Q_obs,
            "Q_modelled": Q_modelled,
            "Q_fast": Q_fast,
            "Q_interflow": Q_interflow,
            "Q_baseflow": Q_baseflow,
            "Icemelt": Icemelt_catchment,
            "Schnee-Outflow": Outflow,
        }, index=dates)

        monthly = comp.groupby(comp.index.month).mean()
        print(monthly.round(2))

        ax = monthly[["Q_fast", "Q_interflow", "Q_baseflow", "Icemelt"]].plot(
            kind="bar", stacked=True, figsize=(10, 4), alpha=0.7)
        ax.plot(range(12), monthly["Q_obs"], color="black", marker="o", label="Q_obs")
        ax.set_xticklabels(range(1, 13), rotation=0)
        ax.set_xlabel("Monat");
        ax.set_ylabel("mm/d")
        ax.legend()
        plt.tight_layout();
        plt.show()