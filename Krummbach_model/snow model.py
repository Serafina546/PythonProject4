### Snow model ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Arial"

# Load CSV files
data_sim = pd.read_csv(r"../data/CAMELS_CH_sim_based_2244.csv", parse_dates=["date"])
data_obs = pd.read_csv(r"../data/CAMELS_CH_obs_based_2244.csv", parse_dates=["date"])


# Merge the datasets
data_obs = data_obs.set_index("date")
data_sim = data_sim.set_index("date")

data = data_obs.join(data_sim, how="inner", suffixes=("_obs", "_sim"))

'''
print(data)
print(data.columns)
first_valid_date = data["swe(mm)"].first_valid_index()
print("First date with observations:", first_valid_date)
'''

# Extract variables
time = data.index
P = data["precipitation(mm/d)"].values      # Precipitation
T = data["temperature_mean(degC)"].values   # Daily mean temperature
SWE = data["swe(mm)"].values                # Snow Water Equivalence
                                            # SWE has only been measured since 1998-09-02 00:00:00
n = len(data)                               # n steps

SWE0 = 0.0 # initial SWE

# Set parameters
T_min = 1.0         # below this: 100% snow (1.0)
T_max = 3.0         # above this: 100% rain (3.0)
m_f = 2.0           # degree-day melt factor [mm/(°C * day)]
T_T = 0.0           # melt threshold temperature

# Create arrays for model outputs
SWE_mod = np.zeros(n)
P_snow = np.zeros(n)
P_rain = np.zeros(n)
Melt = np.zeros(n)
Outflow = np.zeros(n)

SWE_prev = SWE0

# Running snow model loop
for t in range(n):
    # Partition into snow and rain based on air temperature
    if T[t] <= T_min:
        snow_fraction = 1.0
    elif T[t] >= T_max:
        snow_fraction = 0.0
    else:
        snow_fraction = (T_max - T[t])/(T_max - T_min)

    P_snow[t] = P[t] * snow_fraction
    P_rain[t] = P[t] * (1.0 - snow_fraction)

    # Calculate potential melt using temperature-index model
    if T[t] > T_T:
        potential_melt = m_f * (T[t] - T_T)
    else:
        potential_melt = 0

    # Snow water equivalent (SWE) water balance
    available_snow = SWE_prev + P_snow[t]
    Melt[t] = min(potential_melt, available_snow)
    SWE_mod[t] = available_snow - Melt[t]

    # Total outflow (Rainfall + Meltwater)
    Outflow[t] = P_rain[t] + Melt[t]

    # Update or next time step
    SWE_prev = SWE_mod[t]

## Plots to compare modeled and observed SWE

# filter data for years between 1999 until 2020 (years with observed data)
data_99_20 = data.loc["1999-01-01":"2020-12-31"]

# Only plot years 1999 until 2020
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data_99_20.index, data_99_20["swe(mm)"], label="Observed SWE", color="red")
ax.plot(data_99_20.index, SWE_mod[data.index.year.isin([1999, 2000,2001,2002,2003,2004,2005,2006,2007,2008,
                                                        2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])],
        label="Simulated SWE", color="blue")
ax.set_title("Observed vs. simulated SWE for 1999 - 2020 (years with observed data)")
ax.legend()
plt.show()

'''
# Plot the hydrological years to compare observed and simulated SWE
# define hydrological years (start, ende, title)
hydro_years = []
for year in range(1999, 2021):  # from 1999 until 2020
    start_date = f"{year - 1}-10-01"
    end_date = f"{year}-09-30"
    title_text = f"Hydrological Year {year} ({start_date} until {end_date})"
    hydro_years.append((start_date, end_date, title_text))

for start_date, end_date, title_text in hydro_years:
    # 1. Filter data for the exact hydrological year
    df_sub = data.loc[start_date:end_date]
    mask_sub = (data.index >= start_date) & (data.index <= end_date)

    # 2. three panels
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Top: SWE (observed vs. modeled)
    ax1.plot(df_sub.index, df_sub["swe(mm)"], label="Observed SWE", color="black", alpha=0.7, linewidth=1.5)
    ax1.plot(df_sub.index, SWE_mod[mask_sub], label="Simulated SWE", color="blue", linewidth=1.5)
    ax1.set_ylabel("SWE [mm]", fontsize=11)
    ax1.set_title(title_text, fontsize=15, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Middle: precipitation
    ax2.bar(df_sub.index, df_sub["precipitation(mm/d)"], color="darkblue", alpha=0.6, label="Precipitation")
    ax2.set_ylabel("Precip [mm/day]", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # bottom: temperature
    ax3.plot(df_sub.index, df_sub["temperature_mean(degC)"], color="firebrick", label="Mean Temperature", linewidth=1.2)
    ax3.axhline(0, color="gray", linestyle=":", alpha=0.7, label="0°C Threshold")
    ax3.set_xlabel("Date", fontsize=11)
    ax3.set_ylabel("Temp [°C]", fontsize=11)
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
'''