# snow_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def run_snow_model():
    script_dir = Path(__file__).resolve().parent

    data_sim = pd.read_csv(
        script_dir / "data" / "CAMELS_CH_sim_based_2244.csv",
        parse_dates=["date"]
    )

    data_obs = pd.read_csv(
        script_dir / "data" / "CAMELS_CH_obs_based_2244.csv",
        parse_dates=["date"]
    )

    data_obs = data_obs.set_index("date")
    data_sim = data_sim.set_index("date")

    data = data_obs.join(
        data_sim, how="inner", suffixes=("_obs", "_sim")
    )

    dates = data.index
    P = data["precipitation(mm/d)"].to_numpy()
    T = data["temperature_mean(degC)"].to_numpy()
    SWE_obs = data["swe(mm)"].to_numpy()

    n = len(data)

    # Parameter
    SWE0 = 0.0
    T_min = 1.0
    T_max = 3.0
    m_f = 2.0
    T_T = 0.0

    # Output arrays
    SWE_mod = np.zeros(n)
    P_snow = np.zeros(n)
    P_rain = np.zeros(n)
    Melt = np.zeros(n)
    Outflow = np.zeros(n)

    SWE_prev = SWE0

    for t in range(n):
        # Partitioning von Schnee und Regen
        if T[t] <= T_min:
            snow_fraction = 1.0
        elif T[t] >= T_max:
            snow_fraction = 0.0
        else:
            snow_fraction = (
                (T_max - T[t]) / (T_max - T_min)
            )

        P_snow[t] = P[t] * snow_fraction
        P_rain[t] = P[t] * (1.0 - snow_fraction)

        # Potenzielle Schneeschmelze
        if T[t] > T_T:
            potential_melt = m_f * (T[t] - T_T)
        else:
            potential_melt = 0.0

        # SWE-Bilanz
        available_snow = SWE_prev + P_snow[t]
        Melt[t] = min(potential_melt, available_snow)
        SWE_mod[t] = available_snow - Melt[t]

        # Regen + Schmelzwasser
        Outflow[t] = P_rain[t] + Melt[t]

        SWE_prev = SWE_mod[t]

    return {
        "data": data,
        "dates": dates,
        "P": P,
        "T": T,
        "SWE_obs": SWE_obs,
        "SWE_mod": SWE_mod,
        "P_snow": P_snow,
        "P_rain": P_rain,
        "Melt": Melt,
        "Outflow": Outflow
    }


def plot_snow_results(results):
    data = results["data"]
    dates = results["dates"]
    SWE_mod = results["SWE_mod"]

    # Vergleich 1999–2020
    mask = (dates >= "1999-01-01") & (dates <= "2020-12-31")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        dates[mask],
        results["SWE_obs"][mask],
        label="Observed SWE",
        color="red"
    )

    ax.plot(
        dates[mask],
        SWE_mod[mask],
        label="Simulated SWE",
        color="blue"
    )

    ax.set_title("Observed vs. simulated SWE (1999–2020)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Hydrologische Jahre
    for year in range(1999, 2021):
        start_date = f"{year - 1}-10-01"
        end_date = f"{year}-09-30"

        df_sub = data.loc[start_date:end_date]
        mask_sub = (
            (dates >= start_date) & (dates <= end_date)
        )

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 8), sharex=True
        )

        ax1.plot(
            df_sub.index,
            df_sub["swe(mm)"],
            label="Observed SWE",
            color="black"
        )
        ax1.plot(
            dates[mask_sub],
            SWE_mod[mask_sub],
            label="Simulated SWE",
            color="blue"
        )
        ax1.set_ylabel("SWE [mm]")
        ax1.set_title(f"Hydrological Year {year}")
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        ax2.bar(
            df_sub.index,
            df_sub["precipitation(mm/d)"],
            color="darkblue",
            alpha=0.6
        )
        ax2.set_ylabel("Precip [mm/day]")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        ax3.plot(
            df_sub.index,
            df_sub["temperature_mean(degC)"],
            color="firebrick"
        )
        ax3.axhline(0, color="gray", linestyle=":")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Temp [°C]")
        ax3.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()