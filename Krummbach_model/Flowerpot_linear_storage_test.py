# pet_interception.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_pet_interception(
    latitude_deg=46.230046,
    icept_threshold=0.25,
    k=0.1,
    S0=0.0,
    plot=True
):
    plt.rcParams["font.family"] = "Arial"

    # CSV-Datei einlesen
    script_dir = Path(__file__).resolve().parent
    csv_path = (
        script_dir
        / "Precipitation_data"
        / "Camels_Precipitation"
        / "CAMELS_CH_obs_based_2244.csv"
    )
    data = pd.read_csv(csv_path)

    # Niederschlag
    P = data["precipitation(mm/d)"].to_numpy()
    P = np.nan_to_num(P, nan=0.0)
    n_steps = len(P)

    # PET-Berechnung
    data["date"] = pd.to_datetime(data["date"])

    T_mean = data["temperature_mean(degC)"].to_numpy(dtype=float)
    T_min = data["temperature_min(degC)"].to_numpy(dtype=float)
    T_max = data["temperature_max(degC)"].to_numpy(dtype=float)

    J = data["date"].dt.day_of_year.to_numpy()
    phi = np.radians(latitude_deg)

    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)

    ws = np.arccos(
        np.clip(-np.tan(phi) * np.tan(delta), -1, 1)
    )

    Gsc = 0.0820
    Ra = (
        (24 * 60 / np.pi)
        * Gsc
        * dr
        * (
            ws * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(ws)
        )
    )

    temperature_range = np.maximum(T_max - T_min, 0)

    PET = (
        0.0023
        * (T_mean + 17.8)
        * np.sqrt(temperature_range)
        * 0.408
        * Ra
    )

    PET = np.maximum(PET, 0)

    invalid = (
        ~np.isfinite(T_mean)
        | ~np.isfinite(T_min)
        | ~np.isfinite(T_max)
    )
    PET[invalid] = np.nan

    # Interception
    Icept = np.minimum(P, icept_threshold)
    Evap_icept = np.minimum(PET, Icept)
    Unloading = Icept - Evap_icept
    P_eff = P - Evap_icept

    # Speichermodell
    S = np.zeros(n_steps)
    Q = np.zeros(n_steps)
    ET_soil = np.zeros(n_steps)
    S_prev = S0

    for t in range(n_steps):
        Q[t] = k * S_prev

        S_available = max(
            0.0,
            S_prev - Q[t] + P_eff[t]
        )

        PET_remaining = max(
            0.0,
            PET[t] - Evap_icept[t]
        )

        ET_soil[t] = min(PET_remaining, S_available)

        S[t] = S_available - ET_soil[t]
        S_prev = S[t]

    # Beobachteter Abfluss
    Q_obs = data["discharge_spec(mm/d)"].to_numpy(dtype=float)
    dates = data["date"]

    # Plot
    if plot:
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            dates, Q,
            label="Modelled discharge",
            linewidth=0.7,
            color="green"
        )
        ax.plot(
            dates, Q_obs,
            label="Observed discharge",
            linewidth=0.7,
            color="black"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Discharge [mm/day]")
        ax.set_title("Modelled vs Observed Discharge")
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        ax2 = ax.twinx()
        ax2.bar(
            dates, P,
            width=1.0,
            color="darkblue",
            alpha=0.3,
            label="Precipitation"
        )
        ax2.set_ylabel("Precipitation [mm/day]")
        ax2.set_ylim(max(1, np.nanmax(P) * 1.2), 0)

        fig.tight_layout()
        plt.show()

    # Ergebnisse zurückgeben
    return {
        "PET": PET,
        "P": P,
        "Icept": Icept,
        "Evap_icept": Evap_icept,
        "Unloading": Unloading,
        "P_eff": P_eff,
        "Q": Q,
        "Q_obs": Q_obs,
        "S": S,
        "ET_soil": ET_soil,
        "dates": dates
    }