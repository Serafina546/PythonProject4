import numpy as np


def calculate_hargreaves_pet(dates, T_mean, T_min, T_max, latitude_deg=46.230046):
    phi = np.radians(latitude_deg)
    J = dates.day_of_year.to_numpy()

    # Extraterrestrische Strahlung berechnen
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    Gsc = 0.0820  # MJ m-2 min-1

    Ra = ((24 * 60 / np.pi) * Gsc * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)))

    # Hargreaves-Samani: PET in mm/day
    temperature_range = np.maximum(T_max - T_min, 0)
    PET = 0.0023 * (T_mean + 17.8) * np.sqrt(temperature_range) * 0.408 * Ra

    # Negative Werte vermeiden
    PET = np.maximum(PET, 0)

    # Fehlende Temperaturwerte behandeln
    invalid = (~np.isfinite(T_mean) | ~np.isfinite(T_min) | ~np.isfinite(T_max))
    PET[invalid] = np.nan

    return PET