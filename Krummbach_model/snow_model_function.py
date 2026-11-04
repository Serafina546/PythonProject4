import numpy as np


def run_snow_model(P, T, T_min=1.0, T_max=3.0, m_f=2.0, T_T=0.0, SWE0=0.0):
    """
    Parameters:

    P : np.array -> Niederschlag (mm/d)
    T : np.array -> Mittlere Tagestemperatur (°C)
    T_min : float -> Temperatur unter der 100% Schnee fällt
    T_max : float -> Temperatur über der 100% Regen fällt
    m_f : float -> Gradtag-Schmelzfaktor [mm/(°C * day)]
    T_T : float -> Schwellenwertemperatur für Schmelze
    SWE0 : float -> Initialer SWE zu Beginn

    Returns:

    SWE_mod : np.array -> Simulierter Schneewasseräquivalent
    Outflow : np.array -> Gesamter Abfluss aus dem Schneepaket (Regen + Schmelze)
    P_snow  : np.array -> Schneeanteil des Niederschlags
    P_rain  : np.array -> Regenanteil des Niederschlags
    Melt    : np.array -> Tägliche Schmelze
    """
    n = len(P)

    # Arrays für Outputs initialisieren
    SWE_mod = np.zeros(n)
    P_snow = np.zeros(n)
    P_rain = np.zeros(n)
    Melt = np.zeros(n)
    Outflow = np.zeros(n)

    SWE_prev = SWE0

    for t in range(n):
        # 1. Rain/Snow Separation
        if T[t] <= T_min:
            snow_fraction = 1.0
        elif T[t] >= T_max:
            snow_fraction = 0.0
        else:
            snow_fraction = (T_max - T[t]) / (T_max - T_min)

        P_snow[t] = P[t] * snow_fraction
        P_rain[t] = P[t] * (1.0 - snow_fraction)

        # 2. Potential Melt (Degree-Day Method)
        if T[t] > T_T:
            potential_melt = m_f * (T[t] - T_T)
        else:
            potential_melt = 0.0

        # 3. SWE Water Balance & Melt
        available_snow = SWE_prev + P_snow[t]
        Melt[t] = min(potential_melt, available_snow)
        SWE_mod[t] = available_snow - Melt[t]

        # 4. Total Outflow (Rainfall + Meltwater)
        Outflow[t] = P_rain[t] + Melt[t]

        # Update für Zeitschritt t+1
        SWE_prev = SWE_mod[t]

    return SWE_mod, Outflow, P_snow, P_rain, Melt