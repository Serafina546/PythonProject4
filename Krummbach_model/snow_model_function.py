import numpy as np


def run_snow_model(P, T, T_min=1.0, T_max=3.0, m_f=2.0, T_T=0.0, SWE0=0.0):
    """
    P : np.array -> Precipitation (mm/d)
    T : np.array -> Mean daily temperature (°C)
    T_min : float -> Temperature below which 100% snow falls
    T_max : float -> Temperature above which 100% rain falls
    m_f : float -> Degree-day melt factor [mm/(°C * day)]
    T_T : float -> Threshold temperature for melt
    SWE0 : float -> Initial SWE at the start

    Returns:

    SWE_mod : np.array -> Simulated snow water equivalent
    Outflow : np.array -> Total outflow from the snowpack (rain + melt)
    P_snow  : np.array -> Snow fraction of precipitation
    P_rain  : np.array -> Rain fraction of precipitation
    Melt    : np.array -> Daily melt
    """
    n = len(P)

    # Arrays for Outputs
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

        # Update for time step t+1
        SWE_prev = SWE_mod[t]

    return SWE_mod, Outflow, P_snow, P_rain, Melt