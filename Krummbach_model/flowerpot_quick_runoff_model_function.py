import numpy as np
from quick_runoff_model_function import calculate_quick_runoff


def run_flowerpot_model(P_eff, PET_remaining, k=0.1, perc_max=2.0, S_max=150.0, beta=1.5,S0=0.0):
    """
    Lineares Speichermodell (Flowerpot) für den Bodenwasserhaushalt mit Quick Runoff und Groundwater
    """
    n_steps = len(P_eff)
    S = np.zeros(n_steps)
    Q_fast = np.zeros(n_steps)
    Q_interflow = np.zeros(n_steps)
    Percolation = np.zeros(n_steps)
    ET_soil = np.zeros(n_steps)

    S_prev = S0

    for t in range(n_steps):
        p_eff = 0.0 if np.isnan(P_eff[t]) else P_eff[t]
        pet_rem = 0.0 if np.isnan(PET_remaining[t]) else PET_remaining[t]

        # Quick runoff und infiltration
        p_fast, p_infil = calculate_quick_runoff(
            p_eff=p_eff,
            S_prev=S_prev,
            S_max=S_max,
            beta=beta)

        # Zufuhr zum Bodenspeicher (Bodenspeicher auffüllen und Überlauf abfangen)
        S_available = S_prev + p_infil
        excess = 0.0
        if S_available > S_max:
            excess = S_available - S_max
            S_available = S_max

        Q_fast[t] = p_fast + excess

        # Tatsächliche Verdunstung (abhängig von Bodenfeuchte)
        saturation = min(1.0, max(0.0, S_available / S_max)) if S_max > 0 else 0.0
        ET_soil[t] = min(pet_rem * saturation, S_available)
        S_available -= ET_soil[t]

        # Perkolation (Grundwasserneubildung)
        # Abhängig von Bodenfeuchte, begrenzt durch maximale Versickerungsrate
        pot_perc = perc_max * saturation
        Percolation[t] = min(pot_perc, S_available)
        S_available -= Q_interflow[t]

        # Neuen Speicherstand sichern
        S[t] = max(0.0, S_available)
        S_prev = S[t]

        # Gesamtabfluss
        Q_total[t] = Q_slow[t] + Q_fast[t]

    return Q_fast, Q_interflow, Percolation, S, ET_soil