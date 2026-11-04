import numpy as np


def run_flowerpot_model(P_eff, PET_remaining, k=0.1, S0=0.0):
    """
    Lineares Speichermodell (Flowerpot) für den Bodenwasserhaushalt.
    """
    n_steps = len(P_eff)
    S = np.zeros(n_steps)
    Q = np.zeros(n_steps)
    ET_soil = np.zeros(n_steps)

    S_prev = S0

    for t in range(n_steps):
        p_eff = 0.0 if np.isnan(P_eff[t]) else P_eff[t]
        pet_rem = 0.0 if np.isnan(PET_remaining[t]) else PET_remaining[t]

        # Abfluss aus dem Speicher zu Beginn des Zeitschritts
        Q[t] = k * S_prev

        # Verfügbarer Speicher nach Abfluss und effektivem Niederschlag
        S_available = max(0.0, S_prev - Q[t] + p_eff)

        # Tatsächliche Verdunstung aus dem Speicher
        ET_soil[t] = min(pet_rem, S_available)

        # Neuer Speicherstand
        S[t] = S_available - ET_soil[t]
        S_prev = S[t]

    return Q, S, ET_soil