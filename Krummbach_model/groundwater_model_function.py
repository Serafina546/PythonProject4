import numpy as np


def run_groundwater_model(
        Percolation,
        k_gw=0.01,
        alpha=1.0,
        loss_fraction=0.0,
        S_gw0=50.0):
    """
    Gesättigte Zone (Grundwasserspeicher): Berechnet Basisabfluss (Baseflow)
    und optionale Tiefenversickerung (Deep Percolation Loss).

    Parameter:
    - Percolation : np.array -> Zufluss aus der Bodenzone (mm/d)
    - k_gw : float -> Rezessionskoeffizient des Grundwassers
    - alpha : float -> Nicht-Linearität (1.0 = linearer Speicher, >1.0 = nicht-linear)
    - loss_fraction : float -> Anteil (0.0 bis 1.0), der in tiefe Aquifere verloren geht
    - S_gw0 : float -> Anfangsfüllstand des Grundwassers (mm)
    """
    n_steps = len(Percolation)
    S_gw = np.zeros(n_steps)
    Q_baseflow = np.zeros(n_steps)
    GW_loss = np.zeros(n_steps)

    S_gw_prev = S_gw0

    for t in range(n_steps):
        perc = 0.0 if np.isnan(Percolation[t]) else Percolation[t]

        # 1. Grundwasserneubildung (Recharge) abzüglich Tiefenverlust
        GW_loss[t] = perc * loss_fraction
        recharge = perc * (1.0 - loss_fraction)

        S_available = S_gw_prev + recharge

        # 2. Basisabfluss (Baseflow) - linear (alpha=1) oder nicht-linear (alpha>1)
        q_pot = k_gw * (S_available ** alpha)
        Q_baseflow[t] = min(q_pot, S_available)  # Kann nicht mehr auslaufen als vorhanden
        S_available -= Q_baseflow[t]

        # 3. Speicher-Update
        S_gw[t] = max(0.0, S_available)
        S_gw_prev = S_gw[t]

    return Q_baseflow, S_gw, GW_loss