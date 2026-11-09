import numpy as np

def calculate_quick_runoff(p_eff, S_prev, S_max, beta=1.5):
    """
    Berechnet den schnellen Abfluss (Quick Runoff) und den verbleibenden
    Infiltrationsanteil für einen Zeitschritt.

    Parameter:
    - p_eff : float -> Effektiver Niederschlag im aktuellen Zeitschritt (mm/d)
    - S_prev : float -> Vorheriger Speicherstand im Boden (mm)
    - S_max : float -> Maximale Speicherkapazität (mm)
    - beta : float -> Formparameter für den Sättigungsüberfluss

    Returns:
    - p_fast : float -> Schneller Oberflächenabfluss (mm/d)
    - p_infil : float -> Wasser, das in den Boden infiltriert (mm/d)
    """
    # 1. Aktuellen Sättigungsgrad des Bodens berechnen (0.0 bis 1.0)
    if S_max > 0:
        saturation = min(1.0, max(0.0, S_prev / S_max))
    else:
        saturation = 0.0

    # 2. Anteil des Quick Runoffs bestimmen (nicht-linear über Beta)
    q_fast_fraction = saturation ** beta

    # 3. Aufteilung
    p_fast = p_eff * q_fast_fraction
    p_infil = p_eff * (1.0 - q_fast_fraction)

    return p_fast, p_infil