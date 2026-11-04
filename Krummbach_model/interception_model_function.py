import numpy as np


def run_interception_model(P, PET, icept_threshold=1.5):
    n = len(P)
    Icept = np.zeros(n)
    Evap_icept = np.zeros(n)
    Unloading = np.zeros(n)
    PET_transpiration = np.zeros(n)
    P_net = np.zeros(n)

    for t in range(n):
        p_val = 0.0 if np.isnan(P[t]) else P[t]
        pet_val = 0.0 if np.isnan(PET[t]) else PET[t]

        # Interzeption-threshold
        if p_val <= icept_threshold:
            Icept[t] = p_val
        else:
            Icept[t] = icept_threshold

        # evaporation from intercepted water
        Evap_icept[t] = min(pet_val, Icept[t])

        # Unloading
        Unloading[t] = Icept[t] - Evap_icept[t]

        # Reduce PET available for transpiration
        PET_transpiration[t] = max(0.0, pet_val - Evap_icept[t])

        # Precipitation that reaches the ground
        P_net[t] = p_val - Evap_icept[t]

    return P_net, Evap_icept, PET_transpiration, Unloading