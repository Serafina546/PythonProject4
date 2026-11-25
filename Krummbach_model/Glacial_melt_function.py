import numpy as np
from snow_model_function import run_snow_model


def glacier_melt(P, T, glacier_area, catchment_area,
                 ddf_ice=7.0, T0=0.0, snow_params=None):
    """
    Eisschmelze mit Grad-Tag-Ansatz. Die Schneedecke auf dem Gletscher wird
    mit dem gleichen Schneemodell wie im Einzugsgebiet berechnet.
    T muss bereits für die Höhe des Gletschers gelten.

    Output: SWE_glacier, ice_melt (mm/d auf Gletscherfläche),
            ice_melt_catchment (mm/d auf Einzugsgebiet bezogen)
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)

    SWE_glacier, _, _, _, _ = run_snow_model(P=P, T=T, **snow_params)

    ice_melt = np.where(SWE_glacier <= 0, ddf_ice * np.maximum(T - T0, 0.0), 0.0)
    ice_melt_catchment = ice_melt * glacier_area / catchment_area

    return SWE_glacier, ice_melt, ice_melt_catchment