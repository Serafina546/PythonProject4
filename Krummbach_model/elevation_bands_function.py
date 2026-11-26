import numpy as np
from snow_model_function import run_snow_model
from Glacial_melt_function import glacier_melt

def adjust_temperature(T, z_ref, z_band, lapse=-0.0065):
    """
    Temperatur von der Referenzhöhe auf die Höhe eines Bandes umrechnen.
    T       : Temperatur-Zeitreihe auf Referenzhöhe (°C)
    z_ref   : Referenzhöhe (m ü. M.), i.d.R. mittlere Höhe des Einzugsgebiets
    z_band  : mittlere Höhe des Bandes (m ü. M.)
    lapse   : Temperaturgradient (°C/m), Standard -0.0065 = -0.65 °C / 100 m
    """
    T = np.asarray(T, dtype=float)
    return T + lapse * (z_band - z_ref)

def precipitation_factors(bands, z_ref, p_grad=0.0005):
    """
    Niederschlagsfaktoren pro Band berechnen.
    p_grad : relative Zunahme pro Meter (0.0005 = +5 % pro 100 m)
    Die Faktoren werden so normiert, dass das flächengewichtete Mittel = 1 ist.
    So bleibt der Gebietsniederschlag (und damit die Wasserbilanz) unverändert;
    es wird nur umverteilt: oben mehr, unten weniger.
    """
    z = np.array([b["z"] for b in bands], dtype=float)
    area = np.array([b["area"] for b in bands], dtype=float)
    f = 1.0 + p_grad * (z - z_ref)
    f = np.maximum(f, 0.0)                       # keine negativen Faktoren
    f = f / np.sum(f * area / area.sum())        # Normierung auf Gebietsmittel = 1
    return f

def check_bands(bands, z_ref=None, tol=50.0):
    """Einfache Plausibilitätsprüfung der Band-Definition."""
    area = np.array([b["area"] for b in bands], dtype=float)
    g_area = np.array([b.get("glacier_area", 0.0) for b in bands], dtype=float)
    z = np.array([b["z"] for b in bands], dtype=float)

    if np.any(area <= 0):
        raise ValueError("Jedes Band braucht eine positive Fläche.")
    if np.any(g_area < 0) or np.any(g_area > area):
        raise ValueError("Gletscherfläche pro Band muss zwischen 0 und Bandfläche liegen.")

    z_mean = np.sum(z * area) / area.sum()
    if z_ref is not None and abs(z_mean - z_ref) > tol:
        print(f"Warnung: flächengewichtete Bandhöhe ({z_mean:.0f} m) weicht von "
              f"z_ref ({z_ref:.0f} m) ab. Die Gebietsmitteltemperatur bleibt dann nicht erhalten.")
    return z_mean

def run_elevation_bands(P, P_net, T, bands, z_ref, snow_params, ddf_ice=7.0, T0=0.0,
                        lapse=-0.0065, p_grad=0.0005, initial_states=None):
    """
    Schnee- und Gletschermodell pro Höhenband rechnen und flächengewichtet aufsummieren.
    Input
    -----
    P              : Niederschlag (Gebietsmittel, ohne Interzeption) -> für Gletscher
    P_net          : Nettoniederschlag nach Interzeption (Gebietsmittel) -> für Schneemodell
    T              : Temperatur (Gebietsmittel, gilt auf Höhe z_ref)
    bands          : Liste von Dicts, z.B. {"z": 2600, "area": 20.0, "glacier_area": 0.5}
                     z = mittlere Bandhöhe (m), area / glacier_area in km²
    z_ref          : Referenzhöhe der Eingangsdaten (mittlere Höhe EZG)
    snow_params    : Schneeparameter OHNE SWE0 (z.B. T_snow, T_rain, m_f, T_T)
    initial_states : None oder Dict mit Listen "SWE" und "SWE_glacier" (ein Wert pro Band)
    Output (Dict)
    -------------
    Outflow            : Schnee-/Regen-Outflow, auf EZG bezogen (mm/d)
    Icemelt_catchment  : Eisschmelze, auf EZG bezogen (mm/d)
    SWE                : flächengewichtete Schneedecke im EZG (mm)
    T_bands, Outflow_bands, SWE_bands : Werte pro Band (für Plots/Kontrolle)
    final_states       : Endzustände pro Band (für die nächste Periode)
    """

    P = np.asarray(P, dtype=float)
    P_net = np.asarray(P_net, dtype=float)
    T = np.asarray(T, dtype=float)
    n = len(T)
    n_bands = len(bands)

    snow_params = {k: v for k, v in snow_params.items() if k != "SWE0"}

    if initial_states is None:
        initial_states = {"SWE": [0.0] * n_bands, "SWE_glacier": [0.0] * n_bands}

    area = np.array([b["area"] for b in bands], dtype=float)
    catchment_area = area.sum()
    p_fac = precipitation_factors(bands, z_ref, p_grad)

    Outflow = np.zeros(n)
    Icemelt_catchment = np.zeros(n)
    SWE = np.zeros(n)
    T_bands = np.zeros((n_bands, n))
    Outflow_bands = np.zeros((n_bands, n))
    SWE_bands = np.zeros((n_bands, n))
    final_states = {"SWE": [], "SWE_glacier": []}

    for j, b in enumerate(bands):
        weight = b["area"] / catchment_area
        g_area = b.get("glacier_area", 0.0)

        T_b = adjust_temperature(T, z_ref, b["z"], lapse)
        P_b = P * p_fac[j]
        P_net_b = P_net * p_fac[j]

        # Schnee im Band (ganze Bandfläche)
        SWE_b, Outflow_b, _, _, _ = run_snow_model(
            P=P_net_b, T=T_b, SWE0=initial_states["SWE"][j], **snow_params)

        Outflow += Outflow_b * weight
        SWE += SWE_b * weight

        # Gletscher im Band (nur falls vorhanden)
        if g_area > 0:
            SWE_g, _, Icemelt_c = glacier_melt(
                P=P_b, T=T_b,
                glacier_area=g_area,
                catchment_area=catchment_area,
                ddf_ice=ddf_ice, T0=T0,
                snow_params={**snow_params, "SWE0": initial_states["SWE_glacier"][j]})
            Icemelt_catchment += Icemelt_c
            final_states["SWE_glacier"].append(SWE_g[-1])
        else:
            final_states["SWE_glacier"].append(0.0)

        final_states["SWE"].append(SWE_b[-1])
        T_bands[j] = T_b
        Outflow_bands[j] = Outflow_b
        SWE_bands[j] = SWE_b

    return {
        "Outflow": Outflow,
        "Icemelt_catchment": Icemelt_catchment,
        "SWE": SWE,
        "T_bands": T_bands,
        "Outflow_bands": Outflow_bands,
        "SWE_bands": SWE_bands,
        "final_states": final_states,
    }