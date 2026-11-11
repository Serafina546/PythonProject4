import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from snow_model_function import run_snow_model
from PET_model_function import calculate_hargreaves_pet
from interception_model_function import run_interception_model
from flowerpot_model_function import run_flowerpot_model

# ---------------------------------------------------------------------------
# 1. Daten einlesen
# ---------------------------------------------------------------------------
data_sim = pd.read_csv(r"../data/CAMELS_CH_sim_based_2244.csv", parse_dates=["date"]).set_index("date")
data_obs = pd.read_csv(r"../data/CAMELS_CH_obs_based_2244.csv", parse_dates=["date"]).set_index("date")
data = data_obs.join(data_sim, how="inner", suffixes=("_obs", "_sim"))

# Achtung: fehlende Niederschlagswerte werden = 0 gesetzt (entspricht evtl. nicht der Realität)
data["precipitation(mm/d)"] = data["precipitation(mm/d)"].fillna(0.0)

LATITUDE = 46.230046
Q_COL = "discharge_spec(mm/d)"

# Arrays einmalig extrahieren
dates = data.index
P = data["precipitation(mm/d)"].to_numpy(dtype=float)
T_mean = data["temperature_mean(degC)"].to_numpy(dtype=float)
T_min = data["temperature_min(degC)"].to_numpy(dtype=float)
T_max = data["temperature_max(degC)"].to_numpy(dtype=float)
Q_obs = data[Q_COL].to_numpy(dtype=float)

# PET hängt nicht von den Parametern ab -> einmal berechnen
PET = calculate_hargreaves_pet(
    dates=dates, T_mean=T_mean, T_min=T_min, T_max=T_max, latitude_deg=LATITUDE
)

# ---------------------------------------------------------------------------
# 2. Zeiträume als Masken
# ---------------------------------------------------------------------------
calib_mask = (dates >= "1983-01-01") & (dates <= "2009-12-31")
val_mask = (dates >= "2010-01-01") & (dates <= "2020-12-31")
# Warm-up 1981-1982: Das Modell läuft ab 1981, ausgewertet wird erst ab 1983.
# Für die Kalibrierung genügt es, bis Ende 2009 zu simulieren.
calib_end = np.where(calib_mask)[0][-1] + 1


# ---------------------------------------------------------------------------
# 3. Gütemaße
# ---------------------------------------------------------------------------
def compute_nse(q_obs, q_sim):
    mask = ~np.isnan(q_obs) & ~np.isnan(q_sim)
    q_obs, q_sim = q_obs[mask], q_sim[mask]
    return 1 - np.sum((q_obs - q_sim) ** 2) / np.sum((q_obs - np.mean(q_obs)) ** 2)


def compute_log_nse(q_obs, q_sim, eps=1e-3):
    mask = ~np.isnan(q_obs) & ~np.isnan(q_sim)
    q_obs, q_sim = np.log(q_obs[mask] + eps), np.log(q_sim[mask] + eps)
    return 1 - np.sum((q_obs - q_sim) ** 2) / np.sum((q_obs - np.mean(q_obs)) ** 2)


def compute_bias(q_obs, q_sim):
    mask = ~np.isnan(q_obs) & ~np.isnan(q_sim)
    q_obs, q_sim = q_obs[mask], q_sim[mask]
    return np.sum(q_sim - q_obs) / np.sum(q_obs) * 100  # Prozent


def evaluate(q_obs, q_sim):
    return {
        "NSE": compute_nse(q_obs, q_sim),
        "Log-NSE": compute_log_nse(q_obs, q_sim),
        "Bias [%]": compute_bias(q_obs, q_sim),
    }


# ---------------------------------------------------------------------------
# 4. Modellkette (lumped)
# ---------------------------------------------------------------------------
def run_model(params, n_steps=None, initial_states=None):
    """Lässt die gesamte Modellkette bis zum Zeitschritt n_steps laufen."""
    n = len(P) if n_steps is None else n_steps

    SWE0 = 0.0 if initial_states is None else initial_states["SWE"]
    S0 = 0.0 if initial_states is None else initial_states["S"]

    T_min_snow, T_max_snow, m_f, icept_threshold, k = params

    SWE, Outflow, P_snow, P_rain, Melt = run_snow_model(
        P=P[:n], T=T_mean[:n],
        T_min=T_min_snow, T_max=T_max_snow,
        m_f=m_f, T_T=0.0, SWE0=SWE0,
    )

    P_net, Evap_icept, PET_transpiration, Unloading = run_interception_model(
        P=Outflow, PET=PET[:n], icept_threshold=icept_threshold,
    )

    Q_modelled, S, ET_soil = run_flowerpot_model(
        P_eff=P_net, PET_remaining=PET_transpiration, k=k, S0=S0,
    )

    final_states = {"SWE": SWE[-1], "S": S[-1]}
    return np.asarray(Q_modelled), final_states


# ---------------------------------------------------------------------------
# 5. Zielfunktion
# ---------------------------------------------------------------------------
def objective_function(params, objective="nse"):
    Q_sim, _ = run_model(params, n_steps=calib_end)
    q_o = Q_obs[:calib_end][calib_mask[:calib_end]]
    q_s = Q_sim[calib_mask[:calib_end]]

    if not np.all(np.isfinite(q_s)):
        return 1e6

    if objective == "nse":
        return 1 - compute_nse(q_o, q_s)
    if objective == "log_nse":
        return 1 - compute_log_nse(q_o, q_s)
    if objective == "combined":  # Hoch- und Niedrigwasser, Bias bestraft
        return (1 - compute_nse(q_o, q_s)) + (1 - compute_log_nse(q_o, q_s)) \
               + abs(compute_bias(q_o, q_s)) / 100
    raise ValueError(objective)


def obj_nse(p):
    return objective_function(p, "nse")


def obj_log_nse(p):
    return objective_function(p, "log_nse")


def obj_combined(p):
    return objective_function(p, "combined")


# ---------------------------------------------------------------------------
# 6. Kalibrierung + Validierung
# ---------------------------------------------------------------------------
# Parameter: [T_min_snow, T_max_snow, m_f, icept_threshold, k]
bounds = [(0.0, 2.0), (2.0, 5.0), (1.0, 6.0), (0.0, 2.0), (0.001, 0.5)]
# T_min < T_max ist durch die Bounds (max 2.0 vs. min 2.0) immer erfüllt.

if __name__ == "__main__":  # nötig für workers=-1 (v.a. unter Windows)
    objectives = {"NSE": obj_nse, "Log-NSE": obj_log_nse, "Combined": obj_combined}
    results = {}

    for name, func in objectives.items():
        print(f"\n=== Kalibrierung mit Zielfunktion: {name} ===")
        res = differential_evolution(
            func, bounds, seed=42, maxiter=100, popsize=15,
            tol=1e-6, workers=-1, updating="deferred", disp=False,
        )
        print("Erfolg:", res.success, "|", res.message)
        print("Beste Parameter [T_min, T_max, m_f, icept, k]:", np.round(res.x, 4))

        # Gesamtsimulation 1981-2020 mit den besten Parametern
        Q_sim, _ = run_model(res.x)
        results[name] = (res.x, Q_sim)

        print("Kalibrierung:", {k: round(v, 3) for k, v in
                                evaluate(Q_obs[calib_mask], Q_sim[calib_mask]).items()})
        print("Validierung: ", {k: round(v, 3) for k, v in
                                evaluate(Q_obs[val_mask], Q_sim[val_mask]).items()})

    # -----------------------------------------------------------------------
    # 7. Plot (Validierungszeitraum, Combined-Kalibrierung)
    # -----------------------------------------------------------------------
    _, Q_best = results["Combined"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates[val_mask], Q_obs[val_mask], label="Beobachtet", color="k", lw=0.8)
    ax.plot(dates[val_mask], Q_best[val_mask], label="Simuliert", color="tab:blue", lw=0.8)
    ax.set_ylabel("Q [mm/d]")
    ax.set_title("Validierung 2010-2020")
    ax.legend()
    plt.tight_layout()
    plt.show()