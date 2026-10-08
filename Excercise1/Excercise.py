#1a
import numpy as np
import matplotlib.pyplot as plt

# Parameter
n_steps = 50      # Anzahl Zeitschritte
k = 0.1           # Speicherkonstante [1/Zeitschritt]
S0 = 0.0          # Anfangsspeicher [mm]

# Input: 10 mm in den ersten 10 Zeitschritten
P = np.zeros(n_steps)
P[:10] = 10.0

# Arrays für Speicher und Abfluss
S = np.zeros(n_steps)
Q = np.zeros(n_steps)

S_prev = S0
for t in range(n_steps):
    Q[t] = k * S_prev              # Abfluss aus altem Speicher
    S[t] = S_prev + P[t] - Q[t]    # Wasserbilanz
    S_prev = S[t]

# Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax[0].bar(range(n_steps), P, color="steelblue")
ax[0].invert_yaxis()
ax[0].set_ylabel("Precipitation [mm/day]")
ax[0].set_title("Excercise 2b: Linear storage Model")

ax[1].plot(Q, label="Discharge Q", color="darkred")
ax[1].plot(S, label="Storage S", color="green", linestyle="--")
ax[1].set_xlabel("Time step/day")
ax[1].set_ylabel("Precipitation [mm/]")
ax[1].legend()

plt.tight_layout()
plt.show()


#1b
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

n_steps = 30
P = np.zeros(n_steps)
P[:10] = 10.0

def cascade(P, k1, k2, S1_0=0.0, S2_0=0.0):
    n = len(P)
    S1 = np.zeros(n); S2 = np.zeros(n)
    Q1 = np.zeros(n); Q2 = np.zeros(n)
    s1, s2 = S1_0, S2_0
    for t in range(n):
        Q1[t] = k1 * s1
        S1[t] = s1 + P[t] - Q1[t]
        Q2[t] = k2 * s2
        S2[t] = s2 + Q1[t] - Q2[t]
        s1, s2 = S1[t], S2[t]
    return Q1, Q2, S1, S2

k_values = [0.1, 0.4, 0.7]
Qmax = np.zeros((3, 3))
Qmax_time = np.zeros((3, 3), dtype=int)

fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey=True)

for (i, k1), (j, k2) in product(enumerate(k_values), enumerate(k_values)):
    Q1, Q2, S1, S2 = cascade(P, k1, k2)
    Qmax[i, j] = Q2.max()
    Qmax_time[i, j] = Q2.argmax()

    # Wasserbilanz prüfen
    assert np.isclose(P.sum(), Q2.sum() + S1[-1] + S2[-1])

    ax = axes[i, j]
    ax.plot(Q1, color="gray", linestyle="--", label="Q1")
    ax.plot(Q2, color="darkred", label="Q2")
    ax.set_title(f"k1={k1}, k2={k2}", fontsize=9)
    if i == 2: ax.set_xlabel("Zeitschritt")
    if j == 0: ax.set_ylabel("Q [mm/Zeitschritt]")
axes[0, 0].legend()
plt.tight_layout()
plt.show()

# Ergebnistabelle
print("Maximaler Abfluss Q2 [mm/Zeitschritt] (Zeile: k1, Spalte: k2)")
print("        " + "".join(f"k2={k:<8}" for k in k_values))
for i, k1 in enumerate(k_values):
    print(f"k1={k1}  " + "".join(f"{Qmax[i, j]:<11.3f}" for j in range(3)))

print("\nZeitschritt des Maximums")
for i, k1 in enumerate(k_values):
    print(f"k1={k1}  " + "".join(f"{Qmax_time[i, j]:<11d}" for j in range(3)))

