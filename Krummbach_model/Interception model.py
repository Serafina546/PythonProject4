### Interception Model ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Arial"

# Read data
data_obs = pd.read_csv(r"../data/CAMELS_CH_obs_based_2244.csv", parse_dates=["date"])

data_obs = data_obs.set_index("date")

'''
print(data_obs.columns)
'''

# extract variables
time = data_obs.index
P = data_obs["precipitation(mm/d)"].values
n = len(data_obs)

# get PET

# set parameter
Icept_threshold = 1.5       # interception capacity threshold [mm]

# create arrays
Icept = np.zeros(n)
Evap_icept = np.zeros(n)
Unloading = np.zeros(n)     # throughfall/dripping off
PET_transpiration = np.zeros(n)
P_net = np.zeros(n)

for t in range(n):
    # interception theshold
    if P[t] <= Icept_threshold:
        Icept[t] = P[t]
    else:
        Icept[t] = Icept_threshold

    # Evaporation of intercepted water
    Evap_icept[t] = min(PET[t], Icept[t])

    # Unloading
    Unloading[t] = Icept[t] - Evap_icept[t]

    # Reduce PET available for transpiration
    PET_transpiration[t] = PET[t] - Evap_icept[t]

    # Net precipitation reaching below the canopy
    P_net[t] = P[t] - Evap_icept[t]