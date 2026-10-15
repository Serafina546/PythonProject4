import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

### 2a

# Setting the parameters
n_steps = 60      # Number of time steps. Definition in excercise: 1 time step = 1 day.
k = 0.1           # Storage constant [1/time step]
S0 = 0.0          # Water in storage at the beginning of the run is 0 [mm]

# Precipitation input: 10 mm in 10 days/ 10 time steps
P = np.zeros(n_steps)   # creating a list with n_steps
P[:10] = 10.0           # in the first 10 time steps 10mm of precipitation

# Arrays for storage and discharge
S = np.zeros(n_steps)   # creating a list with n_steps
Q = np.zeros(n_steps)   # creating a list with n_steps
S_prev = S0             # storge of previous time step, it starts at zero

for t in range(n_steps):
    Q[t] = k * S_prev               # with each time step in the first 10 time steps, the discharge of the previous precipitation event increases by a factor of k
    S[t] = S_prev + P[t] - Q[t]     # Equation for the water balance. The storage at a given time step depends on the storage of the previous time step, the amount precipitation at the given time step, and the discharge at the given time step.
    S_prev = S[t]                   # After the loop, a new previous storage is defined.

#Plotting x-axis and y-axis in the left of the graph
fig, ax = plt.subplots()                                                                    # Creating the figure and axis
ax.plot(Q, label="Discharge (Q)", color="darkred")                                    # Plotting the values in the list of Q[t]
ax.plot(S, label="Storage (S)", color="darkgreen")                                    # Plotting the values in the list of S[t]
ax.set_xlabel("Time step [Day]", fontsize=14)                                                           # Labeling x-axis
ax.set_ylabel("Storage [mm] and discharge [mm/day]\n", fontsize=14)                                  # Labeling y-axis on the left
ax.set_title("Exercise 2a: Single linear Storage",fontsize=18, fontweight="bold")        # Set title in bold letters
ax.set_xticks(range(0, 65, 5))                                                              # Set a range for the x-ticks
ax.set_yticks(range(0, 71, 10))                                                             # Set a range for the y-ticks

#Plotting y-axis on the right, inverting
ax2 = ax.twinx()                                                        # Establishing a y-axis on the right side of the graph, a second coordinate system
ax2.bar(range(n_steps), P, color="darkblue",label="Precipitation")      # Barchart for precipitation [mm] in the first 10 days
ax2.set_ylabel("Precipitation [mm]", fontsize=14)                       # Labeling y-axis on the right
ax2.set_ylim(0,150)                                                     # Setting limits for y-axis on the right of the graph
ax2.set_yticks(range(0, 15, 10))                                        # Setting the range for the y-axis on the right of the graph
ax2.invert_yaxis()                                                      # Invert the y-axis on the right of the graph

# adding a legend to the plot containing both y-axes
lines1, labels1 = ax.get_legend_handles_labels()                        # Combines names and labels plotted on the left y-axis (Storage and Discharge) and collects them
lines2, labels2 = ax2.get_legend_handles_labels()                       # Combines the name and label plotted on the right y-axis (Precipitation) and collects it
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12)              # Because there are two independant coordinate systems, we have to combine the elements and names of both y-axes in order to put them in the same legend.
ax.grid(axis="y", linestyle="--", alpha=0.5)                            # Adding a grid for better readability
plt.show()


### 2b

# Setting the parameters
n_steps = 60      # Number of time steps. Definition in excercise: 1 time step = 1 day.
S0 = 0.0          # Water in storage at the beginning [mm]

# Precipitation input: 10 mm in the first 10 time steps
P = np.zeros(n_steps)       # creating a list with n_steps
P[:10] = 10.0               # in the first 10 time steps 10mm of precipitation

# Given values in the excercise for the storage parameters k1 and k2
k = [0.1, 0.4, 0.7]

# Dictionary for storing the maximum discharge for each combination
results = {}

# Loop over all 9 combinations of k1 and k2
for k1 in k:                                # here, 0.1, 0.4, and 0.7 are given after another into the loop for k1
    for k2 in k:                            # for each k1 value, 0.1, 0.4, and 0.7 are given after another into the loop

        # Arrays for storage and discharge in both reservoirs
        S1 = np.zeros(n_steps)              # creating an array (= a list where different values can be saved) with n_steps (=length) for S1
        S2 = np.zeros(n_steps)              # creating an array (= a list where different values can be saved) with n_steps (=length) for S2
        Q1 = np.zeros(n_steps)              # creating an array (= a list where different values can be saved) with n_steps (=length) for Q1
        Q2 = np.zeros(n_steps)              # creating an array (= a list where different values can be saved) with n_steps (=length) for Q2

        # Initial storage of both reservoirs is zero
        S1_prev = S0                        # storge of previous time step for S1, it starts at zero
        S2_prev = S0                        # storge of previous time step for S2, it starts at zero

        # Storage cascade: first, the water flows into S1 --> Q1 --> S2 --> Q2
        for t in range(n_steps):

            # First reservoir
            Q1[t] = k1 * S1_prev            # with each time step in the first 10 time steps, the discharge of the previous precipitation event increases by a factor of k
            S1[t] = S1_prev + P[t] - Q1[t]  # Equation for the water balance. The storage at a given time step depends on the storage of the previous time step, the amount precipitation at the given time step, and the discharge at the given time step.

            # Second reservoir: outflow from the first flows into here
            Q2[t] = k2 * S2_prev            # discharge in the second reservoir increases by a factor of k
            S2[t] = S2_prev + Q1[t] - Q2[t] # Equation for the water balance. The storage at a given time step depends on the storage in reservoir 2 of the previous time step, the amount discharge coming from reservoir 1, and the discharge at the given time step.

            # Updating the storage of the previous time step
            S1_prev = S1[t]         # After the loop, the previous storage is updated.
            S2_prev = S2[t]         # After the loop, the previous storage is updated.

        # Determining the maximum discharge from the second reservoir
        max_Q2 = np.max(Q2)

        # Saving the maximum discharge for this combination in the dictionary "results"
        results[(k1, k2)] = max_Q2

# Printing the results
print("Maximum discharge for each combination:")                                                            # output of the discharge for each combination
for (k1, k2), max_Q2 in results.items(): print(f"k1 = {k1}, k2 = {k2}: Maximum Q2 = {max_Q2:.2f} mm/day")   # Format: k1=..., k2=...: Maximum Q2 = ... mm/day (rounded to 2 decimal places)

# Creating barchart by using lists to associate the results
labels = []                    # creating empty list
max_discharges = []            # creating empty list
for ((k1, k2), max_Q2) in results.items():
    (labels.append(f"k1={k1}, k2={k2}"), max_discharges.append(max_Q2)) # loop for putting results into the lists. This is done for plotting the barcharts

# Plotting the maximum discharge for all combinations
fig, ax = plt.subplots()
ax.bar(labels, max_discharges, color="royalblue")                                           # Barchart
ax.set_xlabel("\n Parameter combinations of k1 and k2 values", fontsize=14)
ax.tick_params(axis="x", rotation=45)                                                       # the labels on the x-axis are rotated by 45°, otherwise you cannot read the labels due to overlap
ax.set_ylabel("Maximum discharge [mm/day]", fontsize=14)
ax.set_title("Exercise 2b: Linear Storage Cascade",fontsize=18,fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
# Result: the highest discharge is found at k1=0.7 and k2=0.7.


### 2c

# Setting the parameters
n_steps = 60      # Number of time steps. Definition in exercise: 1 time step = 1 day.
k = 0.1           # Storage constant [1/time step]
S0 = 0.0          # Water in storage at the beginning [mm]
Smax = 50.0       # Maximum storage capacity [mm] (chosen value)
ET0 = 2.0         # Potential evapotranspiration [mm/day] (given value)

# Precipitation input: 10 mm in the first 10 time steps
P = np.zeros(n_steps)   # Creating an array with n_steps
P[:10] = 10.0           # 10 mm of precipitation during the first 10 time steps

# Arrays for storage, discharge and evapotranspiration
S = np.zeros(n_steps)   # Creating an array for storage
Q = np.zeros(n_steps)   # Creating an array for discharge
ET = np.zeros(n_steps)  # Creating an array for evapotranspiration

S_prev = S0             # Storage of the previous time step starts at zero

### 2c

# Setting the parameters
n_steps = 60      # Number of time steps. Definition in exercise: 1 time step = 1 day.
k = 0.1           # Storage constant [1/time step]
S0 = 0.0          # Water in storage at the beginning [mm]
Smax = 50.0       # Maximum storage capacity [mm] (chosen value)
ET0 = 2.0         # Potential evapotranspiration [mm/day]

# Precipitation input: 10 mm in the first 10 time steps
P = np.zeros(n_steps)   # Creating an array with n_steps
P[:10] = 10.0           # 10 mm of precipitation during the first 10 time steps

# Arrays for storage, discharge and evapotranspiration
S = np.zeros(n_steps)   # Creating an array for storage
Q = np.zeros(n_steps)   # Creating an array for discharge
ET = np.zeros(n_steps)  # Creating an array for evapotranspiration

S_prev = S0             # Storage of the previous time step starts at zero

for t in range(n_steps):

    # Calculating the discharge from the previous storage
    Q[t] = k * S_prev

    # Calculating the storage before evapotranspiration
    S_temp = S_prev + P[t] - Q[t]

    # Limiting the storage to the maximum capacity
    # Excess water above Smax leaves the reservoir as overflow
    if S_temp > Smax:
        S_temp = Smax

    # Calculating evapotranspiration
    S_temp = max(0.0, S_temp)      #Storage cannot be negative (no negative roots)

    ET[t] = ET0 * (S_temp / Smax)**0.5

    #ET cannot remove more water than is available
    ET[t] = min(ET[t], S_temp)

    # Updating the storage after evapotranspiration
    S[t] = S_temp - ET[t]

    # Updating the storage for the next time step
    S_prev = S[t]

# Plotting the results
fig, ax = plt.subplots()

# Plotting storage, discharge and evapotranspiration
ax.plot(S, label="Storage (S)", color="darkgreen")
ax.plot(Q, label="Discharge (Q)", color="darkred")
ax.plot(ET, label="Evapotranspiration (ET)", color="darkorange")

ax.set_xlabel("Time step [Day]", fontsize=14)
ax.set_ylabel("Storage [mm], discharge and ET [mm/day]", fontsize=14)
ax.set_title("Exercise 2c: Storage Capacity and ET", fontsize=18, fontweight="bold")
ax.set_xticks(range(0, 65, 5))
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Plotting precipitation on the right y-axis
ax2 = ax.twinx()
ax2.bar(range(n_steps), P, color="darkblue", alpha=0.5, label="Precipitation")
ax2.set_ylabel("Precipitation [mm]", fontsize=14)
ax2.set_ylim(0, 40)
ax2.set_yticks(range(0, 41, 10))
ax2.invert_yaxis()

# Combining the legends from both y-axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)

plt.show()