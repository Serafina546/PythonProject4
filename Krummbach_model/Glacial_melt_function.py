import pandas as pd
import numpy as np

# Load temperature dataset
df = pd.read_csv(r"../data/CAMELS_CH_obs_based_2244.csv")

# Check dataset
print(df.head())
def glacier_melt(temperature, glacier_areas,catchment_area, ddf=5.0, T0=0.0):
    glacier_areas = np.asarray(glacier_areas)
    if catchment_area <= 0:
        raise ValueError("Catchment area must be positive.")
    if np.any(glacier_areas < 0):
        raise ValueError("Glacier areas cannot be negative.")
    if np.sum(glacier_areas) > catchment_area:
        raise ValueError("Glacier area exceeds catchment area.")
    if ddf < 0:
        raise ValueError("DDF cannot be negative.")

    # Calculate daily melt for every temperature
    temperature = np.asarray(temperature)

    melt_depth = ddf * np.maximum(temperature - T0, 0)

    # Calculate melt volume
    melt_per_glacier = (
        melt_depth[:, None] / 1000
        * glacier_areas[None, :]
        * 1e6
    )

    # Total melt volume per day
    total_melt_volume = np.sum(melt_per_glacier, axis=1)

    # Convert to catchment equivalent
    glacier_input = (
        total_melt_volume / (catchment_area * 1e6) * 1000
    )
    return {
        "melt_depth": melt_depth,
        "melt_per_glacier": melt_per_glacier,
        "total_melt_volume": total_melt_volume,
        "glacier_input": glacier_input
    }

# Example input data

temperature = df["temperature_mean(degC)"].to_numpy()

print("Erste Temperaturwerte:")
print(temperature[:20])

print("Minimum:", np.nanmin(temperature))
print("Maximum:", np.nanmax(temperature))
print("Anzahl Werte über 0 °C:", np.sum(temperature > 0))
print("Anzahl fehlender Werte:", np.isnan(temperature).sum())

glacier_areas = [2]  # Gletscherfläche noch anpassen
catchment_area = 80

# Calculate glacier melt
results = glacier_melt(
    temperature=temperature,
    glacier_areas=glacier_areas,
    catchment_area=catchment_area,
    ddf=5.0,
    T0=0.0
)

# Print results
print("Daily glacier melt:")
print("-------------------")
print("Melt depth:",
      results["melt_depth"], "mm/day")
print("Melt per glacier:",
      results["melt_per_glacier"], "m3/day")
print("Total melt volume:",
      results["total_melt_volume"], "m3/day")
print("Glacier input to catchment:",
      results["glacier_input"], "mm/day")
print(results["total_melt_volume"].max())        # sollte ca. 199800 sein
print(np.sum(results["total_melt_volume"] > 0))  # sollte 8074 sein