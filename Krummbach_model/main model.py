from snow_model_test import run_snow_model, plot_snow_results
from Flowerpot_linear_storage_test import run_pet_interception

def main():

    # 1. Snow model
    snow_results = run_snow_model()

    # 2. PET und Interception
    pet_results = run_pet_interception()

    # 3. Plots
    plot_snow_results(snow_results)

    print("Mittlere PET:", pet_results["PET"].mean())
    print("Mittlerer Abfluss:", pet_results["Q"].mean())

if __name__ == "__main__":
    main()