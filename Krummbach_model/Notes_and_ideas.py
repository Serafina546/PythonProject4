# Import der einzelnen Modell-Komponenten
from snow_model import calculate_snow
from pet import calculate_pet
from interception import calculate_interception
from flowerpot import SoilBucket


def run_watershed_model(time_series_data, parameters):
    # 1. Initialisierung von Speichern (z.B. Boden-Speicher / Flowerpot)
    soil_storage = SoilBucket(capacity=parameters['soil_capacity'])

    # Ergebnis-Liste für den Abfluss
    total_discharge = []

    # 2. Zeitliche Schleife (z. B. über alle Zeitschritte / Tage)
    for t, data in enumerate(time_series_data):
        precip = data['precipitation']
        temp = data['temperature']
        radiation = data['radiation']

        # A. Schneeprozesse berechnen
        snow_melt, snow_storage_state = calculate_snow(precip, temp, state=...)

        # B. Potenzielles PET berechnen
        pet_value = calculate_pet(temp, radiation)

        # C. Interzeption abziehen
        effective_precip = calculate_interception(precip + snow_melt, pet_value)

        # D. Boden-Speicher ("Flowerpot") aktualisieren
        soil_storage.update(effective_precip, pet_value)

        # E. Abfluss über "get"-Methoden oder Rückgabewerte abrufen
        runoff = soil_storage.get_runoff()
        total_discharge.append(runoff)

    return total_discharge