import os
import requests
import json
from tqdm import tqdm

BASE_URL = "https://www.cbioportal.org/api"
HEADERS = {"Accept": "application/json"}
STUDY_ID = "msk_chord_2024"
OUTPUT_DIR = "patient_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Get all patients in the study
patients_url = f"{BASE_URL}/studies/{STUDY_ID}/patients"
patients = requests.get(patients_url, headers=HEADERS).json()

# Step 2: Get all clinical attributes
attributes_url = f"{BASE_URL}/clinical-attributes"
attributes = requests.get(attributes_url, headers=HEADERS).json()

# Step 3: Get sample list and molecular profiles (if needed)
sample_lists_url = f"{BASE_URL}/studies/{STUDY_ID}/sample-lists"
sample_lists = requests.get(sample_lists_url, headers=HEADERS).json()

# Pick default sample list
sample_list_id = sample_lists[0]["sampleListId"]

# Optional: mutation data (you can fetch profile list and filter)
mutation_profile_id = f"{STUDY_ID}_mutations"

# Step 4: Loop through each patient and save clinical data
for patient in tqdm(patients, desc="Downloading patient data"):
    patient_id = patient["patientId"]

    # Patient clinical data
    clinical_url = f"{BASE_URL}/studies/{STUDY_ID}/patients/{patient_id}/clinical-data"
    clinical = requests.get(clinical_url, headers=HEADERS).json()

    # Save to file
    with open(f"{OUTPUT_DIR}/{patient_id}.json", "w") as f:
        json.dump({
            "patient_id": patient_id,
            "clinical_data": clinical
        }, f, indent=4)
