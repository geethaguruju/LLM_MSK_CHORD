# === Merged Script for Rescraping Empty Patient Files ===

import os
import json
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, Manager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# --------------------------------------------------------
# STEP 1: Identify empty or invalid JSON files
# --------------------------------------------------------

folder = "patient_data_scraped"
empty_patients = []

print("🔍 Scanning for empty or corrupt patient files...")

for filename in os.listdir(folder):
    if filename.endswith(".json"):
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if not data:
                    empty_patients.append(filename.replace(".json", ""))
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            empty_patients.append(filename.replace(".json", ""))

with open("empty_patients2.txt", "w") as out:
    for pid in empty_patients:
        out.write(f"{pid}\n")

print(f"\n✅ Total empty/invalid patient files: {len(empty_patients)}")

# --------------------------------------------------------
# STEP 2: Rescrape patient data using Selenium
# --------------------------------------------------------

# Load patient IDs to reprocess
with open("empty_patients2.txt", "r") as f:
    patient_ids = [line.strip() for line in f if line.strip()]

# Remove old empty/corrupt files in patient_data_new before retry
for pid in patient_ids:
    path = f"patient_data_new/{pid}.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as jf:
                data = json.load(jf)
            if not data:
                os.remove(path)
        except json.JSONDecodeError:
            os.remove(path)

BASE_URL = "https://www.cbioportal.org/patient/clinicalData?studyId=msk_chord_2024&caseId="

def process_patient(pid):
    output_path = f"patient_data_new/{pid}.json"

    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    service = Service("chromedriver")

    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        url = f"{BASE_URL}{pid}"
        driver.get(url)
        time.sleep(5)

        tables = driver.find_elements(By.TAG_NAME, "table")[1:]  # Skip summary table
        extracted_tables = []

        for table in tables:
            headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, "th")]
            rows = table.find_elements(By.TAG_NAME, "tr")[1:]
            data_rows = [[td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")] for row in rows]
            if headers and data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                extracted_tables.append(df)

        # Group by EVENT_TYPE
        json_tables = defaultdict(list)
        for df in extracted_tables:
            records = json.loads(df.to_json(orient='records'))
            for record in records:
                event_type = record.get("EVENT_TYPE", "CLINICAL_DATA")
                json_tables[event_type].append(record)

        if json_tables:  # Only write non-empty results
            with open(output_path, "w") as f:
                json.dump(json_tables, f, indent=4)
        else:
            raise ValueError("Scraped data is empty.")

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        with open("failed_patient_data.txt", "a") as fail:
            fail.write(f"{pid}\n")
        print(f"❌ Failed: {pid} – {e}")
    finally:
        driver.quit()

# --------------------------------------------------------
# STEP 3: Parallel Execution
# --------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("patient_data_new", exist_ok=True)

    with Manager():
        with Pool(processes=10) as pool:
            list(tqdm(pool.imap_unordered(process_patient, patient_ids), total=len(patient_ids), desc="🔁 Retrying Patients"))
