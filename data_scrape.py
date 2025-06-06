import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager
from collections import defaultdict

# Read the TSV file
data = pd.read_csv('msk_chord_2024_clinical_data.tsv', sep='\t')
patient_ids = sorted(set(data['Patient ID']))

# Filter out patient IDs that already have corresponding JSON files
patient_ids = [pid for pid in patient_ids if not os.path.exists(f"patient_data_new/{pid}.json")]

# Write patient IDs to a file
with open("patient_ids.txt", "w") as file:
    for pid in patient_ids:
        file.write(f"{pid}\n")

BASE_URL = "https://www.cbioportal.org/patient/clinicalData?studyId=msk_chord_2024&caseId="

def process_patient(pid):
    output_path = f"patient_data_scraped/{pid}.json"
    if os.path.exists(output_path):
        return  # Skip already processed

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    service = Service('chromedriver')
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        url = f"{BASE_URL}{pid}"
        driver.get(url)
        time.sleep(12)

        tables = driver.find_elements(By.TAG_NAME, "table")[1:]  # skip summary table
        extracted_tables = []

        for table in tables:
            headers = [header.text.strip() for header in table.find_elements(By.TAG_NAME, "th")]
            rows = table.find_elements(By.TAG_NAME, "tr")[1:]
            table_data = [[cell.text.strip() for cell in row.find_elements(By.TAG_NAME, "td")] for row in rows]
            if headers and table_data:
                df = pd.DataFrame(table_data, columns=headers)
                extracted_tables.append(df)

        # Build output grouped by EVENT_TYPE
        json_tables = defaultdict(list)
        for df in extracted_tables:
            records = json.loads(df.to_json(orient='records'))
            for record in records:
                event_type = record.get("EVENT_TYPE", "CLINICAL_DATA")
                if event_type:
                    json_tables[event_type].append(record)

        with open(output_path, "w") as json_file:
            json.dump(json_tables, json_file, indent=4)

        with open("processed_patients.txt", "a") as log_file:
            log_file.write(f"{pid}\n")
        print(f"Processed: {pid}")

    except Exception as e:
        with open("failed_patient_data.txt", "a") as fail_file:
            fail_file.write(f"{pid}\n")
        print(f"Failed: {pid} – {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    os.makedirs("patient_data_scraped", exist_ok=True)

    with Manager() as manager:
        with Pool(processes=10) as pool:
            args = [(pid,) for pid in patient_ids]
            list(tqdm(pool.starmap(process_patient, args), total=len(patient_ids), desc="Processing Patients"))
