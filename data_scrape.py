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

# Read the TSV file
data = pd.read_csv('msk_chord_2024_clinical_data.tsv', sep='\t')

# Extract the patient ID column
patient_ids = data['Patient ID']
patient_ids = list(set(list(patient_ids)))
patient_ids.sort()

# Filter out patient IDs that already have corresponding JSON files
patient_ids = [pid for pid in patient_ids if not os.path.exists(f"patient_data/{pid}.json")]

# Write patient IDs to a file
with open("patient_ids.txt", "w") as file:
    for pid in patient_ids:
        file.write(f"{pid}\n")

BASE_URL = "https://www.cbioportal.org/patient/clinicalData?studyId=msk_chord_2024&caseId="

table_info = ["clinical_data", "demographic", "diagnosis", "sample_acquisition", "lab_test", "treatment", "surgery", "sequencing", "therapy"]

def process_patient(pid):

    if os.path.exists(f"patient_data/{pid}.json"):
        return # Skip already processed patient IDs

    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    service = Service('chromedriver-linux64/chromedriver')  # Update path to your ChromeDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        url = f"{BASE_URL}{pid}"
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load

        # Extract tables
        tables = driver.find_elements(By.TAG_NAME, "table")
        extracted_tables = []
        
        for table in tables[1:]:
            # Extract header row
            headers = [header.text for header in table.find_elements(By.TAG_NAME, "th")]
            
            # Extract data rows
            rows = table.find_elements(By.TAG_NAME, "tr")
            table_data = []
            
            row_num = 1
            for row in rows[1:]:
                cells = row.find_elements(By.TAG_NAME, "td")
                table_data.append([cell.text for cell in cells])
                row_num += 1
            
            # Create DataFrame with headers if available
            if table_data:
                if headers:
                    extracted_tables.append(pd.DataFrame(table_data, columns=headers))
                else:
                    extracted_tables.append(pd.DataFrame(table_data))

        # Convert extracted tables to JSON format
        json_tables = {}
        for i, df in enumerate(extracted_tables):
            if i < len(table_info):
                key = table_info[i]
            else:
                key = "timeline_data"
            json_tables[key] = json.loads(df.to_json(orient='records'))

        # Save JSON tables to file
        with open(f"patient_data/{pid}.json", "w") as json_file:
            json.dump(json_tables, json_file, indent=4)

        # Log processed patient ID
        with open("processed_patients.txt", "a") as processed_patients_file:
            processed_patients_file.write(f"{pid}\n")

        print(f"Processed patient ID: {pid}")
    except Exception as e:
        with open("failed_patient_data.txt", "a") as failed_patients_file:
            failed_patients_file.write(f"{pid}\n")
        print(f"Failed to process patient ID: {pid}")
        print(e)
    finally:
        driver.quit()

if __name__ == "__main__":

    # Use multiprocessing to process patient IDs
    with Manager() as manager:
        failed_patients = manager.list()
        with Pool(processes=10) as pool:  # Adjust the number of processes as needed
            args = [(pid,) for pid in patient_ids]
            list(tqdm(pool.starmap(process_patient, args), total=len(patient_ids), desc="Processing Patients"))
