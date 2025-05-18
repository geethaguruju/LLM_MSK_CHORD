import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager

# Read the TSV file
data = pd.read_csv('msk_chord_2024_clinical_data.tsv', sep='\t')
patient_ids = sorted(set(data['Patient ID']))
patient_ids = [pid for pid in patient_ids if not os.path.exists(f"patient_data/{pid}.json")]

with open("patient_ids.txt", "w") as file:
    for pid in patient_ids:
        file.write(f"{pid}\n")

BASE_URL = "https://www.cbioportal.org/patient/clinicalData?studyId=msk_chord_2024&caseId="
table_info = ["clinical_data", "patient", "samples", "demographic", "diagnosis", "sample_acquisition", "lab_test", "pathology", "treatment", "surgery", "sequencing", "therapy"]

def process_patient(pid):
    if os.path.exists(f"patient_data/{pid}.json"):
        return

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    service = Service('chromedriver-mac/chromedriver')
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # --- Clinical Data Tab ---
        driver.get(f"{BASE_URL}{pid}")
        time.sleep(5)
        tables = driver.find_elements(By.TAG_NAME, "table")
        extracted_tables = []

        for table in tables[1:]:
            headers = [th.text for th in table.find_elements(By.TAG_NAME, "th")]
            rows = table.find_elements(By.TAG_NAME, "tr")
            table_data = [[td.text for td in row.find_elements(By.TAG_NAME, "td")] for row in rows[1:]]
            if table_data:
                df = pd.DataFrame(table_data, columns=headers) if headers else pd.DataFrame(table_data)
                extracted_tables.append(df)

        json_tables = {}
        for i, df in enumerate(extracted_tables):
            key = table_info[i] if i < len(table_info) else f"table_{i}"
            json_tables[key] = json.loads(df.to_json(orient='records'))

        # # --- Pathway Alteration Table ---
        # try:
        #     driver.get(f"https://www.cbioportal.org/patient/pathways?studyId=msk_chord_2024&caseId={pid}")
        #     WebDriverWait(driver, 10).until(
        #         EC.presence_of_element_located((By.XPATH, "//div[contains(@class,'table-root')]//table"))
        #     )
        #     table_rows = driver.find_elements(By.XPATH, "//div[contains(@class,'table-root')]//tr")
        #     if len(table_rows) > 1:
        #         pathway_info = []
        #         for row in table_rows[1:]:
        #             cells = row.find_elements(By.TAG_NAME, "td")
        #             if len(cells) >= 2:
        #                 pathway_info.append({
        #                     "pathway": cells[0].text.strip(),
        #                     "altered": cells[1].text.strip(),
        #                     "matched_genes": cells[2].text.strip() if len(cells) > 2 else ""
        #                 })
        #         if pathway_info:
        #             json_tables["pathway_summary_table"] = pathway_info
        # except TimeoutException:
        #     print(f"Timeout: Pathway table did not load for {pid}")
        # except Exception as e:
        #     print(f"Pathway extraction error for {pid}: {e}")

        # # --- Summary Tab Tables ---
        # try:
        #     driver.get(f"https://www.cbioportal.org/patient/summary?studyId=msk_chord_2024&caseId={pid}")
        #     time.sleep(8)
        #     summary_tables = driver.find_elements(By.XPATH, "//div[contains(@class, 'table-root')]//table")
        #     keys = ["mutation_table", "structural_variant_table", "copy_number_alteration_table"]
        #     for i in range(min(3, len(summary_tables))):
        #         headers = [h.text.strip() for h in summary_tables[i].find_elements(By.TAG_NAME, "th")]
        #         rows = summary_tables[i].find_elements(By.TAG_NAME, "tr")
        #         data = []
        #         for row in rows[1:]:
        #             cells = row.find_elements(By.TAG_NAME, "td")
        #             data.append({headers[j]: cells[j].text.strip() for j in range(min(len(headers), len(cells)))})
        #         if data:
        #             json_tables[keys[i]] = data
        # except Exception as e:
        #     print(f"Summary tab error for {pid}: {e}")

        with open(f"patient_data/{pid}.json", "w") as f:
            json.dump(json_tables, f, indent=4)
        with open("processed_patients.txt", "a") as f:
            f.write(f"{pid}\n")
        print(f"Processed patient ID: {pid}")

    except Exception as e:
        with open("failed_patient_data.txt", "a") as f:
            f.write(f"{pid}\n")
        print(f"Failed to process patient ID: {pid}\\n{e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    with Manager() as manager:
        with Pool(processes=10) as pool:
            args = [(pid,) for pid in patient_ids]
            list(tqdm(pool.starmap(process_patient, args), total=len(patient_ids), desc="Processing Patients"))
