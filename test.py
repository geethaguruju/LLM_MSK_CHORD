import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

pid = "P-0000628"
study_id = "msk_chord_2024"
chromedriver_path = "chromedriver-mac/chromedriver"  # Update path if needed

# Launch Chrome in visible mode
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Navigate to summary tab
    summary_url = f"https://www.cbioportal.org/patient/summary?studyId={study_id}&caseId={pid}"
    driver.get(summary_url)
    print(f"[INFO] Opened Summary Tab for {pid}")
    time.sleep(10)  # wait for full React load

    # Scroll to bottom to trigger rendering
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)

    # Access shadow root of the <cbio-patient-summary-page> custom element
    host_element = driver.find_element(By.CSS_SELECTOR, "cbio-patient-summary-page")
    shadow_root = driver.execute_script("return arguments[0].shadowRoot", host_element)

    # Now find all tables within the shadow DOM
    tables = shadow_root.find_elements(By.TAG_NAME, "table")
    print(f"[DEBUG] Found {len(tables)} tables inside shadow DOM")

    for i, table in enumerate(tables):
        headers = [th.text.strip() for th in table.find_elements(By.TAG_NAME, "th")]
        rows = table.find_elements(By.TAG_NAME, "tr")
        print(f"Table {i+1}: {len(rows)-1} rows, Headers: {headers}")

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    input("Press Enter to close browser...")
    driver.quit()
