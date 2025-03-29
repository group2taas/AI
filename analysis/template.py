import tempfile
import unittest
import uuid
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from zapv2 import ZAPv2
import os
from dotenv import load_dotenv

load_dotenv()
ZAP_API_KEY = os.getenv("ZAP_API_KEY")
ZAP_PROXY_SERVER = os.getenv("ZAP_PROXY_SERVER")
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH")
# CHROMIUM_BINARY = os.getenv("CHROMIUM_BINARY")

class OWASP_SecurityTests(unittest.TestCase):

    def setUp(self):
        self.target_url = "<target_url>"  
        self.zap_proxy = ZAP_PROXY_SERVER
        self.results = {}  
        self.scan_timeout = 300
        
        try:
            service = Service(executable_path=CHROMEDRIVER_PATH)
            options = webdriver.ChromeOptions()
            # if CHROMIUM_BINARY:
            #     options.binary_location = CHROMIUM_BINARY
            user_data_dir = tempfile.mkdtemp(prefix=f"chrome_data_{uuid.uuid4()}_")
            options.add_argument(f"--user-data-dir={user_data_dir}")

            options.add_argument(f"--proxy-server={self.zap_proxy}")
            options.add_argument('--ignore-certificate-errors')
            options.add_argument("--disable-gpu")
            options.add_argument('--disable-webrtc')
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.delete_all_cookies()
            self.zap = ZAPv2(apikey=ZAP_API_KEY, proxies={"http": self.zap_proxy, "https": self.zap_proxy})
            self.zap.core.delete_all_alerts()
            self.visited_urls = set()
            self.crawl_and_scan(self.target_url)
            self.alert_count_before = len(self.zap.core.alerts(baseurl=self.target_url))

        except Exception as e:
            print(json.dumps({
                "type": "error", 
                "message": f"Setup failed: {str(e)}"
            }))
            raise
    
    def crawl_and_scan(self, url):
        if url in self.visited_urls:
            return
        self.visited_urls.add(url)

        try:
            self.driver.get(url)
            time.sleep(1)
            self.zap.urlopen(url)
            time.sleep(1)
            
            link_elements = self.driver.find_elements(By.TAG_NAME, "a")
            for i in range(len(link_elements)):
                try:
                    link_elements = self.driver.find_elements(By.TAG_NAME, "a")
                    href = link_elements[i].get_attribute("href")
                    if href and href.startswith(self.target_url) and href not in self.visited_urls:
                        self.crawl_and_scan(href)
                except Exception as e:
                    continue

        except Exception as e:
            self.results[f"crawl_{url}"] = f"error: {str(e)}"


    def run_zap_scans(self):
        try:
            
            self.zap.spider.scan(self.target_url)
            
            start_time = time.time()
            while int(self.zap.spider.status()) < 100:
                if time.time() - start_time > self.scan_timeout:
                    return False
                    
                progress = int(self.zap.spider.status())
                time.sleep(2)
            
            start_time = time.time()
            while int(self.zap.pscan.records_to_scan) > 0:
                if time.time() - start_time > self.scan_timeout:

                    return False
                    
                remaining = int(self.zap.pscan.records_to_scan)

                time.sleep(2)

            self.zap.ascan.scan(self.target_url)
            
            start_time = time.time()
            while int(self.zap.ascan.status()) < 100:
                if time.time() - start_time > self.scan_timeout:
                    return False
                    
                progress = int(self.zap.ascan.status()) 
                time.sleep(2)
            return True
            
        except Exception as e:
            return False

    def tearDown(self):
        try:
            self.driver.quit()
            for test in self.results.keys():
                result_data = self.generate_summary_report(test)
                print(json.dumps(result_data))
        except Exception as e:
            print(json.dumps({
                "type": "error",
                "message": f"Error in tearDown: {str(e)}"
            }))

    def generate_summary_report(self, test_name):
        try:
            all_alerts = self.zap.core.alerts(baseurl=self.target_url)
            new_alerts = all_alerts[self.alert_count_before:]
            
            alert_details = []
            for alert in new_alerts:
                alert_details.append({
                    "name": alert.get("name", "Unknown"),
                    "risk": alert.get("risk", "Informational"),
                    "description": alert.get("description", ""),
                    "solution": alert.get("solution", ""),
                    "url": alert.get("url", "")
                })
                   
            severity_counts = {
                "High": 0,
                "Medium": 0,
                "Low": 0,
                "Informational": 0
            }

            for alert in new_alerts:
                severity = alert.get("risk", "Informational")
                if severity in severity_counts:
                    severity_counts[severity] += 1
            
            summary_data = {
                "type": "result",
                "target_url": self.target_url,
                "security_alerts": severity_counts,
                "alert_details": alert_details,
                "test_case": test_name,
                "result": self.results[test_name]
            }

            return summary_data
        except Exception as e:
            return {
                "type": "error",
                "message": f"Error generating report: {str(e)}",
                "test_case": test_name
            }

if __name__ == "__main__":
    unittest.main()
