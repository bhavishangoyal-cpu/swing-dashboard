import requests
import time
from datetime import datetime

class SmartMoneyScanner:
    def __init__(self, email):
        self.headers = {"User-Agent": f"MarketMonitor {email}"}

    def scan(self):
        url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&owner=include&count=40&output=atom"
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking SEC feed...")
            response = requests.get(url, headers=self.headers)

            # Add this to see if you are getting a response at all
            if response.status_code != 200:
                print(f"Warning: Received status code {response.status_code}")
                return

            targets = ["4", "SC 13D", "13F-HR"]
            found_any = False
            for form in targets:
                if f">{form}<" in response.text:
                    alert = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Smart Money detected: {form}"
                    print(alert)
                    with open("alerts.txt", "a") as f:
                        f.write(f"{alert}\n")
                    found_any = True

            if not found_any:
                print("No target filings found in current batch.")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    scanner = SmartMoneyScanner("bhavishangoyal@gmail.com")
    print("Scanner started...")
    while True:
        scanner.scan()
        time.sleep(300) # Polls every 5 minutes