from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def probe():
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # Run headless for speed
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        url = "https://www.aarong.com/bgd/catalogsearch/result?q=kamiz&product_list_order=high_to_low"
        driver.get(url)
        time.sleep(5) # Wait for load

        print(f"Page Title: {driver.title}")
        
        with open("debug.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved page source to debug.html")
        
        products = driver.find_elements(By.CSS_SELECTOR, ".product-item")
        print(f"Found {len(products)} products with .product-item")

    finally:
        driver.quit()

if __name__ == "__main__":
    probe()
