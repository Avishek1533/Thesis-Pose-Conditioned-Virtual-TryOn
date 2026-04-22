import os
import time
import re
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import unquote

# Configuration
SEARCH_URL = "https://www.aarong.com/bgd/catalogsearch/result?q=kurti&product_list_order=high_to_low"
OUTPUT_DIR = os.path.join(os.getcwd(), "Kurti")
TARGET_COUNT = 30

def setup_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # Keep visible for now to debug
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scroll_to_load(driver):
    print("Scrolling to load products...")
    # Scroll down more times to ensure we have enough products
    for _ in range(30): 
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)


def get_product_links(driver):
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = soup.find_all('a', href=True)
    product_links = set()
    
    for link in links:
        href = link.get('href')
        if not href: continue
        
        # Filter for product links
        # Products usually end in .html
        if '/bgd/' in href and href.endswith('.html') and 'catalogsearch' not in href:
            # Basic validation to avoid generic links
            if len(href.split('/')[-1]) > 5:
                if href.startswith('/'):
                    href = "https://www.aarong.com" + href
                product_links.add(href)
    
    return list(product_links)

def download_image(url, folder, filename):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            filepath = os.path.join(folder, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download {url}: Status {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def scrape_product(driver, url):
    print(f"Visiting: {url}")
    
    # Create a safe folder name from title or slug
    slug = url.split('/')[-1].replace('.html', '')
    product_folder = os.path.join(OUTPUT_DIR, slug)
    
    # Skip if already scraped
    if os.path.exists(product_folder) and len(os.listdir(product_folder)) >= 4:
        print(f"Skipping: Already scraped {slug}")
        return True

    driver.get(url)
    time.sleep(3) # Wait for load
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Extract Images
    # Look for high-res images in the page
    # Aarong seems to use next/image, so we look for 'src' or 'srcset' in img tags
    # or link tags with imagesrcset
    
    image_urls = set()
    
    # Method 1: Look for link rel="preload" as="image" (often high quality)
    # This usually captures the main gallery images which are preloaded
    link_imgs = soup.find_all('link', attrs={'as': 'image', 'imagesrcset': True})
    for link in link_imgs:
        srcset = link.get('imagesrcset')
        if srcset:
            # Get the largest image from srcset
            parts = srcset.split(',')
            if parts:
                last_part = parts[-1].strip() # Usually the largest
                img_url = last_part.split(' ')[0]
                # Decode URL if needed
                img_url = unquote(img_url)
                # Fix next/image url param
                if 'url=' in img_url:
                    match = re.search(r'url=(.*?)&', img_url)
                    if match:
                        img_url = unquote(match.group(1))
                
                if img_url.startswith('http'):
                    image_urls.add(img_url)

    # Method 2: Standard img tags (Fallback)
    if len(image_urls) < 4:
        print("Method 1 yielded few images. Trying Method 2 (img tags)...")
        imgs = soup.find_all('img')
        for img in imgs:
            src = img.get('src')
            if src and 'media/catalog/product' in src:
                # Fix next/image url param
                if 'url=' in src:
                    match = re.search(r'url=(.*?)&', src)
                    if match:
                        src = unquote(match.group(1))
                
                if src.startswith('http'):
                    # Filter out small thumbnails if possible, or rely on unique count
                    # Main images usually don't have 'tiny' or 'small' in the name, but sometimes they do.
                    # Let's just collect them and rely on the count.
                    # To avoid "You May Also Like", we could check parents, but let's see if the count filter helps.
                    # Usually related products have 1 image each. If we get 4+ unique images, it's likely the main gallery.
                    image_urls.add(src)
    
    if len(image_urls) < 4:
        print(f"Skipping: Found only {len(image_urls)} images (required 4+)")
        return False

    print(f"Found {len(image_urls)} images. Proceeding to download.")
    
    # Create a safe folder name from title or slug
    slug = url.split('/')[-1].replace('.html', '')
    product_folder = os.path.join(OUTPUT_DIR, slug)
    if not os.path.exists(product_folder):
        os.makedirs(product_folder)
        
    for i, img_url in enumerate(image_urls):
        ext = img_url.split('.')[-1].split('?')[0]
        if len(ext) > 4 or not ext: ext = 'jpg'
        filename = f"{slug}_{i+1}.{ext}"
        download_image(img_url, product_folder, filename)
        
    return True

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    driver = setup_driver()
    try:
        driver.get(SEARCH_URL)
        time.sleep(5)
        
        # Save debug HTML
        with open("debug_kurti.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved debug_kurti.html")
        
        scroll_to_load(driver)
        
        links = get_product_links(driver)
        print(f"Found {len(links)} total product links")
        
        scraped_count = 0
        for link in links:
            if scraped_count >= TARGET_COUNT:
                print(f"Target of {TARGET_COUNT} items reached.")
                break
                
            try:
                success = scrape_product(driver, link)
                if success:
                    scraped_count += 1
                    print(f"Scraped {scraped_count}/{TARGET_COUNT} products")
            except Exception as e:
                print(f"Error scraping {link}: {e}")
            
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
