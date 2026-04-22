from bs4 import BeautifulSoup
import re

def analyze():
    with open("debug.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find the image I saw earlier
    # .../0/6/0610000108279.jpg...
    img = soup.find('link', attrs={'imagesrcset': re.compile(r'0610000108279')})
    if img:
        print("Found image link tag:")
        print(img)
        print("Parent:")
        print(img.parent.name)
    else:
        print("Image link tag not found via BS4")

    # Try to find actual img tags
    imgs = soup.find_all('img')
    print(f"Found {len(imgs)} img tags")
    for i in imgs[:5]:
        print(i.get('src'))
        print(i.parent)
        print("-" * 20)

    # Search for product links
    links = soup.find_all('a', href=True)
    product_links = []
    for link in links:
        href = link.get('href')
        # Heuristic: product links usually end with .html or have a slug
        # and are not standard nav links
        if '/bgd/' in href and 'catalogsearch' not in href and 'customer' not in href:
            product_links.append(link)
    
    print(f"Found {len(product_links)} potential product links")
    if product_links:
        l = product_links[0]
        print(f"Link: {l.get('href')}")
        print(f"Link classes: {l.get('class')}")
        
        # Go up to find the card
        curr = l.parent
        for _ in range(5):
            if not curr: break
            print(f"Parent: {curr.name}, Class: {curr.get('class')}")
            curr = curr.parent

if __name__ == "__main__":
    analyze()
