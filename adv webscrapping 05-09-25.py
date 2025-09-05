from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import logging
import os

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    filename="toscrape_books.log",
    filemode="w",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------------
# Edge Browser Options
# -----------------------------
options = Options()
options.add_argument("--incognito")
options.add_argument("--window-size=1920,1080")
# options.add_argument("--headless")  # Uncomment to run in background

# Optional: Avoid detection
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

# -----------------------------
# Setup Edge Driver (Option 1: Auto-download with webdriver-manager) ✅ RECOMMENDED
# -----------------------------
try:
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=options)
    logging.info("Edge driver set up successfully using webdriver-manager.")
except Exception as e:
    logging.warning(f"Failed to set up webdriver-manager: {e}")
    print("⚠️  Could not use webdriver-manager. Falling back to manual driver...")

    # -----------------------------
    # Fallback: Manual Driver (Option 2)
    # -----------------------------
    driver_path = "msedgedriver.exe"
    if not os.path.exists(driver_path):
        logging.critical(f"Driver not found: {os.path.abspath(driver_path)}")
        print("❌ Error: msedgedriver.exe not found in the current directory.")
        print("👉 Download it from: https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/")
        exit()

    service = Service(executable_path=driver_path)
    driver = webdriver.Edge(service=service, options=options)

# -----------------------------
# Open Website (Fixed URL - no extra spaces!)
# -----------------------------
url = "https://books.toscrape.com"  # 🔴 Fixed: Removed trailing spaces
driver.get(url)

try:
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "article.product_pod"))
    )
    logging.info("Successfully loaded the homepage.")
    print("✅ Successfully accessed the website.")
except Exception as e:
    logging.error(f"Failed to load page: {e}")
    print("❌ Failed to load the website.")
    driver.quit()
    exit()

# -----------------------------
# Scraping Logic
# -----------------------------
books_data = []
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

page_count = 0

while True:
    page_count += 1
    logging.info(f"Scraping page {page_count}")

    try:
        books = driver.find_elements(By.CSS_SELECTOR, "article.product_pod")
        logging.info(f"Found {len(books)} books on page {page_count}")

        for book in books:
            try:
                # Title
                title = book.find_element(By.CSS_SELECTOR, "h3 a").get_attribute("title")

                # Price
                price = book.find_element(By.CSS_SELECTOR, "div.product_price p.price_color").text

                # Rating
                rating_class = book.find_element(By.CSS_SELECTOR, "p.star-rating").get_attribute("class")
                rating_word = rating_class.strip().split()[-1]
                rating = rating_map.get(rating_word, 0)

                books_data.append({
                    "Title": title,
                    "Price": price,
                    "Rating (Stars)": rating
                })

            except Exception as e:
                logging.warning(f"Error parsing individual book: {e}")
                continue

    except Exception as e:
        logging.error(f"Error finding books on page {page_count}: {e}")
        break

    # Try to go to next page
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, "li.next a")
        old_url = driver.current_url
        next_button.click()

        # Wait for new content to load
        WebDriverWait(driver, 10).until(EC.url_changes(old_url))
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article.product_pod"))
        )
        logging.info("Navigated to next page.")

    except Exception:
        logging.info("No more 'next' button found. Reached the last page.")
        print(f"🔚 Finished scraping. Total pages: {page_count}, Total books: {len(books_data)}")
        break

# -----------------------------
# Save Data to CSV
# -----------------------------
if books_data:
    df = pd.DataFrame(books_data)
    output_file = "books.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"✅ Scraping complete. {len(books_data)} books saved to {output_file}")
    print(f"✅ Scraping complete. Data saved to '{output_file}'")
else:
    logging.warning("No books were scraped.")
    print("❌ No data was scraped.")

# -----------------------------
# Close Browser
# -----------------------------
driver.quit()