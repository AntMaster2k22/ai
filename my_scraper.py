# my_scraper.py

from trafilatura import fetch_url, extract
import logging

# --- Configure logging to be less verbose for this module ---
# This prevents trafilatura from printing too much information during scraping.
logging.getLogger("trafilatura").setLevel(logging.WARNING)

def scrape_text_from_url(url: str, silent: bool = False):
    """
    Downloads the content of a URL and extracts the main article text.
    Uses trafilatura for robust and reliable content extraction.

    Args:
        url (str): The URL to scrape.
        silent (bool): If True, suppresses print statements for use in autonomous scripts.

    Returns:
        str: The cleaned main text of the article, or None if scraping fails.
    """
    if not silent:
        print(f"  -> Downloading and extracting content from URL...")

    # 1. Download the webpage content
    downloaded = fetch_url(url)
    
    if downloaded is None:
        if not silent:
            print("[!] Failed to download the webpage. It might be down or blocking requests.")
        return None

    # 2. Extract the main article text, removing comments, tables, and other noise
    text = extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=True # Ensures we only get main content
    )

    if text:
        if not silent:
            print(f"  [✓] Successfully extracted {len(text.split())} words.")
        return text
    else:
        if not silent:
            print("[!] Could not extract a meaningful article from the page.")
        return None