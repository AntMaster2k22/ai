# my_scraper.py

from trafilatura import fetch_url, extract
import logging
import requests # Import requests to catch its exceptions specifically

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
        print(f"  -> Downloading and extracting content from URL: {url}...") # Added URL to print statement

    try:
        # 1. Download the webpage content
        # trafilatura.fetch_url internally handles many requests.exceptions
        downloaded = fetch_url(url)

        if downloaded is None:
            if not silent:
                # Be more specific about why it might fail
                print(f"[!] Failed to download the webpage from {url}. It might be down, blocking requests, or content type not supported.")
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
                print(f"  [✓] Successfully extracted {len(text.split())} words from {url}.") # Added URL to print statement
            return text
        else:
            if not silent:
                print(f"[!] Could not extract a meaningful article from the page {url}. Content might be sparse or non-textual.") # Added URL and more specificity
            return None
    except requests.exceptions.Timeout:
        if not silent:
            print(f"[x] Timeout occurred while scraping {url}.")
        return None
    except requests.exceptions.ConnectionError:
        if not silent:
            print(f"[x] Connection Error while scraping {url}. Check internet connection or URL.")
        return None
    except requests.exceptions.RequestException as e: # Catch any other requests-related errors
        if not silent:
            print(f"[x] An unknown Request Error occurred while scraping {url}: {e}")
        return None
    except Exception as e: # Catch any other unexpected errors during trafilatura processing
        if not silent:
            print(f"[x] An unexpected error occurred during scraping {url}: {e}")
        return None