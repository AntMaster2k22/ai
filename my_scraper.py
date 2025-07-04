import trafilatura
from requests.exceptions import RequestException

def scrape_text_from_url(url: str) -> str:
    """
    Scrapes the main text content from a URL using trafilatura.
    Returns the text as a single string, or an empty string on failure.
    """
    print(f"[*] Scraping: {url}")
    try:
        # Download the page's content
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            print(f"[!] Failed to download content from {url}")
            return ""
        
        # Extract the main text, excluding comments, tables, and reducing duplication
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            deduplicate=True
        )
        
        if text:
            print(f"[✓] Successfully scraped text from {url}")
            # Join lines to create a single block of text
            return '\n'.join(text.splitlines())
        else:
            print(f"[!] No main text could be extracted from {url}")
            return ""
            
    except RequestException as e:
        print(f"[!] Request failed for {url}: {e}")
        return ""
    except Exception as e:
        print(f"[!] An unexpected error occurred while scraping {url}: {e}")
        return ""

