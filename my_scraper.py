import trafilatura
from requests.exceptions import RequestException

def scrape_text_from_url(url: str, silent: bool = False) -> str:
    """
    Scrapes the main text content from a URL using trafilatura.
    Returns the text as a single string, or an empty string on failure.
    
    Args:
        url: The URL to scrape.
        silent: If True, suppresses print statements for cleaner autonomous runs.
    """
    if not silent:
        print(f"[*] Scraping: {url}")
        
    try:
        # Download the page's content
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            if not silent:
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
            if not silent:
                print(f"[✓] Successfully scraped text from {url}")
            # Join lines to create a single block of text
            return '\n'.join(text.splitlines())
        else:
            if not silent:
                print(f"[!] No main text could be extracted from {url}")
            return ""
            
    except RequestException as e:
        if not silent:
            print(f"[!] Request failed for {url}: {e}")
        return ""
    except Exception as e:
        if not silent:
            print(f"[!] An unexpected error occurred while scraping {url}: {e}")
        return ""