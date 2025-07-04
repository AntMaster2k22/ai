import requests
from bs4 import BeautifulSoup

WIKI_BASE = "https://en.wikipedia.org"
URLS_FILE = "urls.txt"
TOPICS_FILE = "topics.txt"

def get_current_urls():
    try:
        with open(URLS_FILE, "r") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def load_topics_from_file():
    try:
        with open(TOPICS_FILE, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[!] No {TOPICS_FILE} file found.")
        return []

def search_wikipedia_topics(search_terms):
    new_urls = []
    existing_urls = get_current_urls()

    for term in search_terms:
        print(f"[🔍] Searching for: {term}")
        resp = requests.get(f"https://en.wikipedia.org/wiki/{term.replace(' ', '_')}")
        if resp.status_code != 200:
            print(f"[!] Could not fetch page for '{term}'")
            continue

        url = resp.url
        if url in existing_urls:
            print(f"[↩️] Already in file: {url}")
            continue

        new_urls.append(url)
        existing_urls.add(url)

    if new_urls:
        with open(URLS_FILE, "a") as f:
            for url in new_urls:
                f.write(url + "\n")
        print(f"[✅] Added {len(new_urls)} new URLs.")
    else:
        print("[✓] No new URLs added.")

if __name__ == "__main__":
    topics = load_topics_from_file()
    if not topics:
        print("[!] No topics to search.")
    else:
        search_wikipedia_topics(topics)
