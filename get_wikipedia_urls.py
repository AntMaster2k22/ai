import os, glob, time, random, threading, requests
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from trafilatura import extract

from config import (
    DATA_DIR, URLS_FILE,
    HARVESTER_BLACKLISTED_DOMAINS,
    HARVESTER_BLACKLISTED_EXTENSIONS,
    HARVESTER_MIN_CONTENT_WORDS,
    SCRAPER_API_KEY,
    # ABSTRACT_API_KEY,      # <-- disabled
    # CRITIQUE_API_KEY       # <-- disabled
)

# --- LINK VALIDATION ---
def is_valid_link(url):
    if not url or not url.startswith('http'):
        return False
    try:
        domain = url.split('//')[1].split('/')[0].lower().replace('www.', '')
        if domain in HARVESTER_BLACKLISTED_DOMAINS:
            return False
        if any(url.lower().endswith(ext) for ext in HARVESTER_BLACKLISTED_EXTENSIONS):
            return False
    except IndexError:
        return False
    return True

def validate_content(url):
    try:
        payload = {'api_key': SCRAPER_API_KEY, 'url': url}
        downloaded = requests.get('http://api.scraperapi.com', params=payload, timeout=60).text
        if not downloaded:
            return False
        content = extract(downloaded, include_comments=False, include_tables=False, with_metadata=False)
        return content and len(content.split()) >= HARVESTER_MIN_CONTENT_WORDS
    except Exception:
        return False

# --- MULTI-API CONFIG & RATE LIMITING (only ScraperAPI active) ---
APIS = [
    {'name': 'ScraperAPI', 'url': 'http://api.scraperapi.com', 'key': SCRAPER_API_KEY},
    # {'name': 'AbstractAPI',  'url': 'https://web-scraping.abstractapi.com/api/v1/', 'key': ABSTRACT_API_KEY},
    # {'name': 'CritiqueLabs', 'url': 'https://api.critique-labs.ai/v1/scrape',       'key': CRITIQUE_API_KEY}
]

_next_request_time = {api['name']: 0.0 for api in APIS}
_rate_limit_interval = 60.0 / 10  # 6 seconds per API

def api_search(target_url):
    """Query only ScraperAPI for now."""
    html_responses = []
    for api in APIS:
        now = time.time()
        wait = _next_request_time[api['name']] - now
        if wait > 0:
            time.sleep(wait)
        try:
            payload = {'api_key': api['key'], 'url': target_url}
            resp = requests.get(api['url'], params=payload, timeout=60)
            resp.raise_for_status()
            html_responses.append(resp.text)
        except requests.exceptions.RequestException as e:
            print(f"[x] {api['name']} Request Failed: {e}")
        finally:
            _next_request_time[api['name']] = time.time() + _rate_limit_interval
    return html_responses

# --- SEARCH ENGINE LOGIC ---
SEARCH_ENGINES = [
    {'name': 'Bing',         'url': lambda q: f"https://www.bing.com/search?q={q}",       'link_selector': 'li.b_algo h2 a'},
    {'name': 'DuckDuckGo',   'url': lambda q: f"https://html.duckduckgo.com/html/?q={q}", 'link_selector': 'h2.result-title > a'}
]

def search_topic(topic):
    print(f"[*] Searching for '{topic}'...")
    query = topic.replace(' ', '+')
    all_links = set()

    for engine in SEARCH_ENGINES:
        target = engine['url'](query)
        html_list = api_search(target)
        for html in html_list:
            if not html:
                continue
            soup = BeautifulSoup(html, 'lxml')
            links = {a.get('href') for a in soup.select(engine['link_selector']) if a.get('href')}
            all_links.update({link for link in links if is_valid_link(link)})

    if all_links:
        print(f"[+] Found {len(all_links)} potential links for '{topic}'")
    else:
        print(f"[x] No links found for '{topic}'")
    return topic, list(all_links)

# --- FILE UTILITIES ---
def load_topics():
    topics = set()
    for f in glob.glob(os.path.join(DATA_DIR, 'topics*.txt')):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                topics.update(line.strip() for line in file if line.strip())
        except IOError as e:
            print(f"[x] Error reading topics file {f}: {e}")
    return list(topics)

def load_existing_urls():
    urls = set()
    for f in glob.glob(os.path.join(DATA_DIR, 'urls*.txt')):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                urls.update(line.strip() for line in file if line.strip())
        except IOError as e:
            print(f"[x] Error reading URL file {f}: {e}")
    return urls

def append_urls(urls_to_add):
    try:
        with open(URLS_FILE, 'a', encoding='utf-8') as f:
            for url in urls_to_add:
                f.write(url + '\n')
    except IOError as e:
        print(f"[x] Error appending to URL file {URLS_FILE}: {e}")

# --- HARVESTER ---
def run_harvester():
    print("\n--- 🚀 Harvester Starting ---")
    missing = [api['name'] for api in APIS if not api['key'] or 'YOUR_' in api['key']]
    if missing:
        print(f"[x] CRITICAL: Missing API keys for: {', '.join(missing)}. Exiting.")
        return

    topics = load_topics()
    if not topics:
        print("[x] No topics found. Please create 'topics*.txt' files.")
        return

    existing_urls = load_existing_urls()
    print(f"[*] Loaded {len(existing_urls)} existing URLs to avoid duplicates.")

    found_this_run = set()
    lock = threading.Lock()

    def job(topic):
        _, links = search_topic(topic)
        new_links = {link for link in links if link not in existing_urls and link not in found_this_run}
        if not new_links:
            return

        validated = [link for link in new_links if validate_content(link)]
        if validated:
            with lock:
                final = [u for u in validated if u not in found_this_run]
                if final:
                    append_urls(final)
                    found_this_run.update(final)
                    existing_urls.update(final)
                    print(f"[✓] {topic}: Added {len(final)} valid new URLs.")

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(job, topics)

    print(f"\n[✔] Harvester finished. Total new URLs added: {len(found_this_run)}")

if __name__ == '__main__':
    run_harvester()
