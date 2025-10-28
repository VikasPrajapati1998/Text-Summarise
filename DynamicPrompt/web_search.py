import time
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from logger import setup_logger

logger = setup_logger("web_search", log_dir="logs", level=20)  # level=20 -> INFO by default

DUCK_SEARCH_URL = "https://duckduckgo.com/html/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

def duckduckgo_search(query: str, num_results: int = 5, pause: float = 0.2) -> List[Dict]:
    """
    Perform a DuckDuckGo HTML search and return a list of results:
    [{"title": ..., "link": ..., "snippet": ...}, ...]
    """
    logger.info("Starting DuckDuckGo search: %s", query)
    if not query or not query.strip():
        logger.warning("Empty query provided to duckduckgo_search()")
        return []

    params = {"q": query}
    try:
        resp = requests.get(DUCK_SEARCH_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.exception("HTTP request to DuckDuckGo failed: %s", e)
        raise

    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    anchors = soup.select("a.result__a")
    logger.info("Found %d result anchors (raw) on DuckDuckGo page", len(anchors))

    for a in anchors[:num_results]:
        try:
            title = a.get_text(strip=True)
            href = a.get("href")
            snippet = ""

            parent = a.find_parent()
            if parent:
                sn = parent.select_one(".result__snippet, .result__snippet--2line, .result__content .result__snippet")
                if sn:
                    snippet = sn.get_text(" ", strip=True)

            if not snippet:
                ancestor = a
                for _ in range(3):
                    ancestor = ancestor.find_parent()
                    if not ancestor:
                        break
                    sn = ancestor.select_one(".result__snippet, .result__snippet--2line, .result__content .result__snippet")
                    if sn:
                        snippet = sn.get_text(" ", strip=True)
                        break

            if not snippet:
                sib = a.find_next_sibling(text=True)
                if sib and isinstance(sib, str):
                    snippet = sib.strip()

            results.append({"title": title, "link": href, "snippet": snippet})
            logger.debug("Result added: title=%s, link=%s", title[:80], href)
            time.sleep(pause)
        except Exception as e:
            logger.exception("Error while parsing an anchor element: %s", e)
            # continue to next anchor

    logger.info("Returning %d parsed results", len(results))
    return results


def print_results(results: List[Dict]):
    """Print results to stdout in a friendly format and log the action."""
    if not results:
        print("⚠️ No results found.")
        logger.info("No results to print.")
        return

    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('title')}")
        # print(f"   {r.get('link')}")
        if r.get("snippet"):
            print(f"   {r.get('snippet')}")
        print()
    logger.info("Printed %d results to console", len(results))


if __name__ == "__main__":
    try:
        query = input("🔎 Enter your search query: ").strip()
        logger.info("User query received: %s", query if len(query) < 200 else query[:200] + "...")
        results = duckduckgo_search(query, num_results=5)
        print("\n================ Search Results ================\n")
        print_results(results)
    except requests.exceptions.HTTPError as http_err:
        logger.error("HTTP error during search: %s", http_err)
        print(f"❌ HTTP Error: {http_err}")
    except Exception as exc:
        logger.exception("Unexpected error in main: %s", exc)
        print(f"❌ Error: {exc}")
