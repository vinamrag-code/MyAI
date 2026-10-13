#!/usr/bin/env python3
"""DDGS.py — standalone web search + page-text extractor (adapted from g4f, zero g4f deps).

A retrieval tool rather than a chat model: it searches the web (DuckDuckGo via the
`ddgs` package), then optionally scrapes each result's readable text so the output
can be fed straight into DeepAI.py / OperaAria.py / Yqcloud.py.

Requires: `pip install ddgs beautifulsoup4`

Usage:
    python3 DDGS.py "who won the 2024 Nobel prize in physics"   # search + scrape
    python3 DDGS.py "python asyncio tutorial" --no-scrape       # results only, fast
    python3 DDGS.py "capital of Australia" --results 3 --words 400
    python3 DDGS.py "news" --region uk-en --json               # raw JSON
    python3 DDGS.py "capital of australia" --pipe               # text straight into DeepAI
    python3 DDGS.py --test                                     # search-relevance battery
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    sys.exit("ddgs is required: pip install ddgs beautifulsoup4")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")
TIMEOUT = 30
CACHE_DIR = Path.home() / ".cache" / "ddgs_scrape"

# Chrome/nav junk that otherwise floods extracted text (Wikipedia's language list,
# "jump to navigation" boxes, edit links...).
JUNK_SELECTORS = [
    "nav", "footer", "script", "style", "noscript", "aside", "form", "button",
    ".c-globalDisclosure", ".navbox", ".sidebar", ".mw-editsection", ".portal",
    ".vector", ".mw-jump", ".thumb", ".infobox", ".reference", ".reflist",
    ".language-list", "#p-collapsible-navigation", ".hatnote", ".shortdescription",
    "[role=navigation]", "[role=banner]", "[role=contentinfo]",
]

# Containers that hold the actual article, best first.
CONTENT_SELECTORS = [
    "article", "main", "#mw-content-text", ".mw-parser-output",
    "#content", ".main-content", ".main-content-wrapper", ".post-content",
    ".entry-content", "#mainContent", "body",
]


def _clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for sel in JUNK_SELECTORS:
        try:
            for el in soup.select(sel):
                el.decompose()
        except Exception:
            pass
    return soup


def scrape_text(html: str, max_words: int | None = None, add_source: bool = True) -> str:
    """Extract readable text from an HTML page."""
    soup = _clean_soup(html)

    for sel in CONTENT_SELECTORS:
        picked = soup.select_one(sel)
        if picked:
            soup = picked
            break

    image_sel = "img[alt][src^=http]:not([alt='']):not(.avatar):not([width])"
    out: list[str] = []
    seen: set[str] = set()
    words_used = 0
    images_left = 2

    for el in soup.select(f"h1, h2, h3, h4, h5, h6, p, pre, table:not(:has(p)), "
                          f"ul:not(:has(p)), ol:not(:has(p)), blockquote, a:has({image_sel})"):
        if images_left > 0:
            img = el.select_one(image_sel)
            if img:
                title = str(el.get("title", el.text)).strip()
                if title and title not in seen:
                    out.append(f"![{title}]({img['src']})\n")
                    seen.add(title)
                    images_left -= 1
                continue

        # headings keep their own line for readability
        is_heading = el.name and el.name.startswith("h")
        for raw_line in el.get_text(" ").splitlines():
            words = [w for w in raw_line.split() if w]
            if not words:
                continue
            line = " ".join(words)
            if line in seen:
                continue
            if max_words is not None:
                if words_used + len(words) > max_words:
                    break
                words_used += len(words)
            seen.add(line)
            prefix = "#" * (int(el.name[1]) if is_heading and el.name[1:].isdigit() else 0)
            out.append(f"{prefix} {line}\n".lstrip())
        if max_words is not None and words_used >= max_words:
            break

    text = "".join(out)
    if add_source:
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            link = canonical["href"]
            text += f"\nSource: [{urlparse(link).netloc}]({link})"
    return text.strip()


def _cache_path(url: str) -> Path:
    host = url.split("?")[0].split("//")[-1].replace("/", " ")[:48]
    digest = hashlib.md5(url.encode(errors="ignore")).hexdigest()[:16]
    return CACHE_DIR / f"{quote_plus(host)}.{date.today()}.{digest}.cache"


def fetch_and_scrape(url: str, max_words: int | None = None,
                     use_cache: bool = True, proxy: str | None = None) -> str:
    """Fetch a URL and scrape its text, with an on-disk daily cache."""
    path = _cache_path(url)
    if use_cache and path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            pass

    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        r = requests.get(url, headers={"User-Agent": UA, "accept-language": "en-US,en;q=0.9"},
                         timeout=TIMEOUT, proxies=proxies)
        if r.status_code != 200:
            return ""
        text = scrape_text(r.text, max_words)
    except requests.RequestException:
        return ""

    if text:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return text


def search(query: str, max_results: int = 5, region: str | None = None,
           safesearch: str = "moderate", timelimit: str | None = "y",
           backend: str | None = None, proxy: str | None = None) -> list[dict]:
    """Run a web search, returning [{title, url, snippet}]."""
    with DDGS(proxy=proxy) as ddgs:
        kwargs = dict(region=region, safesearch=safesearch,
                      timelimit=timelimit, max_results=max_results, backend=backend)
        raw = list(ddgs.text(query, **{k: v for k, v in kwargs.items() if v is not None}))

    out = []
    for r in raw:
        href = r.get("href", "")
        # g4f drops google SERP links; they are never real result pages
        if ".google." in href or not href.startswith(("http://", "https://")):
            continue
        out.append({"title": r.get("title", ""), "url": href, "snippet": r.get("body", "")})
    return out


def ask(query: str, max_results: int = 5, max_words: int = 2500, add_text: bool = True,
        region: str | None = None, backend: str | None = None, use_cache: bool = True,
        verbose: bool = False) -> tuple[list[dict], float]:
    """Search, then optionally scrape each result. Returns (results, elapsed).

    Each result gains a "text" key when scraped.
    """
    t0 = time.time()
    results = search(query, max_results=max_results, region=region, backend=backend)
    if not add_text or not results:
        return results, time.time() - t0

    per_result = max(1, max_words // max(1, len(results)))
    kept, words_used = [], 0
    for entry in results:
        if verbose:
            print(f"  [scraping] {entry['url'][:80]}", file=sys.stderr)
        entry["text"] = fetch_and_scrape(entry["url"], per_result, use_cache=use_cache)
        words_used += len(entry["title"].split()) + len(entry["text"].split())
        kept.append(entry)
        if words_used >= max_words:
            break
    return kept, time.time() - t0


def format_markdown(results: list[dict]) -> str:
    """Render results as markdown, ready to paste into a chat provider."""
    blocks = []
    for i, r in enumerate(results):
        body = (r.get("text") or r.get("snippet") or "").strip()
        if body:
            body = re.sub(r"\n{3,}", "\n\n", body)
        blocks.append(f"### {i+1}. {r['title']}\n\n{body}\n\n> **Source:** [{i+1}]({r['url']})")
    return "\n\n---\n\n".join(blocks)


BATTERY = [
    ("Capital lookup", "What is the capital of Australia?", "Canberra"),
    ("Nobel prize", "Who won the Nobel Prize in Physics in 2024?", "Hopfield"),
    ("Python async", "python asyncio documentation tutorial", "asyncio"),
    ("Current news", "latest technology news headlines", ""),
    ("Numeric fact", "population of Tokyo metropolitan area", "million"),
]


def run_test(max_results: int) -> None:
    print(f"DDGS search battery — {len(BATTERY)} queries "
          f"(results={max_results}, scrape on)\n" + "-" * 60)
    rows = []
    for label, q, expect in BATTERY:
        print(f"\n### {label}\nQ: {q}")
        results, elapsed = ask(q, max_results=max_results, verbose=False)
        hit = "n/a" if not expect else ("YES" if expect.lower() in
                                        format_markdown(results).lower() else "no")
        print(f"   >> {len(results)} result(s) | {elapsed:.1f}s | "
              f"scraped={sum(1 for r in results if r.get('text'))} | expects {expect!r}: {hit}")
        for r in results[:3]:
            print(f"      - {r['title'][:60]}  ({len((r.get('text') or '').split())} words)")
        rows.append((label, len(results), hit))

    print("\n" + "=" * 60)
    print(f"{'Query':<18}{'Results':<10}{'Expected term found'}")
    for label, n, hit in rows:
        print(f"{label:<18}{n:<10}{hit}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone web search + page-text extractor")
    ap.add_argument("query", nargs="?", help="Search query")
    ap.add_argument("--results", type=int, default=5, help="Max results (default: 5)")
    ap.add_argument("--words", type=int, default=2500, help="Max scraped words (default: 2500)")
    ap.add_argument("--region", help="DDGS region code, e.g. us-en, uk-en, wt-wt")
    ap.add_argument("--backend", help="Force a specific DDGS backend")
    ap.add_argument("--no-scrape", action="store_true", help="Return search results only (fast)")
    ap.add_argument("--no-cache", action="store_true", help="Ignore the scrape cache")
    ap.add_argument("--json", action="store_true", help="Emit raw JSON")
    ap.add_argument("--pipe", nargs="?", const="DeepAI.py", default=None, metavar="SCRIPT",
                    help="Pipe the scraped corpus into another wrapper for a written answer "
                         "(default: DeepAI.py; pass e.g. --pipe Yqcloud.py)")
    ap.add_argument("--proxy", help="http(s) proxy URL")
    ap.add_argument("--test", action="store_true", help="Run the search battery")
    args = ap.parse_args()

    if not args.test and not args.query:
        ap.error("query required (or --test)")

    if args.test:
        run_test(args.results)
        return

    t0 = time.time()
    results, elapsed = ask(args.query, max_results=args.results, max_words=args.words,
                           add_text=not args.no_scrape, region=args.region,
                           backend=args.backend, use_cache=not args.no_cache)

    if not results:
        print(f"ERROR: no results for {args.query!r}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    corpus = format_markdown(results)
    scraped = sum(1 for r in results if r.get("text"))

    if args.pipe:
        import subprocess
        # Caller-named target so this file stays independent of its siblings.
        target = Path(args.pipe)
        if not target.is_file():
            target = Path(__file__).parent / args.pipe
        if not target.is_file():
            print(f"ERROR: no such wrapper script: {args.pipe}", file=sys.stderr)
            sys.exit(1)
        question = (f"Using only the research below, answer: {args.query}\n\n"
                    f"--- RESEARCH ---\n{corpus[:12000]}")
        print(f"[piping {len(corpus)} chars into {target.name}]", file=sys.stderr)
        sys.exit(subprocess.call([sys.executable, str(target), question]))

    print(corpus)
    print(f"\n--- {len(results)} result(s), {scraped} scraped, "
          f"{len(corpus.split())} words, {time.time()-t0:.1f}s ---")


if __name__ == "__main__":
    main()