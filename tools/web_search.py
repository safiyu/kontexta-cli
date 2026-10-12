"""
High-performance live web search and webpage content extractor.
Zero external dependencies - works purely with Python standard library (urllib, html, re).
Provides:
  - WebSearchEngine.search(query, max_results=5): Live search via DuckDuckGo + GitHub
  - WebSearchEngine.fetch_url(url, max_chars=4000): HTML-to-clean-Markdown extraction
"""

import re
import json
import html
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional

try:
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    pass

from ..terminal.ui import UI, RICH_AVAILABLE, console


class WebSearchEngine:
    """Zero-dependency live web search and webpage content extractor."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    MAX_READ_BYTES = 2_000_000
    TEXT_TYPES = ("text/", "application/xhtml", "application/xml", "application/json")

    @classmethod
    def search(cls, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a live web search using DuckDuckGo HTML endpoint.
        Returns a list of dicts with keys: 'title', 'url', 'snippet'.
        """
        if not query.strip():
            return []

        post_data = urllib.parse.urlencode({"q": query, "b": ""}).encode("utf-8")
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=post_data,
            headers={
                "User-Agent": cls.DEFAULT_USER_AGENT,
                "Referer": "https://html.duckduckgo.com/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        results = []
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read(cls.MAX_READ_BYTES).decode("utf-8", errors="replace")
            results = cls._parse_ddg(body, max_results)
        except Exception as e:
            # Fallback to GitHub Search API if general search encountered network issue
            gh_results = cls.search_github(query, max_results=max_results)
            if gh_results:
                return gh_results
            return [{"title": "Search Error", "url": "", "snippet": f"Could not perform web search: {e}"}]

        if not results:
            return cls.search_github(query, max_results=max_results)
        return results

    @classmethod
    def _parse_ddg(cls, body: str, max_results: int) -> List[Dict[str, str]]:
        """Parses DDG HTML; each snippet is matched to the title block it follows."""
        results = []
        title_matches = list(re.finditer(r'<h2 class="result__title">([\s\S]*?)</h2>', body))
        for idx, m in enumerate(title_matches):
            if len(results) >= max_results:
                break
            t_html = m.group(1)
            block_end = title_matches[idx + 1].start() if idx + 1 < len(title_matches) else len(body)
            m_snip = re.search(r'<a class="result__snippet"[^>]*>([\s\S]*?)</a>', body[m.end():block_end])
            m_url = re.search(r'href="([^"]+)"', t_html)
            actual_url = m_url.group(1) if m_url else ""
            if "uddg=" in actual_url:
                actual_url = urllib.parse.unquote(actual_url.split("uddg=")[-1].split("&")[0])
            title = html.unescape(re.sub(r"<[^>]+>", "", t_html).strip())
            snippet = html.unescape(re.sub(r"<[^>]+>", "", m_snip.group(1)).strip()) if m_snip else ""
            if actual_url and title:
                results.append({"title": title, "url": actual_url, "snippet": snippet})
        return results

    @classmethod
    def search_github(cls, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Searches GitHub public repositories for libraries, issues, and code."""
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote_plus(query)}&sort=stars&order=desc&per_page={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "myagy/1.0"})
        results = []
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("items", [])[:max_results]:
                results.append({
                    "title": f"GitHub: {item.get('full_name')} ({item.get('stargazers_count', 0)}★)",
                    "url": item.get("html_url", ""),
                    "snippet": item.get("description") or "(No description provided)",
                })
        except Exception:
            pass
        return results

    @classmethod
    def fetch_url(cls, url: str, max_chars: int = 4000) -> str:
        """
        Fetches a web page and converts the HTML into clean, human-readable markdown text.
        Strips script, style, navigation headers, and footer noise.
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": cls.DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                content_type = resp.headers.get("Content-Type", "").lower()
                if content_type and not content_type.startswith(cls.TEXT_TYPES):
                    return f"Error fetching {url}: unsupported content type '{content_type.split(';')[0]}'"
                raw = resp.read(cls.MAX_READ_BYTES).decode("utf-8", errors="replace")

            # Remove scripts, styles, svg, and metadata tags
            text = re.sub(r"<(script|style|noscript|svg|nav|footer|header)[\s\S]*?</\1>", " ", raw, flags=re.I)
            # Convert headings to markdown
            text = re.sub(r"<h1[^>]*>([\s\S]*?)</h1>", r"\n# \1\n", text, flags=re.I)
            text = re.sub(r"<h2[^>]*>([\s\S]*?)</h2>", r"\n## \1\n", text, flags=re.I)
            text = re.sub(r"<h3[^>]*>([\s\S]*?)</h3>", r"\n### \1\n", text, flags=re.I)
            # Convert links and list items
            text = re.sub(r"<li[^>]*>([\s\S]*?)</li>", r"\n* \1", text, flags=re.I)
            text = re.sub(r"<p[^>]*>([\s\S]*?)</p>", r"\n\1\n", text, flags=re.I)
            # Remove remaining tags
            text = re.sub(r"<[^>]+>", " ", text)
            # Decode HTML entities
            text = html.unescape(text)
            # Normalize whitespace
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            clean_text = "\n".join(lines)

            if len(clean_text) > max_chars:
                return clean_text[:max_chars] + f"\n\n... [Content truncated at {max_chars} characters]"
            return clean_text or "(No textual content found at this URL)"

        except Exception as e:
            return f"Error fetching {url}: {e}"

    @classmethod
    def print_search_results(cls, query: str, results: List[Dict[str, str]]):
        """Displays search results in a clean formatted table."""
        if not results:
            print(UI.warn(f"No web search results found for '{query}'."))
            return

        if RICH_AVAILABLE:
            table = Table(
                title=Text.from_ansi(f"{UI.WHITE}WEB SEARCH RESULTS{UI.RST} │ Query: {UI.CYAN}{query}{UI.RST}"),
                box=box.ROUNDED,
                header_style="bold cyan",
                border_style="bright_black",
            )
            table.add_column("#", justify="center", style="bold dim", width=4)
            table.add_column("Title & URL", style="white", width=40)
            table.add_column("Snippet", style="dim")

            for i, r in enumerate(results, 1):
                title_url = f"{UI.WHITE}{r['title']}{UI.RST}\n{UI.DARK_GRAY}{r['url']}{UI.RST}"
                table.add_row(str(i), Text.from_ansi(title_url), r["snippet"])

            console.print()
            console.print(table)
            console.print(f"{UI.GRAY}Tip: Use '/fetch <url>' to view complete webpage content in markdown.{UI.RST}\n")
        else:
            print(f"\n{UI.DARK_GRAY}╭─── {UI.WHITE}WEB SEARCH RESULTS ({len(results)} hits){UI.RST}{UI.DARK_GRAY} ─────────────────────────────╮{UI.RST}")
            for i, r in enumerate(results, 1):
                print(f"{UI.DARK_GRAY}│{UI.RST}  {i}. {UI.WHITE}{r['title']}{UI.RST}")
                print(f"{UI.DARK_GRAY}│{UI.RST}     URL: {UI.CYAN}{r['url']}{UI.RST}")
                print(f"{UI.DARK_GRAY}│{UI.RST}     {UI.GRAY}{r['snippet'][:100]}...{UI.RST}")
            print(f"{UI.DARK_GRAY}╰────────────────────────────────────────────────────────────────────────╯\n")
