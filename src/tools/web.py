import httpx
from bs4 import BeautifulSoup


MAX_CONTENT_CHARS = 8000  # Truncate fetched pages to this many characters


def web_fetch(url: str) -> str:
    """Fetch a URL and return cleaned text content.
    """
    try:
        resp = httpx.get(
            url,
            timeout=20,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for {url}"
    except httpx.RequestError as e:
        return f"Error: request failed for {url}: {e}"
    except Exception as e:
        return f"Error: unexpected error fetching {url}: {e}"

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse repeated blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)
        if len(cleaned) > MAX_CONTENT_CHARS:
            cleaned = cleaned[:MAX_CONTENT_CHARS] + "\n... [truncated]"
        return cleaned
    except Exception as e:
        return f"Error: failed to parse content from {url}: {e}"
