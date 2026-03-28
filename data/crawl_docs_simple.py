"""
Crawl LangChain docs from llms.txt and save as markdown files.

Synchronous version using only `requests` — works inside a Databricks notebook cell.
No aiohttp/aiofiles needed.

Usage (standalone):
    pip install requests
    python crawl_docs_simple.py

Usage (Databricks notebook):
    from crawl_docs_simple import crawl_to_volume
    crawl_to_volume("/Volumes/my_catalog/my_schema/my_volume/langchain-docs")
"""

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

import requests

LLMS_TXT_URL = "https://docs.langchain.com/llms.txt"
OUTPUT_DIR = Path(__file__).parent / "langchain-docs"
TIMEOUT = 15


def fetch_llms_txt(session: requests.Session) -> list[tuple[str, str]]:
    """Fetch llms.txt and extract (title, url) pairs."""
    resp = session.get(LLMS_TXT_URL, timeout=TIMEOUT)
    resp.raise_for_status()

    entries = []
    for line in resp.text.splitlines():
        match = re.match(r"^- \[(.+?)\]\((.+?)\)", line.strip())
        if match:
            title, url = match.group(1), match.group(2)
            if url.endswith(".json"):
                continue
            entries.append((title, url))
    return entries


def url_to_filename(url: str) -> str:
    """Convert URL to a safe filename."""
    parsed = urlparse(url)
    name = parsed.path.strip("/").replace("/", "__")
    name = re.sub(r"\.md$", "", name)
    if not name:
        name = hashlib.md5(url.encode()).hexdigest()[:12]
    if len(name) > 120:
        name = name[:120]
    return f"{name}.md"


def crawl_page(
    session: requests.Session,
    index: int,
    total: int,
    title: str,
    url: str,
    output_dir: Path,
) -> bool:
    """Fetch a single page and save it. Returns True on success."""
    filename = url_to_filename(url)
    filepath = output_dir / filename

    if filepath.exists():
        return True

    try:
        resp = session.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        text = resp.text.strip()
    except (requests.RequestException, ConnectionError) as e:
        print(f"[{index}/{total}] SKIP ({title[:40]}): {e}")
        return False

    if not text or len(text) < 20:
        print(f"[{index}/{total}] SKIP ({title[:40]}): empty or too short")
        return False

    safe_title = title.replace('"', '\\"')
    content = f'---\ntitle: "{safe_title}"\nsource: "{url}"\n---\n\n{text}\n'

    filepath.write_text(content, encoding="utf-8")

    if index % 50 == 0:
        print(f"[{index}/{total}] progress checkpoint — {index} files processed")

    return True


def crawl(output_dir: Path) -> None:
    """Crawl all LangChain docs to the given output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    print(f"Fetching {LLMS_TXT_URL}...")
    entries = fetch_llms_txt(session)
    total = len(entries)
    print(f"Found {total} doc pages to crawl (sequential).\n")

    success = 0
    skipped = 0

    for i, (title, url) in enumerate(entries, 1):
        if crawl_page(session, i, total, title, url, output_dir):
            success += 1
        else:
            skipped += 1

    print(f"\nDone! {success} saved, {skipped} skipped.")
    print(f"Output: {output_dir.resolve()}")

    # Write manifest
    manifest = output_dir / "_manifest.txt"
    files = sorted(output_dir.glob("*.md"))
    manifest.write_text("\n".join(f.name for f in files) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest} ({len(files)} files)")


def crawl_to_volume(volume_path: str) -> None:
    """Crawl docs and write directly to a UC Volume path.

    Call from a Databricks notebook cell:
        crawl_to_volume("/Volumes/my_catalog/my_schema/my_volume/langchain-docs")

    Uses plain file I/O since UC Volumes are FUSE-mounted on Databricks.
    """
    volume_dir = Path(volume_path)
    volume_dir.mkdir(parents=True, exist_ok=True)
    crawl(volume_dir)


if __name__ == "__main__":
    crawl(OUTPUT_DIR)
