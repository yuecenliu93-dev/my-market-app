#!/usr/bin/env python3
"""
Market data aggregator for AM/WM interview preparation.

Required libraries:
    pip install feedparser requests beautifulsoup4 youtube-transcript-api PyMuPDF python-dotenv

Optional environment variables loaded from .env:
    REPORTS_DIR=/path/to/reports
    OUTPUT_FILE=/path/to/daily_context.txt
    REUTERS_BUSINESS_RSS=https://news.google.com/rss/search?q=site%3Areuters.com%20business%20markets%20when%3A7d&hl=en-US&gl=US&ceid=US:en
    GS_INSIGHTS_URL=https://feeds.megaphone.fm/GLD9218176758
    MORGAN_STANLEY_PODCAST_RSS=https://rss.art19.com/thoughts-on-the-market
    MORGAN_STANLEY_PLAYLIST_URL=https://www.youtube.com/playlist?list=PLMUnYeeTvzNsu5Z2B17k_VlzOPnB45A1n
    MAX_PODCAST_EPISODES=3
    MAX_YOUTUBE_VIDEOS=3
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import feedparser
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)


DEFAULT_REUTERS_BUSINESS_RSS = (
    "https://news.google.com/rss/search?"
    "q=site%3Areuters.com%20business%20markets%20when%3A7d"
    "&hl=en-US&gl=US&ceid=US:en"
)
YAHOO_FINANCE_RSS = "https://finance.yahoo.com/news/rssindex"
DEFAULT_MS_PLAYLIST_URL = (
    "https://www.youtube.com/playlist?list=PLMUnYeeTvzNsu5Z2B17k_VlzOPnB45A1n"
)
DEFAULT_MS_PODCAST_RSS = "https://rss.art19.com/thoughts-on-the-market"
DEFAULT_GS_INSIGHTS_URL = (
    "https://feeds.megaphone.fm/GLD9218176758"
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}


def source_block(source_name: str, body: str) -> str:
    """Format a source section for daily_context.txt."""
    clean_body = body.strip() or "No content collected."
    return f"\n\n--- SOURCE: {source_name} ---\n{clean_body}\n"


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)


def fetch_rss(feed_url: str, source_name: str, limit: int = 10) -> str:
    feed = feedparser.parse(feed_url, request_headers=REQUEST_HEADERS)
    if getattr(feed, "bozo", False):
        raise RuntimeError(f"RSS parse warning/error: {feed.bozo_exception}")

    lines: list[str] = []
    for index, entry in enumerate(feed.entries[:limit], start=1):
        title = strip_html(entry.get("title", "Untitled"))
        summary = strip_html(entry.get("summary") or entry.get("description"))
        link = entry.get("link", "")
        published = entry.get("published") or entry.get("updated") or "No timestamp"

        lines.append(f"{index}. {title}")
        lines.append(f"   Published: {published}")
        if summary:
            lines.append(f"   Summary: {summary}")
        if link:
            lines.append(f"   Link: {link}")
        lines.append("")

    return "\n".join(lines).strip() or f"No entries found for {source_name}."


def fetch_podcast_rss(feed_url: str, source_name: str, limit: int = 3) -> str:
    feed = feedparser.parse(feed_url, request_headers=REQUEST_HEADERS)
    if getattr(feed, "bozo", False):
        raise RuntimeError(f"Podcast RSS parse warning/error: {feed.bozo_exception}")

    lines: list[str] = []
    for index, entry in enumerate(feed.entries[:limit], start=1):
        title = strip_html(entry.get("title", "Untitled"))
        published = entry.get("published") or entry.get("updated") or "No timestamp"
        summary = strip_html(
            entry.get("summary")
            or entry.get("subtitle")
            or " ".join(
                content.get("value", "")
                for content in entry.get("content", [])
                if isinstance(content, dict)
            )
        )
        audio_links = [
            link.get("href", "")
            for link in entry.get("links", [])
            if link.get("rel") == "enclosure" or "audio" in link.get("type", "")
        ]

        lines.append(f"{index}. {title}")
        lines.append(f"   Published: {published}")
        if summary:
            lines.append(f"   Summary / transcript note: {summary}")
        for audio_link in audio_links[:1]:
            lines.append(f"   Audio: {audio_link}")
        lines.append("")

    return "\n".join(lines).strip() or f"No entries found for {source_name}."


def extract_json_object(text: str, marker: str) -> dict:
    marker_index = text.find(marker)
    if marker_index == -1:
        raise ValueError(f"Could not find marker: {marker}")

    start = text.find("{", marker_index)
    if start == -1:
        raise ValueError(f"Could not find JSON start after marker: {marker}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : index + 1])

    raise ValueError(f"Could not parse JSON object after marker: {marker}")


def find_video_ids_in_json(value: object) -> list[str]:
    video_ids: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            video_id = node.get("videoId")
            if isinstance(video_id, str) and len(video_id) == 11:
                video_ids.append(video_id)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return list(dict.fromkeys(video_ids))


def get_playlist_video_ids(playlist_url: str, limit: int) -> list[str]:
    response = requests.get(playlist_url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()

    try:
        initial_data = extract_json_object(response.text, "var ytInitialData =")
        video_ids = find_video_ids_in_json(initial_data)
    except Exception:
        video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', response.text)
        video_ids = list(dict.fromkeys(video_ids))

    return video_ids[:limit]


def transcript_text_for_video(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    try:
        transcript = api.fetch(video_id, languages=["en"])
    except (NoTranscriptFound, TranscriptsDisabled):
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_transcript(["en"]).fetch()

    return " ".join(item.text.replace("\n", " ") for item in transcript)


def fetch_youtube_playlist_transcripts(playlist_url: str, max_videos: int = 3) -> str:
    video_ids = get_playlist_video_ids(playlist_url, max_videos)
    if not video_ids:
        return "No videos found in playlist."

    sections: list[str] = []
    for video_id in video_ids:
        try:
            transcript = transcript_text_for_video(video_id)
            sections.append(
                f"Video: https://www.youtube.com/watch?v={video_id}\n{transcript}"
            )
        except Exception as exc:
            sections.append(
                f"Video: https://www.youtube.com/watch?v={video_id}\n"
                f"Transcript unavailable: {exc}"
            )

    return "\n\n".join(sections)


def normalize_gs_insights_url(raw_url: str, base_url: str) -> str | None:
    if not raw_url:
        return None

    clean_url = raw_url.strip().strip('"').replace("\\/", "/")
    clean_url = clean_url.replace("/content/gs/gscom", "")
    full_url = urljoin(base_url, clean_url)
    parsed = urlparse(full_url)

    if parsed.netloc and parsed.netloc != "www.goldmansachs.com":
        return None
    if not parsed.path.startswith("/insights/"):
        return None
    if parsed.path.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".svg", ".gif")):
        return None
    if parsed.fragment:
        return None

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 3:
        return None

    if path_parts[1] not in {"goldman-sachs-exchanges", "the-markets"}:
        return None

    return f"https://www.goldmansachs.com{parsed.path}"


def candidate_gs_article_links(
    soup: BeautifulSoup, base_url: str, raw_html: str = ""
) -> list[str]:
    links: list[str] = []

    patterns = [
        r'"seriesCardGridPages":"([^"]+)"',
        r'"href":"([^"]*/insights/(?:goldman-sachs-exchanges|the-markets)/[^"]+)"',
        r'(/content/gs/gscom/insights/(?:goldman-sachs-exchanges|the-markets)/[^"<>\\ ]+)',
        r'(/insights/(?:goldman-sachs-exchanges|the-markets)/[^"<>\\ ]+)',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, raw_html):
            normalized = normalize_gs_insights_url(match.group(1), base_url)
            if normalized:
                links.append(normalized)

    for anchor in soup.find_all("a", href=True):
        normalized = normalize_gs_insights_url(anchor["href"], base_url)
        if normalized:
            links.append(normalized)

    return list(dict.fromkeys(links))


def visible_page_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return html.unescape(soup.get_text("\n", strip=True))


def extract_pdf_text_from_url(pdf_url: str) -> str:
    response = requests.get(
        pdf_url,
        headers={**REQUEST_HEADERS, "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8"},
        timeout=20,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
        return ""

    lines: list[str] = []
    with fitz.open(stream=response.content, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                lines.append(f"\n[Transcript Page {page_number}]\n{page_text}")

    return "\n".join(lines).strip()


def find_transcript_link(article_soup: BeautifulSoup, article_url: str) -> str | None:
    for anchor in article_soup.find_all("a", href=True):
        label = anchor.get_text(" ", strip=True).lower()
        href = anchor["href"].lower()
        if "transcript" in label or "transcript" in href:
            return urljoin(article_url, anchor["href"])
    return None


def fetch_goldman_sachs_latest_transcript(page_url: str) -> str:
    if page_url.endswith(".xml") or "feeds.megaphone.fm" in page_url:
        return fetch_podcast_rss(page_url, "Goldman Sachs Exchanges Podcast RSS")

    response = requests.get(page_url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    links = candidate_gs_article_links(soup, page_url, response.text)
    pages_to_try = links[:10] or [page_url]

    best_page_url = page_url
    best_text = visible_page_text(soup)

    for link in pages_to_try:
        try:
            article_response = requests.get(link, headers=REQUEST_HEADERS, timeout=20)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.text, "html.parser")
            article_text = visible_page_text(article_soup)

            transcript_link = find_transcript_link(article_soup, link)
            if transcript_link:
                transcript_text = extract_pdf_text_from_url(transcript_link)
                if transcript_text:
                    return (
                        f"Page: {link}\n"
                        f"Transcript PDF: {transcript_link}\n\n"
                        f"{transcript_text}"
                    )
                return (
                    f"Page: {link}\n"
                    f"Transcript PDF link found but direct PDF extraction was blocked: "
                    f"{transcript_link}\n\n"
                    f"{article_text}"
                )

            if "transcript" in article_text.lower():
                return f"Page: {link}\n\n{article_text}"
            if len(article_text) > len(best_text):
                best_page_url = link
                best_text = article_text
        except Exception:
            continue

    return (
        f"Page: {best_page_url}\n\n"
        "No explicit transcript marker found. Collected the most relevant page text:\n\n"
        f"{best_text}"
    )


def iter_report_files(reports_dir: Path) -> Iterable[Path]:
    if not reports_dir.exists():
        return []
    return sorted(
        path
        for path in reports_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
    )


def read_pdf(path: Path) -> str:
    lines: list[str] = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                lines.append(f"\n[Page {page_number}]\n{text}")
    return "\n".join(lines).strip()


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").strip()


def scan_local_reports(reports_dir: Path) -> str:
    files = list(iter_report_files(reports_dir))
    if not files:
        return f"No .pdf or .txt files found in {reports_dir}."

    sections: list[str] = []
    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                content = read_pdf(path)
            else:
                content = read_text_file(path)
            sections.append(f"FILE: {path}\n{content or '[No readable text found]'}")
        except Exception as exc:
            sections.append(f"FILE: {path}\nCould not read file: {exc}")

    return "\n\n".join(sections)


def main() -> None:
    load_dotenv()

    base_dir = Path(__file__).resolve().parent
    reports_dir = Path(os.getenv("REPORTS_DIR", base_dir / "reports")).expanduser()
    output_file = Path(os.getenv("OUTPUT_FILE", base_dir / "daily_context.txt")).expanduser()
    ms_podcast_rss = os.getenv("MORGAN_STANLEY_PODCAST_RSS", DEFAULT_MS_PODCAST_RSS)
    ms_playlist_url = os.getenv("MORGAN_STANLEY_PLAYLIST_URL", DEFAULT_MS_PLAYLIST_URL)
    gs_insights_url = os.getenv("GS_INSIGHTS_URL", DEFAULT_GS_INSIGHTS_URL)
    reuters_business_rss = os.getenv("REUTERS_BUSINESS_RSS", DEFAULT_REUTERS_BUSINESS_RSS)
    max_podcast_episodes = int(os.getenv("MAX_PODCAST_EPISODES", "3"))
    max_youtube_videos = int(os.getenv("MAX_YOUTUBE_VIDEOS", "3"))

    blocks: list[str] = [
        source_block(
            "Run Metadata",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Reports folder: {reports_dir}",
        )
    ]

    try:
        text = fetch_rss(reuters_business_rss, "Reuters Business RSS")
        blocks.append(source_block("Reuters Business RSS", text))
        print("Successfully fetched Reuters RSS")
    except Exception as exc:
        blocks.append(source_block("Reuters Business RSS", f"ERROR: {exc}"))
        print(f"Reuters RSS failed: {exc}")

    try:
        text = fetch_rss(YAHOO_FINANCE_RSS, "Yahoo Finance RSS")
        blocks.append(source_block("Yahoo Finance RSS", text))
        print("Successfully fetched Yahoo Finance RSS")
    except Exception as exc:
        blocks.append(source_block("Yahoo Finance RSS", f"ERROR: {exc}"))
        print(f"Yahoo Finance RSS failed: {exc}")

    try:
        text = fetch_podcast_rss(
            ms_podcast_rss,
            "Morgan Stanley Thoughts on the Market Podcast RSS",
            max_podcast_episodes,
        )
        blocks.append(source_block("Morgan Stanley Thoughts on the Market Podcast", text))
        print("Successfully fetched Morgan Stanley Thoughts on the Market podcast RSS")
    except Exception as exc:
        blocks.append(
            source_block("Morgan Stanley Thoughts on the Market Podcast", f"ERROR: {exc}")
        )
        print(f"Morgan Stanley Thoughts on the Market podcast RSS failed: {exc}")

    try:
        text = fetch_youtube_playlist_transcripts(ms_playlist_url, max_youtube_videos)
        blocks.append(source_block("Morgan Stanley YouTube", text))
        print("Successfully fetched Morgan Stanley YouTube transcripts")
    except Exception as exc:
        blocks.append(source_block("Morgan Stanley YouTube", f"ERROR: {exc}"))
        print(f"Morgan Stanley YouTube transcripts failed: {exc}")

    try:
        text = fetch_goldman_sachs_latest_transcript(gs_insights_url)
        blocks.append(source_block("Goldman Sachs Insights Podcast", text))
        print("Successfully fetched Goldman Sachs Insights transcript")
    except Exception as exc:
        blocks.append(source_block("Goldman Sachs Insights Podcast", f"ERROR: {exc}"))
        print(f"Goldman Sachs Insights transcript failed: {exc}")

    try:
        text = scan_local_reports(reports_dir)
        blocks.append(source_block("Local PDF/Text Reports", text))
        print("Successfully scanned local reports")
    except Exception as exc:
        blocks.append(source_block("Local PDF/Text Reports", f"ERROR: {exc}"))
        print(f"Local report scan failed: {exc}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(blocks).strip() + "\n", encoding="utf-8")
    print(f"Successfully wrote consolidated output to {output_file}")


if __name__ == "__main__":
    main()
