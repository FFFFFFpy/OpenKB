"""Experimental browser + LLM webpage article exporter.

This script is intentionally standalone. It is not wired into the OpenKB import
pipeline and does not depend on any OpenKB internals.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
MAX_LLM_INPUT_CHARS = 110_000
MAX_CHUNK_CHARS = 55_000
MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"}
IMAGE_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/bmp": ".bmp",
    "image/avif": ".avif",
}
VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-m4v": ".m4v",
}

SYSTEM_PROMPT = """你是一个严谨的网页文章 Markdown 编排器。

我会给你一个网页文章的可见内容快照。快照中包含按页面顺序排列的文本块、图片、视频、链接、列表、表格和其他媒体信息。

你的任务是像人类阅读网页一样理解这篇文章，然后输出一个结构合理、语义清晰的 Markdown 文档。

重要要求：

1. 只输出 Markdown，不要解释，不要使用代码块包裹。
2. 不要杜撰网页中没有的信息。
3. 不要改变原文含义。
4. 不要把网页 DOM 标签机械转换成 Markdown。
5. 不要机械依据 h1/h2/h3、字号、加粗、短句、竖线、emoji、空行来决定标题。
6. 你需要根据语义判断：
   - 哪个是文章标题
   - 哪些是正文段落
   - 哪些是章节标题
   - 哪些只是强调句
   - 哪些是引用
   - 哪些是列表
   - 图片应该插入在哪些段落附近
   - 视频应该插入在哪些段落附近
7. 如果文章没有明显章节，就不要强行拆出很多 `##`。
8. 只有语义上进入新的大主题时，才使用 `##`。
9. 只有二级主题下确有子主题时，才使用 `###`。
10. 保留原文叙事顺序。
11. 保留重要图片，并使用输入中提供的本地相对路径。
12. 保留重要视频，并使用输入中提供的视频 Markdown 或 HTML 表示。
13. 对广告、二维码、关注公众号、相关推荐、阅读原文等非正文内容，可以放到文末“附：页面附带信息”，
也可以省略明显噪声，但不要误删正文内容。
14. 如果不确定某个短句是不是标题，优先当作普通段落或强调句，而不是标题。
15. 输出必须包含 YAML frontmatter。

输出格式：

---
title: "文章标题"
source_url: "原始链接"
author: "作者或账号名"
published_at: "发布时间"
converted_by: "browser_llm_article_to_markdown"
---

# 文章标题

正文内容...

![图片说明](assets/image_001.jpg)

<video controls src="assets/video_001.mp4"></video>"""

EXTRACT_SNAPSHOT_JS = r"""
() => {
  const now = new Date().toISOString();
  const cleanText = (value, limit = 12000) => {
    if (!value) return "";
    const text = String(value).replace(/\s+/g, " ").trim();
    return text.length > limit ? text.slice(0, limit) + "..." : text;
  };
  const cssPath = (el) => {
    if (!el || !el.tagName) return "";
    if (el.id) return "#" + el.id;
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === Node.ELEMENT_NODE && parts.length < 5) {
      let part = cur.tagName.toLowerCase();
      if (cur.className && typeof cur.className === "string") {
        const cls = cur.className.trim().split(/\s+/).slice(0, 2).join(".");
        if (cls) part += "." + cls;
      }
      parts.unshift(part);
      cur = cur.parentElement;
    }
    return parts.join(" > ");
  };
  const absoluteUrl = (value) => {
    if (!value) return "";
    try {
      return new URL(value, document.baseURI).href;
    } catch {
      return value;
    }
  };
  const firstSrcset = (srcset) => {
    if (!srcset) return "";
    const first = String(srcset).split(",")[0].trim().split(/\s+/)[0];
    return first || "";
  };
  const firstBackgroundUrl = (backgroundImage) => {
    if (!backgroundImage || backgroundImage === "none") return "";
    const match = String(backgroundImage).match(/url\((['"]?)(.*?)\1\)/);
    return match ? match[2] : "";
  };
  const rectFor = (el) => {
    const rect = el.getBoundingClientRect();
    return {
      x: Math.round(rect.x),
      y: Math.round(rect.y + window.scrollY),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    };
  };
  const styleFor = (el) => {
    const style = window.getComputedStyle(el);
    return {
      display: style.display,
      visibility: style.visibility,
      opacity: style.opacity,
      fontSize: style.fontSize,
      fontWeight: style.fontWeight,
      fontStyle: style.fontStyle,
      lineHeight: style.lineHeight,
      textAlign: style.textAlign,
    };
  };
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return (
      style.display !== "none" &&
      style.visibility !== "hidden" &&
      style.opacity !== "0" &&
      rect.width > 0 &&
      rect.height > 0
    );
  };
  const nearbyText = (el) => {
    const parent = el.closest("figure, section, article, p, div, li, td, body");
    return cleanText(parent ? parent.innerText || parent.textContent : "", 700);
  };
  const metadataText = (selector) => {
    const el = document.querySelector(selector);
    return el ? cleanText(el.innerText || el.textContent, 500) : "";
  };
  const metaContent = (selector) => {
    const el = document.querySelector(selector);
    return el ? cleanText(el.getAttribute("content"), 500) : "";
  };
  const rootCandidates = [
    ["#js_content", document.querySelector("#js_content")],
    ["article", document.querySelector("article")],
    ["main", document.querySelector("main")],
    ['[role="main"]', document.querySelector('[role="main"]')],
    ["body", document.body],
  ].filter((item) => item[1]);
  const rootEntry = rootCandidates[0];
  const rootSelector = rootEntry ? rootEntry[0] : "body";
  const root = rootEntry ? rootEntry[1] : document.body;

  const metadata = {
    title:
      metadataText("#activity-name") ||
      metaContent('meta[property="og:title"]') ||
      cleanText(document.title, 500),
    author:
      metadataText("#js_name") ||
      metaContent('meta[name="author"]') ||
      metaContent('meta[property="article:author"]'),
    published_at:
      metadataText("#publish_time") ||
      metaContent('meta[property="article:published_time"]') ||
      metaContent('meta[name="publishdate"]') ||
      metaContent('meta[name="date"]'),
  };
  const hardTextTags = new Set([
    "H1", "H2", "H3", "H4", "H5", "H6", "P", "LI", "FIGCAPTION",
    "CAPTION", "PRE", "TD", "TH",
  ]);
  const softTextTags = new Set(["DIV", "SECTION", "ARTICLE"]);
  const blockishTags = new Set([
    ...hardTextTags,
    "DIV", "SECTION", "ARTICLE", "TABLE", "UL", "OL", "BLOCKQUOTE",
  ]);
  const hasChildBlockText = (el) => {
    for (const child of el.querySelectorAll("*")) {
      if (!blockishTags.has(child.tagName) || !isVisible(child)) continue;
      if (cleanText(child.innerText || child.textContent, 300).length > 0) return true;
    }
    return false;
  };
  const blocks = [];
  const pushBlock = (el, data) => {
    const style = styleFor(el);
    const rect = rectFor(el);
    const visible = isVisible(el);
    blocks.push({
      index: blocks.length + 1,
      type: data.type,
      tag: el.tagName.toLowerCase(),
      selector: cssPath(el),
      text: data.text || "",
      href: data.href || "",
      src: data.src || "",
      data_src: data.data_src || "",
      current_src: data.current_src || "",
      poster: data.poster || "",
      alt: data.alt || "",
      title: data.title || "",
      aria_label: data.aria_label || "",
      width: data.width || rect.width,
      height: data.height || rect.height,
      visible,
      low_confidence: !visible || data.low_confidence || false,
      rect,
      style,
      nearby_text: data.nearby_text || nearbyText(el),
    });
  };

  const elements = [root, ...Array.from(root.querySelectorAll("*"))];
  for (const el of elements) {
    const tag = el.tagName;
    if (["SCRIPT", "STYLE", "NOSCRIPT", "SVG"].includes(tag)) continue;
    const style = window.getComputedStyle(el);
    const bgUrl = firstBackgroundUrl(style.backgroundImage);
    if (tag === "IMG" || bgUrl) {
      const src =
        el.currentSrc ||
        el.getAttribute("src") ||
        el.getAttribute("data-src") ||
        el.getAttribute("data-original") ||
        el.getAttribute("data-backsrc") ||
        firstSrcset(el.getAttribute("srcset")) ||
        bgUrl;
      if (src) {
        pushBlock(el, {
          type: "image",
          src: absoluteUrl(src),
          data_src: absoluteUrl(el.getAttribute("data-src") || ""),
          current_src: absoluteUrl(el.currentSrc || ""),
          alt: cleanText(el.getAttribute("alt"), 1000),
          title: cleanText(el.getAttribute("title"), 1000),
          aria_label: cleanText(el.getAttribute("aria-label"), 1000),
          width: Number(el.getAttribute("width")) || 0,
          height: Number(el.getAttribute("height")) || 0,
        });
      }
      continue;
    }
    if (tag === "VIDEO") {
      const source = el.querySelector("source[src]");
      const src = el.currentSrc || el.getAttribute("src") || (source && source.src) || "";
      pushBlock(el, {
        type: "video",
        src: absoluteUrl(src),
        current_src: absoluteUrl(el.currentSrc || ""),
        poster: absoluteUrl(el.getAttribute("poster") || ""),
        title: cleanText(el.getAttribute("title"), 1000),
        aria_label: cleanText(el.getAttribute("aria-label"), 1000),
      });
      continue;
    }
    if (tag === "IFRAME") {
      const src = el.getAttribute("src") || el.getAttribute("data-src") || "";
      const videoLike =
        /video|player|bilibili|youtube|youtu\.be|v\.qq|txp|douyin|xiaohongshu|mp\.weixin/i
          .test(src);
      pushBlock(el, {
        type: videoLike ? "video" : "iframe",
        src: absoluteUrl(src),
        data_src: absoluteUrl(el.getAttribute("data-src") || ""),
        title: cleanText(el.getAttribute("title"), 1000),
        aria_label: cleanText(el.getAttribute("aria-label"), 1000),
      });
      continue;
    }
    if (tag === "AUDIO") {
      const source = el.querySelector("source[src]");
      const src = el.currentSrc || el.getAttribute("src") || (source && source.src) || "";
      pushBlock(el, {
        type: "audio",
        src: absoluteUrl(src),
        current_src: absoluteUrl(el.currentSrc || ""),
        title: cleanText(el.getAttribute("title"), 1000),
        aria_label: cleanText(el.getAttribute("aria-label"), 1000),
      });
      continue;
    }
    if (tag === "TABLE") {
      pushBlock(el, {type: "table", text: cleanText(el.innerText || el.textContent)});
      continue;
    }
    if (tag === "UL" || tag === "OL") {
      pushBlock(el, {type: "list", text: cleanText(el.innerText || el.textContent)});
      continue;
    }
    if (tag === "BLOCKQUOTE") {
      pushBlock(el, {type: "quote", text: cleanText(el.innerText || el.textContent)});
      continue;
    }
    if (tag === "A" && el.href && cleanText(el.innerText || el.textContent, 1000)) {
      const parent = el.parentElement;
      const parentTag = parent ? parent.tagName : "";
      if (!hardTextTags.has(parentTag)) {
        pushBlock(el, {
          type: "link",
          text: cleanText(el.innerText || el.textContent, 4000),
          href: absoluteUrl(el.getAttribute("href") || ""),
        });
      }
      continue;
    }
    const text = cleanText(el.innerText || el.textContent);
    if (!text) continue;
    if (hardTextTags.has(tag) || (softTextTags.has(tag) && !hasChildBlockText(el))) {
      pushBlock(el, {type: "text", text});
    }
  }

  return {
    url: window.location.href,
    title: cleanText(document.title, 500),
    captured_at: now,
    root_selector: rootSelector,
    metadata_candidates: metadata,
    page_text: cleanText(document.body ? document.body.innerText : "", 200000),
    blocks,
  };
}
"""


@dataclass
class Config:
    url: str
    out_dir: Path
    browser: str
    browser_executable: Path | None
    model: str | None
    headless: bool
    timeout: int
    save_screenshot: bool
    max_media: int
    no_video: bool


@dataclass
class Capture:
    snapshot: dict[str, Any]
    source_html: str
    screenshot: bytes | None
    browser_used: str
    warnings: list[str] = field(default_factory=list)
    cookies: list[dict[str, Any]] = field(default_factory=list)


class BrowserUnavailable(RuntimeError):
    """Raised when a requested browser runtime cannot be imported or launched."""


def parse_args(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Export a rendered webpage article to Markdown + assets using a browser and LLM."
        ),
    )
    parser.add_argument("--url", required=True, help="Article URL to capture.")
    parser.add_argument("--out-dir", default="./browser_llm_out", help="Output root directory.")
    parser.add_argument(
        "--browser",
        choices=["camoufox", "playwright"],
        default="camoufox",
        help="Browser backend. Camoufox falls back to Playwright when unavailable.",
    )
    parser.add_argument(
        "--browser-executable",
        default=os.getenv("CAMOUFOX_EXECUTABLE") or None,
        help=(
            "Optional browser executable path. Useful for a local Camoufox binary; "
            "requires Playwright or Camoufox Python automation support."
        ),
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"), help="LLM model name.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the browser in headless mode.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Page load timeout in seconds.")
    parser.add_argument(
        "--save-screenshot",
        action="store_true",
        help="Save a full-page screenshot for debugging.",
    )
    parser.add_argument("--max-media", type=int, default=200, help="Maximum media files to fetch.")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video downloads and save only video metadata/posters.",
    )
    args = parser.parse_args(argv)
    return Config(
        url=args.url,
        out_dir=Path(args.out_dir),
        browser=args.browser,
        browser_executable=Path(args.browser_executable) if args.browser_executable else None,
        model=args.model,
        headless=args.headless,
        timeout=args.timeout,
        save_screenshot=args.save_screenshot,
        max_media=max(args.max_media, 0),
        no_video=args.no_video,
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, data: str) -> None:
    path.write_text(data, encoding="utf-8", newline="\n")


def make_slug(title: str, url: str) -> str:
    raw = title.strip() or urllib.parse.urlparse(url).path.strip("/") or "article"
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw)
    slug = re.sub(r"-+", "-", slug).strip("-")[:64] or "article"
    suffix = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii")[:8].rstrip("=")
    return f"{slug}-{suffix}"


def extension_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    ext = Path(path).suffix.lower()
    return ext if ext and len(ext) <= 8 else ""


def extension_from_content_type(content_type: str, default: str) -> str:
    media_type = content_type.split(";")[0].strip().lower()
    if media_type in IMAGE_CONTENT_TYPES:
        return IMAGE_CONTENT_TYPES[media_type]
    if media_type in VIDEO_CONTENT_TYPES:
        return VIDEO_CONTENT_TYPES[media_type]
    guessed = mimetypes.guess_extension(media_type)
    return guessed or default


def cookie_header(cookies: list[dict[str, Any]]) -> str:
    pairs = []
    for cookie in cookies:
        name = cookie.get("name")
        value = cookie.get("value")
        if name and value is not None:
            pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def decode_data_uri(data_uri: str, dest_stem: Path, expected_kind: str) -> Path:
    header, payload = data_uri.split(",", 1)
    content_type = header.split(";", 1)[0].replace("data:", "") or f"{expected_kind}/octet-stream"
    ext = extension_from_content_type(content_type, ".bin")
    dest = dest_stem.with_suffix(ext)
    if ";base64" in header:
        data = base64.b64decode(payload)
    else:
        data = urllib.parse.unquote_to_bytes(payload)
    dest.write_bytes(data)
    return dest


def download_url(
    url: str,
    dest_stem: Path,
    *,
    expected_kind: str,
    referer: str,
    cookies: list[dict[str, Any]],
) -> tuple[Path | None, str]:
    if not url:
        return None, "missing_url"
    if url.startswith("data:"):
        return decode_data_uri(url, dest_stem, expected_kind), "downloaded"

    headers = {"User-Agent": USER_AGENT, "Referer": referer}
    cookie_value = cookie_header(cookies)
    if cookie_value:
        headers["Cookie"] = cookie_value
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            content_type = response.headers.get("Content-Type", "")
            media_type = content_type.split(";")[0].strip().lower()
            url_ext = extension_from_url(response.geturl()) or extension_from_url(url)
            if expected_kind == "image" and not media_type.startswith("image/") and not url_ext:
                return None, f"not_image:{content_type or 'unknown'}"
            if expected_kind == "video":
                is_video = media_type.startswith("video/") or url_ext in VIDEO_EXTENSIONS
                if not is_video:
                    return None, f"not_direct_video:{content_type or 'unknown'}"
            ext = url_ext or extension_from_content_type(content_type, ".bin")
            dest = dest_stem.with_suffix(ext)
            total = 0
            with dest.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        handle.close()
                        dest.unlink(missing_ok=True)
                        return None, "too_large"
                    handle.write(chunk)
            return dest, "downloaded"
    except urllib.error.HTTPError as exc:
        return None, f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, f"download_error:{exc.__class__.__name__}"


async def capture_with_page(page: Any, context: Any, config: Config, browser_used: str) -> Capture:
    timeout_ms = config.timeout * 1000
    await page.goto(config.url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 20_000))
    except Exception:
        pass
    await page.wait_for_timeout(1500)
    await page.evaluate(
        """async () => {
          const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
          const maxY = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
          for (let y = 0; y < maxY; y += Math.max(window.innerHeight * 0.8, 600)) {
            window.scrollTo(0, y);
            await delay(250);
          }
          window.scrollTo(0, 0);
          await delay(500);
        }"""
    )
    snapshot = await page.evaluate(EXTRACT_SNAPSHOT_JS)
    snapshot["browser"] = browser_used
    source_html = await page.content()
    screenshot = await page.screenshot(full_page=True) if config.save_screenshot else None
    cookies = await context.cookies()
    return Capture(
        snapshot=snapshot,
        source_html=source_html,
        screenshot=screenshot,
        browser_used=browser_used,
        cookies=cookies,
    )


async def capture_playwright(config: Config) -> Capture:
    try:
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as exc:
        raise BrowserUnavailable("Playwright is not installed.") from exc
    try:
        async with async_playwright() as playwright:
            launch_kwargs: dict[str, Any] = {"headless": config.headless}
            browser_type = playwright.chromium
            if config.browser_executable:
                launch_kwargs["executable_path"] = str(config.browser_executable)
                executable_name = config.browser_executable.name.lower()
                if "camoufox" in executable_name or "firefox" in executable_name:
                    browser_type = playwright.firefox
            browser = await browser_type.launch(**launch_kwargs)
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1280, "height": 900},
            )
            page = await context.new_page()
            try:
                return await capture_with_page(page, context, config, "playwright")
            finally:
                await context.close()
                await browser.close()
    except BrowserUnavailable:
        raise
    except Exception as exc:
        message = str(exc)
        if "Executable doesn't exist" in message or "browserType.launch" in message:
            raise BrowserUnavailable(
                "Playwright browser runtime is not installed. "
                "Run: python -m playwright install chromium"
            ) from exc
        raise


async def capture_camoufox(config: Config) -> Capture:
    try:
        from camoufox.async_api import AsyncCamoufox
    except ModuleNotFoundError as exc:
        raise BrowserUnavailable("Camoufox is not installed.") from exc
    launched = False
    try:
        async with AsyncCamoufox(headless=config.headless) as browser:
            launched = True
            context = await browser.new_context(
                user_agent=USER_AGENT,
                viewport={"width": 1280, "height": 900},
            )
            page = await context.new_page()
            try:
                return await capture_with_page(page, context, config, "camoufox")
            finally:
                await context.close()
    except BrowserUnavailable:
        raise
    except Exception as exc:
        if launched:
            raise
        raise BrowserUnavailable(f"Camoufox could not be launched: {exc}") from exc


async def capture_browser(config: Config) -> Capture:
    if config.browser == "playwright":
        return await capture_playwright(config)
    try:
        return await capture_camoufox(config)
    except BrowserUnavailable as exc:
        logging.warning("%s Falling back to Playwright automation.", exc)
        capture = await capture_playwright(config)
        capture.warnings.append(f"Camoufox unavailable; used Playwright fallback: {exc}")
        return capture


def add_asset(
    manifest: dict[str, Any],
    block: dict[str, Any],
    *,
    asset_id: str,
    asset_type: str,
    original_url: str,
    local_path: str | None,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    item = {
        "id": asset_id,
        "type": asset_type,
        "original_url": original_url,
        "local_path": local_path,
        "status": status,
    }
    if extra:
        item.update(extra)
    manifest["assets"].append(item)
    block["asset_id"] = asset_id
    if local_path:
        block["local_path"] = local_path


def prepare_media(
    snapshot: dict[str, Any],
    article_dir: Path,
    config: Config,
    manifest: dict[str, Any],
    cookies: list[dict[str, Any]],
) -> None:
    assets_dir = article_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    video_count = 0
    media_count = 0
    downloaded_count = 0
    referer = snapshot.get("url") or config.url
    for block in snapshot.get("blocks", []):
        if downloaded_count >= config.max_media:
            if block.get("type") in {"image", "video", "audio"}:
                manifest["warnings"].append(
                    f"Skipped media after max-media limit: block {block['index']}"
                )
            continue
        block_type = block.get("type")
        if block_type == "image":
            image_count += 1
            asset_id = f"image_{image_count:03d}"
            source = block.get("src") or block.get("current_src") or block.get("data_src")
            dest, status = download_url(
                source,
                assets_dir / asset_id,
                expected_kind="image",
                referer=referer,
                cookies=cookies,
            )
            local_path = f"assets/{dest.name}" if dest else None
            if dest:
                downloaded_count += 1
            else:
                manifest["warnings"].append(f"{asset_id} download failed: {status} ({source})")
            add_asset(
                manifest,
                block,
                asset_id=asset_id,
                asset_type="image",
                original_url=source,
                local_path=local_path,
                status=status,
            )
        elif block_type == "video":
            video_count += 1
            media_count += 1
            asset_id = f"video_{video_count:03d}"
            media_id = f"media_{media_count:03d}"
            source = block.get("src") or block.get("current_src") or block.get("data_src")
            poster = block.get("poster")
            local_path = None
            poster_local_path = None
            video_status = "metadata_only" if config.no_video else "not_downloaded"
            if source and not config.no_video:
                dest, video_status = download_url(
                    source,
                    assets_dir / asset_id,
                    expected_kind="video",
                    referer=referer,
                    cookies=cookies,
                )
                if dest:
                    local_path = f"assets/{dest.name}"
                    downloaded_count += 1
            if poster and downloaded_count < config.max_media:
                poster_dest, poster_status = download_url(
                    poster,
                    assets_dir / f"{asset_id}.thumbnail",
                    expected_kind="image",
                    referer=referer,
                    cookies=cookies,
                )
                if poster_dest:
                    poster_local_path = f"assets/{poster_dest.name}"
                    downloaded_count += 1
                    block["poster_local_path"] = poster_local_path
                else:
                    manifest["warnings"].append(
                        f"{asset_id} poster download failed: {poster_status}"
                    )
            if not local_path:
                meta_path = assets_dir / f"{media_id}.meta.json"
                write_json(
                    meta_path,
                    {
                        "asset_id": asset_id,
                        "media_id": media_id,
                        "type": "video",
                        "original_url": source,
                        "poster": poster,
                        "poster_local_path": poster_local_path,
                        "tag": block.get("tag"),
                        "nearby_text": block.get("nearby_text"),
                        "status": video_status,
                    },
                )
                block["meta_path"] = f"assets/{meta_path.name}"
                if video_status != "metadata_only":
                    manifest["warnings"].append(
                        f"{asset_id} not directly downloaded: {video_status}"
                    )
            extra = {"poster_local_path": poster_local_path, "meta_path": block.get("meta_path")}
            add_asset(
                manifest,
                block,
                asset_id=asset_id,
                asset_type="video",
                original_url=source,
                local_path=local_path,
                status="downloaded" if local_path else video_status,
                extra=extra,
            )


def block_to_llm_section(block: dict[str, Any]) -> str:
    lines = [f"[{block.get('index')}] {str(block.get('type', 'unknown')).upper()}"]
    for key in ("tag", "text", "href", "asset_id", "local_path", "poster_local_path", "meta_path"):
        value = block.get(key)
        if value:
            lines.append(f"{key}: {value}")
    original_src = block.get("src") or block.get("current_src") or block.get("data_src")
    if original_src:
        lines.append(f"original_src: {original_src}")
    if block.get("alt"):
        lines.append(f"alt: {block['alt']}")
    if block.get("nearby_text"):
        lines.append(f"nearby_text: {block['nearby_text']}")
    rect = block.get("rect") or {}
    if rect:
        lines.append(
            "visual_position: "
            f"x={rect.get('x')} y={rect.get('y')} "
            f"width={rect.get('width')} height={rect.get('height')}"
        )
    if block.get("type") == "video":
        if block.get("local_path"):
            lines.append(f'markdown_hint: <video controls src="{block["local_path"]}"></video>')
        elif block.get("poster_local_path") and original_src:
            lines.append(f"markdown_hint: [视频]({original_src})")
            lines.append(f"markdown_hint_poster: ![视频封面]({block['poster_local_path']})")
        elif block.get("meta_path"):
            lines.append(
                f"markdown_hint: <!-- {block.get('asset_id')}: 原网页包含嵌入视频，"
                f"未能直接下载。详情见 {block['meta_path']} -->"
            )
    return "\n".join(lines)


def build_llm_input(snapshot: dict[str, Any]) -> tuple[str, list[str]]:
    metadata = snapshot.get("metadata_candidates") or {}
    header = textwrap.dedent(
        f"""\
        SOURCE_URL: {snapshot.get("url", "")}
        PAGE_TITLE: {snapshot.get("title", "")}
        ROOT_SELECTOR: {snapshot.get("root_selector", "")}
        METADATA_CANDIDATES:
        - title: {metadata.get("title", "")}
        - author: {metadata.get("author", "")}
        - published_at: {metadata.get("published_at", "")}

        VISIBLE_PAGE_BLOCKS_IN_ORDER:
        """
    ).strip()
    sections = [block_to_llm_section(block) for block in snapshot.get("blocks", [])]
    return header + "\n\n" + "\n\n".join(sections), sections


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int = 180,
) -> str:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint = endpoint + "/v1"
    endpoint = endpoint + "/chat/completions"
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": 0.2},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenKB-browser-llm-exporter/0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "OpenAI-compatible response did not contain choices[0].message.content"
        ) from exc


def strip_wrapping_fence(markdown: str) -> str:
    text = markdown.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.I)
    return match.group(1).strip() if match else text


def validate_markdown(markdown: str) -> list[str]:
    warnings = []
    if not markdown.lstrip().startswith("---"):
        warnings.append("document.md does not start with YAML frontmatter")
    if not re.search(r"(?m)^#\s+\S", markdown):
        warnings.append("document.md does not contain a level-one heading")
    if re.search(r"(?m)^\[\d+\]\s+(TEXT|IMAGE|VIDEO|LINK|TABLE|LIST|QUOTE)", markdown):
        warnings.append("document.md appears to contain llm_input debug block markers")
    if "<MEDIA" in markdown:
        warnings.append("document.md appears to contain intermediate media placeholders")
    return warnings


def run_llm(llm_input: str, sections: list[str], config: Config, manifest: dict[str, Any]) -> str:
    model = config.model or os.getenv("LLM_MODEL")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    manifest["llm"]["model"] = model
    manifest["llm"]["base_url"] = base_url
    if not model:
        raise RuntimeError(
            "Missing LLM model. Pass --model, or set LLM_MODEL in the environment."
        )
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in the environment.")
    if len(llm_input) <= MAX_LLM_INPUT_CHARS:
        manifest["llm"]["chunked"] = False
        return call_openai_compatible(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": llm_input},
            ],
        )

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for section in sections:
        section_len = len(section) + 2
        if current and current_len + section_len > MAX_CHUNK_CHARS:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(section)
        current_len += section_len
    if current:
        chunks.append("\n\n".join(current))

    manifest["llm"]["chunked"] = True
    manifest["llm"]["chunks"] = len(chunks)
    partials = []
    for index, chunk in enumerate(chunks, start=1):
        partial = call_openai_compatible(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"这是长网页的第 {index}/{len(chunks)} 个连续片段。"
                        "请只清理并保留该片段内容和媒体位置，不要决定全局章节结构。\n\n"
                        + chunk
                    ),
                },
            ],
        )
        partials.append(strip_wrapping_fence(partial))
    return call_openai_compatible(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "下面是同一网页按顺序分块整理后的 Markdown 片段。"
                    "请合并为一篇统一、结构合理的最终 Markdown，保留媒体相对路径。\n\n"
                    + "\n\n---\n\n".join(partials)
                ),
            },
        ],
    )


async def async_main(argv: list[str]) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = parse_args(argv)
    if config.browser_executable and not config.browser_executable.exists():
        raise BrowserUnavailable(f"Browser executable does not exist: {config.browser_executable}")
    capture = await capture_browser(config)
    snapshot = capture.snapshot
    title = (snapshot.get("metadata_candidates") or {}).get("title") or snapshot.get("title") or ""
    article_dir = config.out_dir / make_slug(title, snapshot.get("url") or config.url)
    article_dir.mkdir(parents=True, exist_ok=True)

    if capture.screenshot:
        (article_dir / "screenshot.png").write_bytes(capture.screenshot)
    write_text(article_dir / "source.html", capture.source_html)
    write_text(article_dir / "page_text.txt", snapshot.get("page_text") or "")

    metadata = snapshot.get("metadata_candidates") or {}
    manifest: dict[str, Any] = {
        "source_url": snapshot.get("url") or config.url,
        "title": metadata.get("title") or snapshot.get("title") or "",
        "author": metadata.get("author") or "",
        "published_at": metadata.get("published_at") or "",
        "browser": capture.browser_used,
        "output_dir": str(article_dir),
        "markdown": "document.md",
        "assets": [],
        "llm": {
            "model": config.model,
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
            "chunked": False,
        },
        "warnings": list(capture.warnings),
    }

    prepare_media(snapshot, article_dir, config, manifest, capture.cookies)
    write_json(article_dir / "page_snapshot.json", snapshot)
    llm_input, sections = build_llm_input(snapshot)
    write_text(article_dir / "llm_input.txt", llm_input)

    try:
        raw_markdown = run_llm(llm_input, sections, config, manifest)
        markdown = strip_wrapping_fence(raw_markdown)
        write_text(article_dir / "llm_output.raw.md", raw_markdown)
        write_text(article_dir / "document.md", markdown)
        manifest["warnings"].extend(validate_markdown(markdown))
    except Exception as exc:
        write_text(article_dir / "llm_output.raw.md", "")
        manifest["llm"]["error"] = str(exc)
        write_json(article_dir / "manifest.json", manifest)
        logging.error("LLM call failed after capture. Outputs saved in: %s", article_dir)
        raise

    write_json(article_dir / "manifest.json", manifest)
    logging.info("Export complete: %s", article_dir)
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main(sys.argv[1:])))
    except BrowserUnavailable as exc:
        logging.error(
            "%s Install the Camoufox Python package or Playwright. "
            "Playwright setup: pip install playwright; "
            "if no --browser-executable is supplied, also run "
            "python -m playwright install chromium",
            exc,
        )
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
