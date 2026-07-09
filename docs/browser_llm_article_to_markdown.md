# Experimental browser + LLM article exporter

`scripts/browser_llm_article_to_markdown.py` is a standalone experiment for
turning a rendered webpage article into Markdown plus downloaded attachments.
It is not part of the normal OpenKB import pipeline, does not call
`wechat_article_to_markdown`, and does not modify the OKF compiler.

The tool deliberately separates responsibilities:

- The browser opens the real page, waits for rendering, captures visible text
  and media blocks in page order, saves debug artifacts, and downloads media
  when possible.
- The LLM receives a structured visible-page snapshot and decides the article
  title, paragraphs, quotes, lists, section hierarchy, and where images or
  videos belong in the final Markdown.
- The script does not mechanically convert DOM tags into the final Markdown
  structure.

## Usage

```bash
python scripts/browser_llm_article_to_markdown.py \
  --url "https://mp.weixin.qq.com/s/xxxx" \
  --out-dir ./browser_llm_out \
  --browser camoufox \
  --model "$LLM_MODEL"
```

Playwright can be selected directly:

```bash
python scripts/browser_llm_article_to_markdown.py \
  --url "https://mp.weixin.qq.com/s/xxxx" \
  --out-dir ./browser_llm_out \
  --browser playwright \
  --browser-executable "C:\Users\Helw\AppData\Local\camoufox\camoufox\Cache\camoufox.exe" \
  --model "$LLM_MODEL"
```

Useful options:

```text
--url              Required article URL.
--out-dir          Output root directory. Default: ./browser_llm_out
--browser          camoufox | playwright. Default: camoufox.
--browser-executable
                   Optional browser executable path, such as a local Camoufox binary.
--model            LLM model name. Defaults to LLM_MODEL.
--headless         Run headless. Default: true. Use --no-headless to inspect.
--timeout          Page load timeout in seconds. Default: 60.
--save-screenshot  Save screenshot.png for debugging.
--max-media        Maximum media files to download. Default: 200.
--no-video         Skip direct video downloads and save metadata/posters only.
```

## Environment

The LLM call uses an OpenAI-compatible `/v1/chat/completions` endpoint:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
LLM_MODEL
CAMOUFOX_EXECUTABLE
```

`OPENAI_BASE_URL` defaults to `https://api.openai.com`. API keys are not written
to `manifest.json`, logs, or output artifacts.

`CAMOUFOX_EXECUTABLE` can be used instead of passing `--browser-executable`.

## Browser dependencies

Camoufox is attempted first when `--browser camoufox` is used. If Camoufox is
not importable or cannot launch, the script falls back to Playwright Chromium.
When `--browser-executable` points to a Camoufox or Firefox binary and
Playwright is available, the fallback launches it through Playwright's Firefox
driver.

Install Playwright manually when needed:

```bash
pip install playwright
python -m playwright install chromium
```

Camoufox is optional and intentionally not added to OpenKB runtime
dependencies.

## Output

Each run creates one article directory under `--out-dir`:

```text
browser_llm_out/<article_slug>/
  document.md
  assets/
    image_001.jpg
    image_002.png
    video_001.mp4
    video_001.thumbnail.jpg
    media_001.meta.json
  source.html
  page_text.txt
  page_snapshot.json
  llm_input.txt
  llm_output.raw.md
  manifest.json
  screenshot.png              # only with --save-screenshot
```

`page_snapshot.json` contains the browser-captured blocks, geometry, style
fields, metadata candidates, original media URLs, and local asset paths after
download attempts.

`llm_input.txt` is the compact structured input given to the LLM. It preserves
visible page order and includes media nodes near their original positions.

`llm_output.raw.md` stores the direct LLM response. `document.md` stores the
same response after only minimal whole-document code-fence unwrapping.

## Media behavior

Images are detected from `src`, `currentSrc`, `data-src`, `data-original`,
`data-backsrc`, `srcset`, and CSS `background-image`. They are downloaded in
page order as `assets/image_001.*`, `assets/image_002.*`, and so on. The script
uses the URL suffix or `Content-Type` to choose the extension. Failed image
downloads are recorded in `manifest.json` warnings and do not fail the article.

Videos are detected from `<video>`, `<source>`, and video-like iframes. Direct
video files are downloaded as `assets/video_001.*` when possible. If a video
cannot be directly downloaded, the script saves poster imagery when available
and writes `assets/media_001.meta.json` with the original URL, poster, nearby
text, and status. It does not invent a successful video download.

## LLM role

The LLM prompt is embedded in the script. The model decides:

- the article title and frontmatter values from the visible snapshot;
- which blocks are body paragraphs, quotes, lists, or headings;
- whether `##` or `###` sections are semantically justified;
- where downloaded images and video placeholders belong.

For long pages, the script chunks the ordered blocks, asks the LLM to preserve
and clean each chunk, and then asks the LLM to merge those chunks into one final
Markdown document. `manifest.json` records whether chunking was used.

## Known limitations

- WeChat and other platforms may block automated browsers.
- Some images reject hotlink-style downloads even after browser rendering.
- Embedded videos often cannot be downloaded directly.
- Output structure quality depends on the selected LLM.
- Long-document chunking can cause slight global structure drift.
- The extractor captures a broad visible snapshot; noisy sidebars or footer
  blocks may appear in `llm_input.txt` for the LLM to judge.
