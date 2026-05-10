from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from telethon import TelegramClient, functions, types
from telethon.errors import UsernameInvalidError
from telethon.tl.custom.message import Message
from urllib3.util.retry import Retry


class RewrittenPost(BaseModel):
    title: str
    body: str


class NewsArticle(BaseModel):
    title: str
    url: str
    published_at: str
    image_url: str = ""
    summary: str = ""


@dataclass
class AppConfig:
    openai_api_key: str
    openai_model: str
    tg_api_id: int
    tg_api_hash: str
    tg_session_name: str
    tg_channel: str
    dry_run: bool
    limit_per_run: int
    fallback_images: List[str]
    context_image_map: dict[str, str]
    image_search_template: str
    image_style_rotation: List[str]
    max_post_chars: int
    pexels_api_key: str
    image_state_file: str


HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://ru.investing.com/"})
HTTP.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
    ),
)


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(
        openai_api_key=require_env("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        tg_api_id=int(require_env("TG_API_ID")),
        tg_api_hash=require_env("TG_API_HASH"),
        tg_session_name=os.getenv("TG_SESSION_NAME", "editor_session"),
        tg_channel=require_env("TG_CHANNEL"),
        dry_run=os.getenv("DRY_RUN", "true").lower() == "true",
        limit_per_run=int(os.getenv("LIMIT_PER_RUN", "10")),
        fallback_images=parse_csv_env(
            os.getenv(
                "FALLBACK_CRYPTO_IMAGES",
                (
                    "https://images.unsplash.com/photo-1621761191319-c6fb62004040"
                    ",https://images.unsplash.com/photo-1640826844123-4a0b0cbf4fb8"
                    ",https://images.unsplash.com/photo-1518544866330-4e38f5c9d3f7"
                ),
            )
        ),
        context_image_map=build_context_image_map(),
        image_search_template=os.getenv(
            "IMAGE_SEARCH_TEMPLATE",
            "https://source.unsplash.com/1600x900/?{query}",
        ),
        image_style_rotation=parse_csv_env(
            os.getenv(
                "IMAGE_STYLE_ROTATION",
                (
                    "macro product photo,dramatic studio lighting,"
                    "cinematic close-up,high-detail 3d render,"
                    "editorial photo style,moody low-key lighting,"
                    "metallic texture focus,minimalist composition"
                ),
            )
        ),
        max_post_chars=int(os.getenv("MAX_POST_CHARS", "800")),
        pexels_api_key=os.getenv("PEXELS_API_KEY", "").strip(),
        image_state_file=os.getenv("IMAGE_STATE_FILE", "image_state.json"),
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def parse_csv_env(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_map_env(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in value.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k and v:
            result[k] = v
    return result


def build_context_image_map() -> dict[str, str]:
    raw = os.getenv(
        "CONTEXT_IMAGE_MAP",
        (
            "bitcoin:https://images.unsplash.com/photo-1518546305927-5a555bb7020d,"
            "ethereum:https://images.unsplash.com/photo-1622630998477-20aa696ecb05,"
            "solana:https://images.unsplash.com/photo-1642104704074-907c0698cbd9,"
            "xrp:https://images.unsplash.com/photo-1639762681485-074b7f938ba0,"
            "etf:https://images.unsplash.com/photo-1642790551116-18e150f248e5,"
            "sec:https://images.unsplash.com/photo-1450101499163-c8848c66ca85,"
            "hack:https://images.unsplash.com/photo-1550751827-4bd374c3f58b,"
            "mining:https://images.unsplash.com/photo-1518770660439-4636190af475,"
            "defi:https://images.unsplash.com/photo-1621504450181-5d356f61d307,"
            "nft:https://images.unsplash.com/photo-1634973357973-f2ed2657db3c,"
            "stablecoin:https://images.unsplash.com/photo-1520607162513-77705c0f0d4a,"
            "market:https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3,"
            "crypto:https://images.unsplash.com/photo-1621761191319-c6fb62004040"
        ),
    )
    return parse_map_env(raw)


def parse_iso_datetime(value: str) -> Optional[dt.datetime]:
    value = value.strip()
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_day_arg(value: str) -> dt.date:
    value = value.strip().lower()
    today = dt.date.today()
    if value == "today":
        return today
    if value == "yesterday":
        return today - dt.timedelta(days=1)
    return dt.date.fromisoformat(value)


def normalize_tg_channel(raw: str) -> object:
    value = raw.strip()
    if value.startswith("https://t.me/") or value.startswith("http://t.me/"):
        value = value.rstrip("/").split("/")[-1]
    if value.startswith("@"):
        return value
    if re.fullmatch(r"-100\d{5,}", value):
        return int(value)
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{3,31}", value):
        return f"@{value}"
    return value


def fetch_html(url: str) -> Optional[BeautifulSoup]:
    try:
        r = HTTP.get(url, timeout=(8, 25))
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return BeautifulSoup(r.text, "lxml")


def extract_article_links_from_news_page(soup: BeautifulSoup) -> List[str]:
    links: List[str] = []
    seen = set()
    for a in soup.select("a[href^='/news/']"):
        href = a.get("href", "").strip()
        if not href:
            continue
        if not re.search(r"/news/.+-\d+$", href):
            continue
        abs_url = urljoin("https://ru.investing.com", href)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        links.append(abs_url)
    return links


def fetch_article(url: str) -> Optional[NewsArticle]:
    soup = fetch_html(url)
    if not soup:
        return None

    h1 = soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    if not title:
        return None

    published = ""
    meta_pub = soup.select_one("meta[property='article:published_time']")
    if meta_pub:
        published = (meta_pub.get("content") or "").strip()
    if not published:
        tnode = soup.select_one("time[datetime]")
        if tnode:
            published = (tnode.get("datetime") or "").strip()
    if not published:
        return None

    image_url = ""
    meta_img = soup.select_one("meta[property='og:image']")
    if meta_img:
        image_url = (meta_img.get("content") or "").strip()

    paragraphs: List[str] = []
    for p in soup.select("article p, div[class*='article'] p"):
        txt = p.get_text(" ", strip=True)
        if txt and len(txt) > 40:
            paragraphs.append(txt)
        if len(paragraphs) >= 3:
            break

    return NewsArticle(
        title=title,
        url=url,
        published_at=published,
        image_url=image_url,
        summary=" ".join(paragraphs),
    )


def fetch_first_news_by_day(start: dt.date, end: dt.date) -> dict[str, NewsArticle]:
    picked: dict[str, NewsArticle] = {}
    page = 1
    pages_without_links = 0

    while True:
        news_url = (
            "https://ru.investing.com/news/cryptocurrency-news"
            if page == 1
            else f"https://ru.investing.com/news/cryptocurrency-news/{page}"
        )

        soup = fetch_html(news_url)
        if not soup:
            pages_without_links += 1
            if pages_without_links >= 3:
                break
            page += 1
            continue

        links = extract_article_links_from_news_page(soup)
        if not links:
            pages_without_links += 1
            if pages_without_links >= 3:
                break
            page += 1
            continue
        pages_without_links = 0

        oldest_on_page: Optional[dt.date] = None
        for link in links:
            try:
                article = fetch_article(link)
                if not article:
                    continue

                pub_dt = parse_iso_datetime(article.published_at)
                if not pub_dt:
                    continue

                day = pub_dt.date()
                if oldest_on_page is None or day < oldest_on_page:
                    oldest_on_page = day

                if day < start or day > end:
                    continue

                key = day.isoformat()
                if key not in picked:
                    picked[key] = article
                else:
                    old_dt = parse_iso_datetime(picked[key].published_at)
                    if old_dt and pub_dt < old_dt:
                        picked[key] = article
            except Exception:
                continue

            time.sleep(0.2)

        if len(picked) >= (end - start).days + 1:
            break

        if oldest_on_page and oldest_on_page < start:
            break

        page += 1

    return picked


def build_prompt(original_text: str, day_news: Optional[NewsArticle], day: dt.date) -> str:
    news_payload = day_news.model_dump_json(indent=2, ensure_ascii=False) if day_news else "{}"
    return (
        "Ты редактор телеграм-канала про рынок крипты. "
        "Перепиши старый пост в формате как в образце: короткий заголовок и 3-5 абзацев по делу, "
        "читаемо, без воды, на русском, стиль новостной. "
        "Добавляй тикеры с $ (например $BTC, $ETH, $SOL), если уместно. "
        "Если есть новость дня, используй её как главный факт-контекст. "
        "Не выдумывай факты, используй только исходный текст и news data. "
        "Держи итоговый текст компактным, максимум 800 символов.\n\n"
        f"Дата поста: {day.isoformat()}\n"
        f"News data JSON:\n{news_payload}\n\n"
        f"Исходный текст:\n{original_text}\n\n"
        "Верни только JSON с полями title и body."
    )


def rewrite_text_with_openai(
    client: OpenAI,
    model: str,
    original_text: str,
    day_news: Optional[NewsArticle],
    day: dt.date,
) -> RewrittenPost:
    prompt = build_prompt(original_text, day_news, day)
    resp = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "rewritten_post",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["title", "body"],
                    "additionalProperties": False,
                },
            }
        },
    )

    raw = resp.output_text.strip()
    data = json.loads(raw)
    post = RewrittenPost(**data)
    post.body = re.sub(r"\n{3,}", "\n\n", post.body.strip())
    post.title = post.title.strip()
    return post


def format_final_text(post: RewrittenPost) -> str:
    return f"{post.title}\n\n{post.body}"


def clamp_text(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def pick_context_fallback_image(
    day: dt.date,
    images: List[str],
    context_image_map: dict[str, str],
    original_text: str,
    day_news: Optional[NewsArticle],
) -> str:
    haystack = " ".join(
        [
            original_text.lower(),
            (day_news.title.lower() if day_news else ""),
            (day_news.summary.lower() if day_news else ""),
        ]
    )
    priorities = [
        "bitcoin",
        "ethereum",
        "solana",
        "xrp",
        "etf",
        "sec",
        "hack",
        "mining",
        "defi",
        "nft",
        "stablecoin",
        "market",
        "crypto",
    ]
    for key in priorities:
        if key in haystack and key in context_image_map:
            return context_image_map[key]
    if images:
        return images[day.toordinal() % len(images)]
    return ""


def extract_context_keywords(original_text: str, day_news: Optional[NewsArticle]) -> List[str]:
    haystack = " ".join(
        [
            original_text.lower(),
            (day_news.title.lower() if day_news else ""),
            (day_news.summary.lower() if day_news else ""),
        ]
    )
    keywords: List[str] = []
    mapping = {
        "bitcoin": ["bitcoin", "btc"],
        "ethereum": ["ethereum", "eth"],
        "solana": ["solana", "sol"],
        "ripple": ["ripple", "xrp"],
        "etf": ["etf"],
        "sec": ["sec", "регулятор", "regulator"],
        "hacking": ["hack", "взлом", "exploit"],
        "mining": ["mining", "майнинг"],
        "defi": ["defi"],
        "nft": ["nft"],
        "stablecoin": ["stablecoin", "usdt", "usdc"],
        "crypto market": ["рынок", "market", "trading"],
    }
    for out, variants in mapping.items():
        if any(v in haystack for v in variants):
            keywords.append(out)
    if not keywords:
        keywords = ["crypto", "blockchain", "market"]
    return keywords[:3]


def build_context_search_image_url(
    template: str,
    original_text: str,
    day_news: Optional[NewsArticle],
) -> str:
    keywords = extract_context_keywords(original_text, day_news)
    query = ",".join(keywords)
    return template.replace("{query}", query)


def search_pexels_image_urls(api_key: str, query: str) -> List[str]:
    if not api_key or not query.strip():
        return []
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": 10,
        "orientation": "landscape",
        "size": "large",
    }
    try:
        r = HTTP.get(url, headers=headers, params=params, timeout=(8, 25))
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except ValueError:
        return []
    photos = data.get("photos") or []
    urls: List[str] = []
    for p in photos:
        src = p.get("src") or {}
        candidate = src.get("large2x") or src.get("large") or src.get("original") or ""
        if candidate:
            urls.append(candidate)
    return urls


def load_used_images(path: str) -> set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    items = data.get("used_images") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return set()
    return {str(x) for x in items if isinstance(x, str)}


def save_used_images(path: str, used: set[str]) -> None:
    payload = {"used_images": sorted(used)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def choose_unique_image(candidates: List[str], used: set[str]) -> str:
    for url in candidates:
        if url not in used:
            return url
    return ""


def select_image_style(day: dt.date, msg_id: int, styles: List[str]) -> str:
    if not styles:
        return "editorial crypto photo"
    idx = (day.toordinal() + msg_id) % len(styles)
    return styles[idx]


def download_image_bytes(url: str) -> tuple[Optional[bytes], str]:
    try:
        r = HTTP.get(url, timeout=(8, 25))
    except requests.RequestException:
        return None, ""
    if r.status_code != 200 or not r.content:
        return None, ""

    ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if not ctype.startswith("image/"):
        return None, ""

    ext = mimetypes.guess_extension(ctype) or ".jpg"
    if ext == ".jpe":
        ext = ".jpg"
    return r.content, f"cover{ext}"


async def edit_message_as_photo(
    tg: TelegramClient,
    entity: object,
    msg_id: int,
    caption: str,
    image_bytes: bytes,
    image_name: str,
) -> None:
    bio = io.BytesIO(image_bytes)
    bio.name = image_name or "cover.jpg"
    uploaded = await tg.upload_file(bio)
    media = types.InputMediaUploadedPhoto(file=uploaded)
    await tg(
        functions.messages.EditMessageRequest(
            peer=entity,
            id=msg_id,
            message=caption,
            media=media,
            no_webpage=True,
        )
    )


async def process_channel(config: AppConfig, start: dt.date, end: dt.date, skip_news: bool) -> None:
    client = OpenAI(api_key=config.openai_api_key)
    used_images = load_used_images(config.image_state_file)

    if skip_news:
        news_by_day: dict[str, NewsArticle] = {}
        print("Skipping Investing news by --skip-news")
    else:
        print("Building Investing daily first-news index (ru.investing.com). This may take time...")
        news_by_day = fetch_first_news_by_day(start, end)
        print(f"News days indexed: {len(news_by_day)}")

    async with TelegramClient(config.tg_session_name, config.tg_api_id, config.tg_api_hash) as tg:
        channel_ref = normalize_tg_channel(config.tg_channel)
        try:
            entity = await tg.get_entity(channel_ref)
        except UsernameInvalidError as e:
            raise RuntimeError(
                "Invalid TG_CHANNEL. Use @username, https://t.me/username, or -100... id"
            ) from e

        edited = 0
        async for msg in tg.iter_messages(entity, reverse=True):
            if edited >= config.limit_per_run:
                break
            if not isinstance(msg, Message) or not msg.message:
                continue

            day = msg.date.date()
            if day < start or day > end:
                continue

            day_news = news_by_day.get(day.isoformat())
            rewritten = rewrite_text_with_openai(client, config.openai_model, msg.message, day_news, day)
            final_text = clamp_text(format_final_text(rewritten), config.max_post_chars)

            print(f"\\n--- Message {msg.id} ({day.isoformat()}) ---")
            print(final_text[:700] + ("..." if len(final_text) > 700 else ""))
            if day_news:
                print(f"News: {day_news.title} ({day_news.url})")

            if not config.dry_run:
                image_url = ""
                if day_news and day_news.image_url:
                    image_url = day_news.image_url
                    if image_url in used_images:
                        image_url = ""
                else:
                    style_hint = select_image_style(day, msg.id, config.image_style_rotation)
                    keywords = extract_context_keywords(f"{msg.message} {style_hint}", day_news)
                    query = " ".join(keywords)
                    pexels_candidates = search_pexels_image_urls(config.pexels_api_key, query)
                    image_url = choose_unique_image(pexels_candidates, used_images)
                    if not image_url:
                        image_url = build_context_search_image_url(
                            config.image_search_template,
                            original_text=f"{msg.message} {style_hint}",
                            day_news=day_news,
                        )

                if not image_url:
                    fallback_candidate = pick_context_fallback_image(
                        day=day,
                        images=config.fallback_images,
                        context_image_map=config.context_image_map,
                        original_text=msg.message,
                        day_news=day_news,
                    )
                    if fallback_candidate and fallback_candidate not in used_images:
                        image_url = fallback_candidate

                if image_url:
                    caption = final_text
                    image_bytes, image_name = download_image_bytes(image_url)
                    if image_bytes:
                        try:
                            await edit_message_as_photo(tg, entity, msg.id, caption, image_bytes, image_name)
                        except Exception:
                            bio = io.BytesIO(image_bytes)
                            bio.name = image_name or "cover.jpg"
                            try:
                                await tg.edit_message(
                                    entity,
                                    msg.id,
                                    caption,
                                    file=bio,
                                    force_document=False,
                                    link_preview=False,
                                )
                            except Exception:
                                await tg.edit_message(entity, msg.id, final_text, link_preview=False)
                        used_images.add(image_url)
                        save_used_images(config.image_state_file, used_images)
                    else:
                        print(f"Image download failed for message {msg.id}: {image_url}")
                        await tg.edit_message(entity, msg.id, final_text, link_preview=False)
                else:
                    print(f"No image source resolved for message {msg.id}")
                    await tg.edit_message(entity, msg.id, final_text, link_preview=False)
                print(f"Edited message {msg.id}")
            else:
                print("DRY_RUN=true -> no edit sent")

            edited += 1
            time.sleep(1.0)

        print(f"Done. Processed: {edited}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rewrite Telegram channel daily posts by date range")
    p.add_argument("--start", default="2024-01-01", help="YYYY-MM-DD | today | yesterday")
    p.add_argument("--end", default="2024-12-31", help="YYYY-MM-DD | today | yesterday")
    p.add_argument("--skip-news", action="store_true", help="Skip Investing parsing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    start = parse_day_arg(args.start)
    end = parse_day_arg(args.end)
    if start > end:
        raise RuntimeError("Invalid date range: --start must be <= --end")

    import asyncio

    asyncio.run(process_channel(cfg, start, end, skip_news=args.skip_news))


if __name__ == "__main__":
    main()
