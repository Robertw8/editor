from __future__ import annotations

import argparse
import datetime as dt
import html
import io
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
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
    max_edits_per_hour: int
    fallback_images: List[str]
    context_image_map: dict[str, str]
    image_search_template: str
    image_style_rotation: List[str]
    max_post_chars: int
    pexels_api_key: str
    prefer_pexels_context_image: bool
    image_state_file: str
    require_news_context: bool
    add_news_link: bool
    news_provider: str
    cryptonews_api_key: str
    cryptonews_max_pages_per_range: int
    freecrypto_api_key: str
    cryptodotnews_max_pages: int
    news_date_tolerance_days: int


HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://cryptonews-api.com/"})
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
        max_edits_per_hour=int(os.getenv("MAX_EDITS_PER_HOUR", "2")),
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
        prefer_pexels_context_image=os.getenv("PREFER_PEXELS_CONTEXT_IMAGE", "true").lower() == "true",
        image_state_file=os.getenv("IMAGE_STATE_FILE", "image_state.json"),
        require_news_context=(
            os.getenv("REQUIRE_NEWS_CONTEXT")
            or os.getenv("REQUIRE_INVESTING_NEWS")
            or "true"
        ).lower()
        == "true",
        add_news_link=os.getenv("ADD_NEWS_LINK", "false").lower() == "true",
        news_provider=os.getenv("NEWS_PROVIDER", "cryptonews").strip().lower(),
        cryptonews_api_key=(os.getenv("CRYPTONEWS_API_KEY") or "").strip(),
        cryptonews_max_pages_per_range=int(os.getenv("CRYPTONEWS_MAX_PAGES_PER_RANGE", "40")),
        freecrypto_api_key=(
            os.getenv("FREECRYPTO_API_KEY")
            or os.getenv("CRYPTONEWS_API_KEY")
            or ""
        ).strip(),
        cryptodotnews_max_pages=int(os.getenv("CRYPTODOTNEWS_MAX_PAGES", "120")),
        news_date_tolerance_days=int(os.getenv("NEWS_DATE_TOLERANCE_DAYS", "1")),
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
        pass
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ]
    for f in fmts:
        try:
            return dt.datetime.strptime(value, f)
        except ValueError:
            continue
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


def parse_cryptonews_datetime(value: str) -> Optional[dt.datetime]:
    value = (value or "").strip()
    if not value:
        return None
    parsed = parse_iso_datetime(value)
    if parsed:
        return parsed
    fmts = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
    ]
    for f in fmts:
        try:
            return dt.datetime.strptime(value, f)
        except ValueError:
            continue
    return None


def iter_month_ranges(start: dt.date, end: dt.date) -> List[tuple[dt.date, dt.date]]:
    ranges: List[tuple[dt.date, dt.date]] = []
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        next_month = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        last_day = next_month - dt.timedelta(days=1)
        ranges.append((max(start, cur), min(end, last_day)))
        cur = next_month
    return ranges


def fmt_mmddyyyy(d: dt.date) -> str:
    return d.strftime("%m%d%Y")


def map_cryptonews_item_to_article(item: dict) -> Optional[NewsArticle]:
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    url = str(item.get("news_url") or item.get("url") or "").strip()
    if not url:
        return None
    published = str(item.get("date") or item.get("published_at") or "").strip()
    if not published:
        return None
    image_url = str(item.get("image_url") or item.get("imageurl") or "").strip()
    summary = str(item.get("text") or item.get("description") or "").strip()
    return NewsArticle(title=title, url=url, published_at=published, image_url=image_url, summary=summary)


def fetch_cryptonews_by_day(start: dt.date, end: dt.date, api_key: str, max_pages_per_range: int) -> dict[str, List[NewsArticle]]:
    if not api_key:
        return {}
    picked: dict[str, List[NewsArticle]] = {}
    for r_start, r_end in iter_month_ranges(start, end):
        date_param = f"{fmt_mmddyyyy(r_start)}-{fmt_mmddyyyy(r_end)}"
        for page in range(1, max_pages_per_range + 1):
            params = {
                "section": "alltickers",
                "items": "100",
                "page": str(page),
                "date": date_param,
                "token": api_key,
            }
            try:
                r = HTTP.get("https://cryptonews-api.com/api/v1/category", params=params, timeout=(8, 25))
            except requests.RequestException:
                break
            if r.status_code != 200:
                break
            try:
                payload = r.json()
            except ValueError:
                break
            rows = payload.get("data")
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                article = map_cryptonews_item_to_article(row)
                if not article:
                    continue
                pub_dt = parse_cryptonews_datetime(article.published_at)
                if not pub_dt:
                    continue
                day = pub_dt.date()
                if day < start or day > end:
                    continue
                key = day.isoformat()
                if key not in picked:
                    picked[key] = []
                picked[key].append(article)
            time.sleep(0.15)
    for k, arr in picked.items():
        arr.sort(key=lambda a: parse_cryptonews_datetime(a.published_at) or dt.datetime.max)
        uniq: List[NewsArticle] = []
        seen = set()
        for a in arr:
            if a.url in seen:
                continue
            seen.add(a.url)
            uniq.append(a)
        picked[k] = uniq
    return picked


def strip_html_tags(text: str) -> str:
    plain = re.sub(r"<[^>]+>", " ", text or "")
    plain = html.unescape(plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain


def map_cryptodotnews_item_to_article(item: dict) -> Optional[NewsArticle]:
    title_obj = item.get("title") or {}
    title_raw = title_obj.get("rendered") if isinstance(title_obj, dict) else ""
    title = strip_html_tags(str(title_raw))
    if not title:
        return None

    link = str(item.get("link") or "").strip()
    if not link:
        return None

    published = str(item.get("date_gmt") or item.get("date") or "").strip()
    if not published:
        return None

    excerpt_obj = item.get("excerpt") or {}
    excerpt_raw = excerpt_obj.get("rendered") if isinstance(excerpt_obj, dict) else ""
    summary = strip_html_tags(str(excerpt_raw))

    image_url = ""
    emb = item.get("_embedded") or {}
    media = emb.get("wp:featuredmedia") if isinstance(emb, dict) else None
    if isinstance(media, list) and media:
        first = media[0] or {}
        image_url = str(first.get("source_url") or "").strip()

    return NewsArticle(
        title=title,
        url=link,
        published_at=published,
        image_url=image_url,
        summary=summary,
    )


def fetch_cryptodotnews_by_day(start: dt.date, end: dt.date, max_pages: int) -> dict[str, List[NewsArticle]]:
    picked: dict[str, List[NewsArticle]] = {}
    after = f"{start.isoformat()}T00:00:00"
    before = f"{(end + dt.timedelta(days=1)).isoformat()}T00:00:00"
    for page in range(1, max_pages + 1):
        params = {
            "after": after,
            "before": before,
            "per_page": "100",
            "page": str(page),
            "orderby": "date",
            "order": "desc",
            "_embed": "1",
        }
        try:
            r = HTTP.get("https://crypto.news/wp-json/wp/v2/posts", params=params, timeout=(8, 25))
        except requests.RequestException:
            break
        if r.status_code != 200:
            break
        try:
            rows = r.json()
        except ValueError:
            break
        if not isinstance(rows, list) or not rows:
            break

        min_day_on_page: Optional[dt.date] = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            article = map_cryptodotnews_item_to_article(row)
            if not article:
                continue
            pub_dt = parse_iso_datetime(article.published_at) or parse_cryptonews_datetime(article.published_at)
            if not pub_dt:
                continue
            day = pub_dt.date()
            if min_day_on_page is None or day < min_day_on_page:
                min_day_on_page = day
            if day < start or day > end:
                continue
            k = day.isoformat()
            if k not in picked:
                picked[k] = []
            picked[k].append(article)

        if min_day_on_page and min_day_on_page <= start and len(picked) >= 1:
            # We already reached earliest needed day in this range.
            # Keep looping only if you want full coverage; here we can continue safely.
            pass
        time.sleep(0.1)
    for k, arr in picked.items():
        arr.sort(key=lambda a: parse_iso_datetime(a.published_at) or parse_cryptonews_datetime(a.published_at) or dt.datetime.max)
        uniq: List[NewsArticle] = []
        seen = set()
        for a in arr:
            if a.url in seen:
                continue
            seen.add(a.url)
            uniq.append(a)
        picked[k] = uniq
    return picked


def parse_any_day(value: object) -> Optional[dt.date]:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return dt.date.fromisoformat(s[:10])
        except ValueError:
            dtx = parse_iso_datetime(s) or parse_cryptonews_datetime(s)
            return dtx.date() if dtx else None
    return None


def flatten_dict_points(obj: object) -> List[dict]:
    out: List[dict] = []
    if isinstance(obj, list):
        for x in obj:
            out.extend(flatten_dict_points(x))
    elif isinstance(obj, dict):
        if any(k in obj for k in ["date", "time", "timestamp", "open", "close", "price", "value"]):
            out.append(obj)
        for v in obj.values():
            out.extend(flatten_dict_points(v))
    return out


def pick_close_value(point: dict) -> Optional[float]:
    for key in ["close", "price", "value", "c"]:
        if key in point:
            try:
                return float(point[key])
            except Exception:
                continue
    return None


def fetch_freecrypto_context_by_day(start: dt.date, end: dt.date, api_key: str) -> dict[str, List[NewsArticle]]:
    if not api_key:
        return {}
    symbols = ["BTC", "ETH", "SOL"]
    per_day: dict[str, dict[str, float]] = {}
    for sym in symbols:
        params = {"symbol": sym, "start": start.isoformat(), "end": end.isoformat()}
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        try:
            r = HTTP.get("https://api.freecryptoapi.com/v1/getTimeframe", params=params, headers=headers, timeout=(8, 25))
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue
        try:
            payload = r.json()
        except ValueError:
            continue
        points = flatten_dict_points(payload)
        for p in points:
            d = parse_any_day(p.get("date") or p.get("time") or p.get("timestamp"))
            if not d or d < start or d > end:
                continue
            close = pick_close_value(p)
            if close is None:
                continue
            k = d.isoformat()
            if k not in per_day:
                per_day[k] = {}
            per_day[k][sym] = close

    mapped: dict[str, List[NewsArticle]] = {}
    for day, vals in per_day.items():
        if not vals:
            continue
        parts = [f"{k}: {v:.2f}" for k, v in sorted(vals.items())]
        mapped[day] = [
            NewsArticle(
                title=f"FreeCrypto market context {day}",
                url="https://freecryptoapi.com/documentation/",
                published_at=f"{day}T00:00:00Z",
                image_url="",
                summary=" | ".join(parts),
            )
        ]
    return mapped


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


def maybe_append_news_link(text: str, day_news: Optional[NewsArticle], add_news_link: bool) -> str:
    if not add_news_link or not day_news or not day_news.url:
        return text
    return f"{text}\n\nИсточник: {day_news.url}"


def pick_news_for_message_day(
    message_day: dt.date,
    news_by_day: dict[str, List[NewsArticle]],
    tolerance_days: int,
    day_usage: dict[str, int],
) -> Optional[NewsArticle]:
    def pick_from(day_key: str) -> Optional[NewsArticle]:
        items = news_by_day.get(day_key) or []
        if not items:
            return None
        idx = day_usage.get(day_key, 0)
        chosen = items[idx % len(items)]
        day_usage[day_key] = idx + 1
        return chosen

    # exact day first
    exact = pick_from(message_day.isoformat())
    if exact:
        return exact
    if tolerance_days <= 0:
        return None

    # then nearest day within tolerance (prefer past day over future on tie)
    for offset in range(1, tolerance_days + 1):
        prev = pick_from((message_day - dt.timedelta(days=offset)).isoformat())
        if prev:
            return prev
        nxt = pick_from((message_day + dt.timedelta(days=offset)).isoformat())
        if nxt:
            return nxt
    return None


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
        "fear greed index": ["fear", "greed", "страх", "жадн", "индекс"],
        "crypto sentiment": ["sentiment", "настроен", "аппетит к риску", "risk appetite"],
        "institutional crypto": ["institutional", "фонды", "капитал", "инвесторы"],
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


def search_pexels_image_urls(api_key: str, query: str, pages: int = 5) -> List[str]:
    if not api_key or not query.strip():
        return []
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    photos: List[dict] = []
    for page in range(1, max(1, pages) + 1):
        params = {
            "query": query,
            "per_page": 30,
            "page": page,
            "orientation": "landscape",
            "size": "large",
        }
        try:
            r = HTTP.get(url, headers=headers, params=params, timeout=(8, 25))
        except requests.RequestException:
            break
        if r.status_code != 200:
            break
        try:
            data = r.json()
        except ValueError:
            break
        rows = data.get("photos") or []
        if not isinstance(rows, list) or not rows:
            break
        photos.extend([p for p in rows if isinstance(p, dict)])
        if not data.get("next_page"):
            break

    urls: List[str] = []
    seen: set[str] = set()
    for p in photos:
        src = p.get("src") or {}
        candidate = src.get("large2x") or src.get("large") or src.get("original") or ""
        key = normalize_image_key(candidate)
        if candidate and key not in seen:
            seen.add(key)
            urls.append(candidate)
    return urls


def normalize_image_key(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    host = parts.netloc.lower()
    if "images.pexels.com" in host or "images.unsplash.com" in host or "source.unsplash.com" in host:
        return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/"), "", ""))
    return raw


def load_used_images(path: str) -> set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    items = data.get("used_images") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return set()
    return {normalize_image_key(str(x)) for x in items if isinstance(x, str) and normalize_image_key(str(x))}


def load_edited_message_ids(path: str) -> set[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    items = data.get("edited_message_ids") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return set()
    out: set[int] = set()
    for x in items:
        try:
            out.add(int(x))
        except Exception:
            pass
    return out


def save_state(path: str, used_images: set[str], edited_message_ids: set[int]) -> None:
    payload = {
        "used_images": sorted(used_images),
        "edited_message_ids": sorted(edited_message_ids),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def choose_unique_image(candidates: List[str], used: set[str]) -> str:
    for url in candidates:
        key = normalize_image_key(url)
        if key and key not in used:
            return url
    return ""


def choose_best_effort_image(
    pexels_candidates: List[str],
    used_images: set[str],
    search_template_url: str,
    fallback_candidate: str,
) -> str:
    # 1) Prefer unused Pexels
    chosen = choose_unique_image(pexels_candidates, used_images)
    if chosen:
        return chosen
    # 2) Template-based search URL
    if search_template_url and normalize_image_key(search_template_url) not in used_images:
        return search_template_url
    # 3) Final mapped fallback
    if fallback_candidate and normalize_image_key(fallback_candidate) not in used_images:
        return fallback_candidate
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
    edited_message_ids = load_edited_message_ids(config.image_state_file)
    day_news_usage: dict[str, int] = {}
    hourly_window_started = time.time()
    edits_in_window = 0
    if skip_news:
        news_by_day: dict[str, NewsArticle] = {}
        print("Skipping news context by --skip-news")
    else:
        if config.news_provider == "cryptonews":
            if not config.cryptonews_api_key:
                raise RuntimeError("Missing CRYPTONEWS_API_KEY in .env")
            print("Building CryptoNews API daily first-news index. This may take time...")
            news_by_day = fetch_cryptonews_by_day(
                start,
                end,
                config.cryptonews_api_key,
                config.cryptonews_max_pages_per_range,
            )
            print(f"News days indexed (CryptoNews API): {len(news_by_day)}")
        elif config.news_provider == "freecrypto":
            if not config.freecrypto_api_key:
                raise RuntimeError("Missing FREECRYPTO_API_KEY in .env")
            print("Building FreeCryptoAPI daily market context index. This may take time...")
            news_by_day = fetch_freecrypto_context_by_day(start, end, config.freecrypto_api_key)
            print(f"Context days indexed (FreeCryptoAPI): {len(news_by_day)}")
        elif config.news_provider == "crypto_news_site":
            print("Building crypto.news daily first-news index. This may take time...")
            news_by_day = fetch_cryptodotnews_by_day(start, end, config.cryptodotnews_max_pages)
            print(f"News days indexed (crypto.news): {len(news_by_day)}")
        else:
            raise RuntimeError("Unsupported NEWS_PROVIDER. Use: cryptonews, freecrypto, or crypto_news_site")

        if news_by_day:
            sample_days = sorted(news_by_day.keys())[:5]
            for d in sample_days:
                arr = news_by_day[d]
                if arr:
                    a = arr[0]
                    print(f"Indexed {d}: {a.title} | {a.url} (variants: {len(arr)})")

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
            if msg.id in edited_message_ids:
                continue

            day = msg.date.date()
            if day < start or day > end:
                continue

            day_news = pick_news_for_message_day(
                message_day=day,
                news_by_day=news_by_day,
                tolerance_days=config.news_date_tolerance_days,
                day_usage=day_news_usage,
            )
            if config.require_news_context and not day_news:
                print(f"Skip message {msg.id} ({day.isoformat()}): no FreeCrypto context for this date")
                continue
            rewritten = rewrite_text_with_openai(client, config.openai_model, msg.message, day_news, day)
            final_text = format_final_text(rewritten)
            final_text = maybe_append_news_link(final_text, day_news, config.add_news_link)
            final_text = clamp_text(final_text, config.max_post_chars)

            print(f"\\n--- Message {msg.id} ({day.isoformat()}) ---")
            print(final_text[:700] + ("..." if len(final_text) > 700 else ""))
            if day_news:
                print(f"News: {day_news.title} ({day_news.url})")

            if not config.dry_run:
                if config.max_edits_per_hour > 0:
                    now = time.time()
                    elapsed = now - hourly_window_started
                    if elapsed >= 3600:
                        hourly_window_started = now
                        edits_in_window = 0
                    elif edits_in_window >= config.max_edits_per_hour:
                        sleep_for = max(1, int(3600 - elapsed))
                        print(
                            f"Rate limit reached ({config.max_edits_per_hour}/hour). "
                            f"Sleeping {sleep_for}s before next edit..."
                        )
                        time.sleep(sleep_for)
                        hourly_window_started = time.time()
                        edits_in_window = 0

                image_url = ""
                style_hint = select_image_style(day, msg.id, config.image_style_rotation)
                keywords = extract_context_keywords(f"{msg.message} {style_hint}", day_news)
                query = " ".join(keywords)
                pexels_candidates = search_pexels_image_urls(config.pexels_api_key, query)
                fallback_candidate = pick_context_fallback_image(
                    day=day,
                    images=config.fallback_images,
                    context_image_map=config.context_image_map,
                    original_text=msg.message,
                    day_news=day_news,
                )
                template_url = build_context_search_image_url(
                    config.image_search_template,
                    original_text=f"{msg.message} {style_hint}",
                    day_news=day_news,
                )

                # Prefer contextual Pexels image, then news image, then fallbacks.
                if config.prefer_pexels_context_image:
                    image_url = choose_best_effort_image(
                        pexels_candidates=pexels_candidates,
                        used_images=used_images,
                        search_template_url="",
                        fallback_candidate="",
                    )
                    if not image_url and day_news and day_news.image_url and normalize_image_key(day_news.image_url) not in used_images:
                        image_url = day_news.image_url
                    if not image_url:
                        image_url = choose_best_effort_image(
                            pexels_candidates=[],
                            used_images=used_images,
                            search_template_url=template_url,
                            fallback_candidate=fallback_candidate,
                        )
                else:
                    if day_news and day_news.image_url and normalize_image_key(day_news.image_url) not in used_images:
                        image_url = day_news.image_url
                    if not image_url:
                        image_url = choose_best_effort_image(
                            pexels_candidates=pexels_candidates,
                            used_images=used_images,
                            search_template_url=template_url,
                            fallback_candidate=fallback_candidate,
                        )

                if image_url:
                    print(f"Image candidate for message {msg.id}: {image_url}")
                    caption = final_text
                    image_bytes, image_name = download_image_bytes(image_url)
                    if image_bytes:
                        media_updated = False
                        try:
                            await edit_message_as_photo(tg, entity, msg.id, caption, image_bytes, image_name)
                            media_updated = True
                        except Exception as e:
                            print(f"Primary photo edit failed for message {msg.id}: {e}")
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
                                media_updated = True
                            except Exception as e:
                                print(f"Fallback photo edit failed for message {msg.id}: {e}")
                                await tg.edit_message(entity, msg.id, final_text, link_preview=False)
                        if media_updated:
                            used_images.add(normalize_image_key(image_url))
                        edited_message_ids.add(int(msg.id))
                        save_state(config.image_state_file, used_images, edited_message_ids)
                    else:
                        print(f"Image download failed for message {msg.id}: {image_url}")
                        await tg.edit_message(entity, msg.id, final_text, link_preview=False)
                        edited_message_ids.add(int(msg.id))
                        save_state(config.image_state_file, used_images, edited_message_ids)
                else:
                    print(f"No image source resolved for message {msg.id}")
                    await tg.edit_message(entity, msg.id, final_text, link_preview=False)
                    edited_message_ids.add(int(msg.id))
                    save_state(config.image_state_file, used_images, edited_message_ids)
                print(f"Edited message {msg.id}")
                edits_in_window += 1
            else:
                print("DRY_RUN=true -> no edit sent")

            edited += 1
            time.sleep(1.0)

        print(f"Done. Processed: {edited}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rewrite Telegram channel daily posts by date range")
    p.add_argument("--start", default="2024-01-01", help="YYYY-MM-DD | today | yesterday")
    p.add_argument("--end", default="2024-12-31", help="YYYY-MM-DD | today | yesterday")
    p.add_argument("--skip-news", action="store_true", help="Skip FreeCrypto context parsing")
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
