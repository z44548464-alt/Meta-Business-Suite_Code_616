#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Money.impact : publication automatique d'actualités financières (Facebook + Instagram)

Principes de cette version (sense-guard) :
  1. Le LLM ne dessine pas "librement" : il EXTRAIT le sens de la news (sujet, pays,
     direction, objet visuel concret) en JSON. Le prompt d'image est construit ensuite
     par le code à partir de ces champs.
  2. Chaque image (IA ou photo) est vérifiée par un LLM de vision : l'image doit
     illustrer clairement CETTE news, sans texte, sans déformation.
  3. Chaîne 100 % gratuite avec secours : Cloudflare FLUX -> Gemini Image -> Pexels
     -> Pixabay -> carte Pillow (titre arabe de la news, ne dépend d'aucun service).
  4. Tous les LLM ont un secours : Groq (modèle détecté automatiquement) -> Gemini
     -> Cloudflare Workers AI.
"""
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import datetime as dt

import requests
import feedparser
from PIL import Image, ImageDraw, ImageFilter, ImageFont

VERSION = "2026-10-26-sense-guard"
T0 = time.time()


# ════════════════════════════════════════════════════════════════════
# Outils : logs, variables d'environnement, état
# ════════════════════════════════════════════════════════════════════
def log(msg, icon="  "):
    print(f"[{time.time() - T0:6.1f}s] {icon} {msg}", flush=True)


def ok(msg):
    log(msg, "✅")


def warn(msg):
    log(msg, "⚠️ ")


def err(msg):
    log(msg, "❌")


def info_log(msg):
    log(msg, "➡️ ")


def env(name, default=""):
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def env_int(name, default):
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default


def banner(title):
    print("═" * 68, flush=True)
    print(f" {title}", flush=True)
    print("═" * 68, flush=True)


# --- Secrets ---
GROQ_KEY = env("TOKEN_AI_DESCRYB_IMG") or env("GROQ_API_KEY")
META_TOKEN = env("PAGE_ACCESS_TOKEN")
PAGE_ID = env("PAGE_ID")
IG_ID = env("INSTAGRAM_BUSINESS_ID")
CF_ACCOUNT = env("CF_ACCOUNT_ID")
CF_TOKEN = env("CF_API_TOKEN")
GEMINI_KEY = env("GEMINI_API_KEY")
PEXELS_KEY = env("PEXELS_API_KEY")
PIXABAY_KEY = env("PIXABAY_API_KEY")
CLD_NAME = env("CLOUDINARY_CLOUD_NAME")
CLD_KEY = env("CLOUDINARY_API_KEY")
CLD_SECRET = env("CLOUDINARY_API_SECRET")

# --- Réglages ---
GRAPH = env("GRAPH_VERSION", "v26.0")
MAX_POSTS = env_int("MAX_POSTS", 1)
MAX_ATTEMPTS = env_int("MAX_ATTEMPTS", 3)
MAX_AGE_H = env_int("MAX_AGE_HOURS", 72)
MIN_IMG_W = env_int("MIN_IMG_W", 900)
IMAGE_CANDIDATES = env_int("IMAGE_CANDIDATES", 2)  # essais max par générateur IA
PHOTO_TRIES = env_int("PHOTO_TRIES", 4)  # photos testées par banque d'images
OUT_W, OUT_H = 1080, 1350
MIN_RELEVANCE = env_int("MIN_RELEVANCE", 8)  # sens : note mini /10
MIN_QUALITY = env_int("MIN_QUALITY", 7)
VISION_CHECK = env("VISION_CHECK", "1") == "1"
# 1 = si aucun LLM de vision ne répond, on n'accepte PAS l'image (on passe à la carte)
VISION_REQUIRED = env("VISION_REQUIRED", "1") == "1"
IMAGE_ORDER = [s.strip() for s in env("IMAGE_ORDER", "cloudflare,gemini,pexels,pixabay").split(",") if s.strip()]
STATE_FILE = env("STATE_FILE", "published_news.json")
BRAND = env("BRAND_NAME", "Money.impact")

GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
CF_BASE = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}/ai/run"
CF_IMAGE_MODEL = env("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
CF_TEXT_MODEL = env("CF_TEXT_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
GEMINI_TEXT_MODEL = env("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_IMAGE_MODEL = env("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

GROQ_TEXT_PREF = [m for m in [
    env("GROQ_MODEL"), env("GROQ_PROMPT_MODEL"), "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile", "qwen/qwen3-32b", "llama-3.1-8b-instant"] if m]
GROQ_VISION_PREF = [m for m in [
    env("GROQ_VISION_MODEL"), "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct"] if m]

DEFAULT_FEEDS = [
    ("ActionForex Fundamental", "https://www.actionforex.com/category/action-insight/feed/"),
    ("FXStreet", "https://www.fxstreet.com/rss/news"),
    ("ForexLive", "https://www.forexlive.com/feed/"),
]


def parse_feeds():
    raw = env("FEEDS")
    if not raw:
        return DEFAULT_FEEDS
    out = []
    for part in raw.split(","):
        if "|" in part:
            n, u = part.split("|", 1)
            out.append((n.strip(), u.strip()))
    return out or DEFAULT_FEEDS


# ════════════════════════════════════════════════════════════════════
# Désactivation temporaire des services en échec
# ════════════════════════════════════════════════════════════════════
DISABLED = {}


def is_disabled(name):
    return DISABLED.get(name, 0) > time.time()


def disable(name, minutes, why):
    DISABLED[name] = time.time() + minutes * 60
    warn(f"{name} : désactivé {minutes} min ({why})")


def minutes_to_midnight_utc():
    now = dt.datetime.now(dt.timezone.utc)
    nxt = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((nxt - now).total_seconds() // 60) + 1


# ════════════════════════════════════════════════════════════════════
# JSON : extraction tolérante
# ════════════════════════════════════════════════════════════════════
def parse_json(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("pas de JSON dans la réponse")
    return json.loads(m.group(0))


# ════════════════════════════════════════════════════════════════════
# LLM : Groq -> Gemini -> Cloudflare (texte) | Groq vision -> Gemini (vision)
# ════════════════════════════════════════════════════════════════════
GROQ_AVAILABLE = None


def groq_models():
    global GROQ_AVAILABLE
    if GROQ_AVAILABLE is not None:
        return GROQ_AVAILABLE
    GROQ_AVAILABLE = set()
    if not GROQ_KEY:
        return GROQ_AVAILABLE
    try:
        r = requests.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=20)
        if r.ok:
            GROQ_AVAILABLE = {m["id"] for m in r.json().get("data", [])}
            log(f"Groq : {len(GROQ_AVAILABLE)} modèles disponibles sur le compte")
        else:
            warn(f"Groq /models : HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001
        warn(f"Groq /models : {type(e).__name__}")
    return GROQ_AVAILABLE


def groq_pick(pref, vision=False):
    """Modèles à essayer, dans l'ordre, filtrés par ce qui existe réellement."""
    avail = groq_models()
    if not avail:
        return list(pref)
    chosen = [m for m in pref if m in avail]
    if vision:
        chosen += [m for m in sorted(avail) if ("llama-4" in m or "vision" in m) and m not in chosen]
    else:
        chosen += [m for m in sorted(avail)
                   if not any(x in m for x in ("whisper", "guard", "tts", "vision", "llama-4", "orpheus", "embed"))
                   and m not in chosen][:2]
    return chosen


def groq_chat(model, messages, json_mode=True, max_tokens=2048):
    body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    hdr = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    r = requests.post(f"{GROQ_BASE}/chat/completions", headers=hdr, json=body, timeout=60)
    if r.status_code == 400 and json_mode:
        body.pop("response_format", None)
        r = requests.post(f"{GROQ_BASE}/chat/completions", headers=hdr, json=body, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Groq {model} HTTP {r.status_code}: {r.text[:160]}")
    return r.json()["choices"][0]["message"]["content"] or ""


def gemini_call(model, parts, system=None, image_out=False, json_mode=False):
    body = {"contents": [{"parts": parts}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    gen = {}
    if json_mode:
        gen["responseMimeType"] = "application/json"
    if image_out:
        gen["responseModalities"] = ["TEXT", "IMAGE"]
        gen["imageConfig"] = {"aspectRatio": "4:5"}
    if gen:
        body["generationConfig"] = gen
    url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_KEY}"
    r = requests.post(url, json=body, timeout=120)
    if r.status_code == 400 and image_out:  # certains comptes refusent imageConfig
        body["generationConfig"].pop("imageConfig", None)
        r = requests.post(url, json=body, timeout=120)
    if not r.ok:
        raise RuntimeError(f"Gemini {model} HTTP {r.status_code}: {r.text[:160]}")
    return r.json()


def gemini_text(data):
    out = []
    for c in data.get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            if "text" in p:
                out.append(p["text"])
    return "".join(out)


def text_providers(system, user):
    """Liste de (nom, fonction) pour tâches texte, dans l'ordre de préférence."""
    provs = []
    if GROQ_KEY:
        for m in groq_pick(GROQ_TEXT_PREF)[:3]:
            provs.append((f"Groq:{m}", lambda m=m: groq_chat(
                m, [{"role": "system", "content": system}, {"role": "user", "content": user}])))
    if GEMINI_KEY:
        provs.append((f"Gemini:{GEMINI_TEXT_MODEL}", lambda: gemini_text(
            gemini_call(GEMINI_TEXT_MODEL, [{"text": user}], system=system, json_mode=True))))
    if CF_ACCOUNT and CF_TOKEN and not is_disabled("cloudflare"):
        def cf_txt():
            r = requests.post(f"{CF_BASE}/{CF_TEXT_MODEL}", headers={"Authorization": f"Bearer {CF_TOKEN}"},
                              json={"messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}], "max_tokens": 1500},
                              timeout=60)
            if not r.ok:
                raise RuntimeError(f"Cloudflare texte HTTP {r.status_code}")
            return r.json()["result"]["response"]
        provs.append((f"Cloudflare:{CF_TEXT_MODEL}", cf_txt))
    return provs


def vision_providers(system, user, jpeg):
    provs = []
    b64 = base64.b64encode(jpeg).decode()
    if GROQ_KEY:
        for m in groq_pick(GROQ_VISION_PREF, vision=True)[:2]:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
            provs.append((f"Groq-vision:{m}", lambda m=m, msgs=msgs: groq_chat(m, msgs, max_tokens=600)))
    if GEMINI_KEY:
        parts = [{"text": user}, {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]
        provs.append((f"Gemini-vision:{GEMINI_TEXT_MODEL}", lambda: gemini_text(
            gemini_call(GEMINI_TEXT_MODEL, parts, system=system, json_mode=True))))
    return provs


# ════════════════════════════════════════════════════════════════════
# Analyse de la news : extraction du SENS (JSON)
# ════════════════════════════════════════════════════════════════════
TOPICS = ["currency", "gold", "oil", "stocks", "crypto", "central_bank", "inflation", "jobs",
          "trade", "geopolitics", "other"]

ANALYSIS_SYSTEM = (
    "You are a financial news editor. You receive ONE English market news item. "
    "Return ONLY a JSON object, no commentary. Stay strictly faithful to the source text: "
    "never add facts, countries, numbers or names that are not in it. If something is not "
    "stated, leave it out.")

ANALYSIS_USER = """News title: {title}
News text: {text}

Return JSON with exactly these keys:
- "title_ar": short Arabic headline (max 90 characters), faithful to the title
- "caption_ar": Arabic social post (2 to 4 sentences, max 450 characters), faithful to the text, no invented facts
- "topic": one of {topics}
- "region": main country or zone concerned, in English (e.g. "Australia", "Eurozone", "USA", "Global")
- "direction": one of "up", "down", "neutral", "mixed" (overall market tone of the news)
- "key_fact": one English sentence stating the core fact of the news
- "visual_object": ONE concrete physical object that best symbolises the news, in English, 4 to 10 words,
  tied to the region/asset (e.g. "stacked Australian polymer banknotes", "euro banknotes and coins",
  "gold bars", "oil pipeline valves at sunset"). No people, no text, no logos, no charts with numbers.
- "photo_query": 2 to 4 English words to search a stock photo of that object"""


def has_arabic(s):
    return bool(re.search(r"[\u0600-\u06FF]", s or ""))


def analyze_news(news):
    user = ANALYSIS_USER.format(title=news["title"], text=news["text"][:1800], topics=", ".join(TOPICS))
    for name, fn in text_providers(ANALYSIS_SYSTEM, user):
        t = time.time()
        try:
            data = parse_json(fn())
            if not has_arabic(data.get("title_ar")) or not has_arabic(data.get("caption_ar")):
                raise ValueError("texte arabe manquant")
            if len(str(data.get("visual_object", ""))) < 4:
                raise ValueError("visual_object manquant")
            data["topic"] = data.get("topic") if data.get("topic") in TOPICS else "other"
            data["direction"] = data.get("direction") if data.get("direction") in (
                "up", "down", "neutral", "mixed") else "neutral"
            data["region"] = str(data.get("region") or "Global")
            data["key_fact"] = str(data.get("key_fact") or news["title"])
            data["photo_query"] = str(data.get("photo_query") or data["visual_object"])[:60]
            ok(f"Analyse reçue en {time.time() - t:.1f}s ({name})")
            return data
        except Exception as e:  # noqa: BLE001
            warn(f"Analyse {name} : {type(e).__name__} {str(e)[:120]}")
    return None


# Règles de secours (le code décide l'objet si le LLM échoue sur ce champ)
OBJECT_RULES = [
    (r"aussie|australia|\baud\b|rba", "stacked Australian polymer banknotes"),
    (r"kiwi|new zealand|\bnzd\b|rbnz", "New Zealand dollar banknotes"),
    (r"euro|eurozone|\becb\b|europe", "euro banknotes and coins"),
    (r"yen|japan|\bboj\b|\bjpy\b", "Japanese yen banknotes"),
    (r"pound|sterling|\bgbp\b|\bboe\b|\buk\b", "British pound sterling banknotes"),
    (r"swiss|franc|\bchf\b|\bsnb\b", "Swiss franc banknotes"),
    (r"canad|loonie|\bcad\b|\bboc\b", "Canadian dollar banknotes"),
    (r"yuan|china|pboc|\bcny\b", "Chinese yuan banknotes"),
    (r"gold|xau", "polished gold bars"),
    (r"oil|crude|brent|\bwti\b|opec", "oil pipeline valves at sunset"),
    (r"bitcoin|crypto|ethereum", "a physical golden coin on a dark surface"),
    (r"dollar|\busd\b|\bfed\b|treasury|payroll|nfp", "stacked US dollar bills"),
]
MOOD = {
    "up": "warm golden light, optimistic mood",
    "down": "cool dim blue light, heavy cautious mood",
    "neutral": "balanced soft light, calm mood",
    "mixed": "contrasting warm and cool light, tense mood",
}


def fallback_object(news):
    blob = f"{news['title']} {news['text']}".lower()
    for pat, obj in OBJECT_RULES:
        if re.search(pat, blob):
            return obj
    return "abstract glowing candlestick shapes without numbers"


def build_image_prompt(info):
    obj = info["visual_object"]
    return (f"{obj}, placed in the upper part of the frame. {MOOD[info['direction']]}. "
            "Lower 40% of the frame dark, smooth and uncluttered. "
            "Photorealistic editorial photography, 85mm lens, shallow depth of field, "
            "deep navy blue and warm gold colour grading, ultra detailed, sharp focus. "
            "No text, no letters, no numbers, no logos, no people.")


# ════════════════════════════════════════════════════════════════════
# Images : préparation et contrôle qualité / sens
# ════════════════════════════════════════════════════════════════════
class BadImage(Exception):
    pass


def open_image(raw):
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise BadImage(f"image illisible ({type(e).__name__})")


def to_canvas(img):
    w, h = img.size
    target = OUT_W / OUT_H
    if w / h > target:
        nw = int(h * target)
        left = (w - nw) // 2
        img = img.crop((left, 0, left + nw, h))
    else:
        nh = int(w / target)
        top = int((h - nh) * 0.4)
        img = img.crop((0, top, w, top + nh))
    img = img.resize((OUT_W, OUT_H), Image.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))


def to_jpeg(img, quality=92):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


VISION_SYSTEM = ("You are a strict quality controller for images illustrating financial news. "
                 "Answer ONLY with a JSON object.")
VISION_USER = """The image must illustrate this news item:
Title: {title}
Core fact: {fact}
Region: {region}

Return JSON with keys:
- "relevance": integer 0-10, how clearly the image visually evokes THIS news (right country/asset/theme). Generic or wrong region = low score.
- "quality": integer 0-10 (sharp, well composed, professional, not blurry, not distorted)
- "has_text": true if any readable text, letters, numbers or watermark is visible
- "has_people": true if any person or face is visible
- "deformed": true if objects or hands look warped, melted or unnatural
- "reason": short sentence"""


def vision_ok(img, info, news, is_ai):
    """Retourne True/False, ou None si aucun LLM de vision n'a pu répondre."""
    if not VISION_CHECK:
        return True
    small = img.copy()
    small.thumbnail((768, 768))
    jpeg = to_jpeg(small, 85)
    user = VISION_USER.format(title=news["title"], fact=info["key_fact"], region=info["region"])
    for name, fn in vision_providers(VISION_SYSTEM, user, jpeg):
        try:
            d = parse_json(fn())
            rel, qual = int(d.get("relevance", 0)), int(d.get("quality", 0))
            bad = []
            if rel < MIN_RELEVANCE:
                bad.append(f"sens {rel}/{MIN_RELEVANCE}")
            if qual < MIN_QUALITY:
                bad.append(f"qualité {qual}/{MIN_QUALITY}")
            if d.get("has_text"):
                bad.append("texte visible")
            if d.get("deformed"):
                bad.append("objets déformés")
            if is_ai and d.get("has_people"):
                bad.append("personnes")
            if bad:
                warn(f"Vision {name} : REJET ({', '.join(bad)}) | {str(d.get('reason', ''))[:100]}")
                return False
            ok(f"Vision {name} : OK (sens {rel}/10, qualité {qual}/10)")
            return True
        except Exception as e:  # noqa: BLE001
            warn(f"Vision {name} : {type(e).__name__} {str(e)[:100]}")
    return None


def accept(img_raw_img, info, news, is_ai, label):
    """Taille + vision. Retourne le jpeg final ou None."""
    w, h = img_raw_img.size
    if min(w, h) < MIN_IMG_W:
        warn(f"{label} : image {w}x{h} trop petite (min {MIN_IMG_W}px) -> rejetée")
        return None
    canvas = to_canvas(img_raw_img)
    verdict = vision_ok(canvas, info, news, is_ai)
    if verdict is False:
        return None
    if verdict is None:
        if VISION_REQUIRED:
            warn(f"{label} : aucun LLM de vision disponible, sens non vérifiable -> rejetée")
            return None
        warn(f"{label} : vision indisponible, image acceptée (VISION_REQUIRED=0)")
    return to_jpeg(canvas)


# ---- Générateurs IA ----
def gen_cloudflare(prompt):
    if not (CF_ACCOUNT and CF_TOKEN):
        return None
    if is_disabled("cloudflare"):
        return None
    for attempt in (1, 2):
        info_log(f"Cloudflare[{CF_IMAGE_MODEL.split('/')[-1]}] : essai {attempt}/2…")
        try:
            r = requests.post(f"{CF_BASE}/{CF_IMAGE_MODEL}",
                              headers={"Authorization": f"Bearer {CF_TOKEN}"},
                              json={"prompt": prompt, "steps": 8}, timeout=120)
        except Exception as e:  # noqa: BLE001
            warn(f"Cloudflare : {type(e).__name__}")
            continue
        if r.status_code == 429 and "daily free allocation" in r.text:
            disable("cloudflare", minutes_to_midnight_utc(), "quota quotidien épuisé, reset 00:00 UTC")
            return None
        if not r.ok:
            warn(f"Cloudflare essai {attempt}/2 : HTTP {r.status_code} -> {r.text[:140]}")
            time.sleep(2)
            continue
        ctype = r.headers.get("content-type", "")
        if "json" in ctype:
            b64 = (r.json().get("result") or {}).get("image")
            if b64:
                return base64.b64decode(b64)
        elif r.content:
            return r.content
    return None


def gen_gemini(prompt):
    if not GEMINI_KEY or is_disabled("gemini"):
        return None
    info_log(f"Gemini[{GEMINI_IMAGE_MODEL}] : génération…")
    try:
        data = gemini_call(GEMINI_IMAGE_MODEL, [{"text": prompt}], image_out=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        warn(f"Gemini : {msg[:160]}")
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            disable("gemini", minutes_to_midnight_utc(), "quota")
        elif "HTTP 4" in msg:
            disable("gemini", 240, "refus de l'API (clé/billing)")
        return None
    for c in data.get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            inline = p.get("inlineData") or p.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    warn("Gemini : réponse sans image")
    return None


# ---- Banques de photos libres ----
def pexels_urls(query):
    r = requests.get("https://api.pexels.com/v1/search", headers={"Authorization": PEXELS_KEY},
                     params={"query": query, "orientation": "portrait", "size": "large", "per_page": 15},
                     timeout=30)
    if r.status_code == 429:
        disable("pexels", 60, "limite de requêtes")
        return []
    r.raise_for_status()
    return [(p["id"], p["src"].get("large2x") or p["src"]["large"]) for p in r.json().get("photos", [])]


def pixabay_urls(query):
    r = requests.get("https://pixabay.com/api/", params={
        "key": PIXABAY_KEY, "q": query, "image_type": "photo", "orientation": "vertical",
        "min_width": MIN_IMG_W, "safesearch": "true", "per_page": 15, "order": "popular"}, timeout=30)
    if r.status_code == 429:
        disable("pixabay", 60, "limite de requêtes")
        return []
    r.raise_for_status()
    return [(p["id"], p["largeImageURL"]) for p in r.json().get("hits", [])]


USED_PHOTOS = set()


def photos_from(service, queries, info, news):
    key = PEXELS_KEY if service == "pexels" else PIXABAY_KEY
    if not key or is_disabled(service):
        return None
    finder = pexels_urls if service == "pexels" else pixabay_urls
    tested = 0
    for q in queries:
        try:
            urls = finder(q)
        except Exception as e:  # noqa: BLE001
            warn(f"{service} : {type(e).__name__} {str(e)[:100]}")
            return None
        info_log(f"{service} : {len(urls)} photos pour « {q} »")
        for pid, url in urls:
            if (service, pid) in USED_PHOTOS:
                continue
            if tested >= PHOTO_TRIES:
                return None
            USED_PHOTOS.add((service, pid))
            tested += 1
            try:
                raw = requests.get(url, timeout=60).content
                jpeg = accept(open_image(raw), info, news, False, f"{service}#{pid}")
            except Exception as e:  # noqa: BLE001
                warn(f"{service}#{pid} : {type(e).__name__}")
                continue
            if jpeg:
                return jpeg
    return None


# ---- Carte Pillow (secours garanti, titre arabe de la news) ----
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansArabic-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
]


def load_font(size):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa: BLE001
            continue
    warn("Aucune police arabe trouvée : installe fonts-noto-core (voir README)")
    return ImageFont.load_default()


def ar_display(s):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(s))
    except Exception:  # noqa: BLE001
        return s


def wrap_ar(draw, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(ar_display(trial), font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def make_card(info):
    navy_top, navy_bot, gold = (14, 30, 66), (5, 10, 26), (212, 175, 55)
    img = Image.new("RGB", (OUT_W, OUT_H))
    d = ImageDraw.Draw(img)
    for y in range(OUT_H):
        t = y / OUT_H
        d.line([(0, y), (OUT_W, y)], fill=tuple(int(navy_top[i] * (1 - t) + navy_bot[i] * t) for i in range(3)))
    # halo doré décoratif
    glow = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([OUT_W // 2 - 320, 150, OUT_W // 2 + 320, 790], fill=(212, 175, 55, 55))
    glow = glow.filter(ImageFilter.GaussianBlur(70))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    d = ImageDraw.Draw(img)
    # symbole de direction
    cx, cy, s = OUT_W // 2, 470, 170
    up = [(cx, cy - s), (cx + s, cy + s * 0.7), (cx - s, cy + s * 0.7)]
    down = [(cx, cy + s), (cx + s, cy - s * 0.7), (cx - s, cy - s * 0.7)]
    dirn = info.get("direction", "neutral")
    if dirn == "up":
        d.polygon(up, fill=gold)
    elif dirn == "down":
        d.polygon(down, fill=(200, 80, 80))
    elif dirn == "mixed":
        d.polygon([(cx - 150, cy - 40 - 110), (cx - 15, cy + 100), (cx - 285, cy + 100)], fill=gold)
        d.polygon([(cx + 150, cy + 40 + 110), (cx + 285, cy - 100), (cx + 15, cy - 100)], fill=(200, 80, 80))
    else:
        d.rounded_rectangle([cx - 190, cy - 22, cx + 190, cy + 22], radius=22, fill=gold)
    # titre
    f_title = load_font(74)
    margin = 90
    lines = wrap_ar(d, info["title_ar"], f_title, OUT_W - 2 * margin)[:5]
    y = 800
    for ln in lines:
        txt = ar_display(ln)
        w = d.textlength(txt, font=f_title)
        d.text(((OUT_W - w) / 2, y), txt, font=f_title, fill=(255, 255, 255))
        y += 104
    # filet doré et marque
    d.rectangle([OUT_W // 2 - 90, 1215, OUT_W // 2 + 90, 1221], fill=gold)
    f_brand = load_font(44)
    w = d.textlength(BRAND, font=f_brand)
    d.text(((OUT_W - w) / 2, 1250), BRAND, font=f_brand, fill=gold)
    return to_jpeg(img)


# ---- Orchestration des images ----
def get_image(info, news):
    info_log("── Génération de l'image ──")
    prompt = build_image_prompt(info)
    log(f"Objet visuel : {info['visual_object']} | région : {info['region']} | direction : {info['direction']}")
    log(f"Prompt IA : {prompt}")
    for service in IMAGE_ORDER:
        if service in ("cloudflare", "gemini"):
            gen = gen_cloudflare if service == "cloudflare" else gen_gemini
            for k in range(1, IMAGE_CANDIDATES + 1):
                if is_disabled(service) or (service == "gemini" and not GEMINI_KEY):
                    break
                info_log(f"{service} : candidat {k}/{IMAGE_CANDIDATES}")
                raw = gen(prompt)
                if not raw:
                    break
                try:
                    jpeg = accept(open_image(raw), info, news, True, service)
                except BadImage as e:
                    warn(f"{service} : {e}")
                    jpeg = None
                if jpeg:
                    return jpeg, service
        elif service in ("pexels", "pixabay"):
            queries = [info["photo_query"], info["visual_object"]]
            jpeg = photos_from(service, queries, info, news)
            if jpeg:
                return jpeg, service
        warn(f"{service} : pas d'image validée, service suivant…")
    warn("Aucune image IA/photo validée : carte Pillow (titre exact de la news)")
    return make_card(info), "carte-pillow"


# ════════════════════════════════════════════════════════════════════
# Flux d'actualités et état
# ════════════════════════════════════════════════════════════════════
def clean_html(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"&nbsp;|&#160;", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def news_id(title, link):
    return hashlib.sha1(f"{link}|{title}".encode("utf-8")).hexdigest()[:20]


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:  # noqa: BLE001
        return set()


def save_state(ids):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(ids)[-500:], f)
    except Exception as e:  # noqa: BLE001
        warn(f"État non sauvegardé : {e}")


def fetch_news():
    items, ok_n, fail_n = [], 0, 0
    now = dt.datetime.now(dt.timezone.utc)
    for name, url in parse_feeds():
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 (money-impact-bot)"})
            r.raise_for_status()
            feed = feedparser.parse(r.content)
            if not feed.entries:
                raise ValueError("flux vide")
            ok_n += 1
        except Exception as e:  # noqa: BLE001
            fail_n += 1
            warn(f"Flux {name} : {type(e).__name__} {str(e)[:80]}")
            continue
        for e in feed.entries:
            pp = e.get("published_parsed") or e.get("updated_parsed")
            when = dt.datetime(*pp[:6], tzinfo=dt.timezone.utc) if pp else now
            title = clean_html(e.get("title", ""))
            text = clean_html(e.get("summary", "") or e.get("description", ""))
            if len(title) < 10:
                continue
            items.append({"source": name, "title": title, "text": text or title, "link": e.get("link", ""),
                          "when": when, "id": news_id(title, e.get("link", ""))})
    items.sort(key=lambda x: x["when"], reverse=True)
    return items, ok_n, fail_n


# ════════════════════════════════════════════════════════════════════
# Publication
# ════════════════════════════════════════════════════════════════════
HASHTAGS = {
    "currency": "#فوركس #العملات #الدولار", "gold": "#الذهب #المعادن_الثمينة", "oil": "#النفط #الطاقة",
    "stocks": "#الأسهم #البورصة", "crypto": "#بيتكوين #العملات_الرقمية",
    "central_bank": "#البنوك_المركزية #أسعار_الفائدة", "inflation": "#التضخم #الاقتصاد",
    "jobs": "#الوظائف #الاقتصاد", "trade": "#التجارة #الاقتصاد", "geopolitics": "#الاقتصاد_العالمي",
    "other": "#الاقتصاد #الأسواق",
}


def build_caption(info, news):
    tags = HASHTAGS.get(info["topic"], HASHTAGS["other"]) + f" #{BRAND.replace('.', '_')}"
    cap = f"{info['title_ar']}\n\n{info['caption_ar']}\n\n📰 المصدر: {news['source']}\n{tags}"
    return cap[:2000]


def cloudinary_upload(jpeg):
    if not (CLD_NAME and CLD_KEY and CLD_SECRET):
        raise RuntimeError("Cloudinary : CLOUDINARY_API_SECRET manquant")
    ts = str(int(time.time()))
    sig = hashlib.sha1(f"timestamp={ts}{CLD_SECRET}".encode()).hexdigest()
    r = requests.post(f"https://api.cloudinary.com/v1_1/{CLD_NAME}/image/upload",
                      data={"api_key": CLD_KEY, "timestamp": ts, "signature": sig},
                      files={"file": ("post.jpg", jpeg, "image/jpeg")}, timeout=90)
    if not r.ok:
        raise RuntimeError(f"Cloudinary HTTP {r.status_code}: {r.text[:150]}")
    return r.json()["secure_url"]


def publish_facebook(jpeg, caption):
    r = requests.post(f"https://graph.facebook.com/{GRAPH}/{PAGE_ID}/photos",
                      data={"caption": caption, "access_token": META_TOKEN},
                      files={"source": ("post.jpg", jpeg, "image/jpeg")}, timeout=120)
    if not r.ok:
        raise RuntimeError(f"Facebook HTTP {r.status_code}: {r.text[:200]}")
    return r.json().get("post_id") or r.json().get("id")


def publish_instagram(image_url, caption):
    base = f"https://graph.facebook.com/{GRAPH}"
    r = requests.post(f"{base}/{IG_ID}/media", data={
        "image_url": image_url, "caption": caption, "access_token": META_TOKEN}, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Instagram media HTTP {r.status_code}: {r.text[:200]}")
    cid = r.json()["id"]
    for _ in range(12):
        s = requests.get(f"{base}/{cid}", params={"fields": "status_code", "access_token": META_TOKEN}, timeout=30)
        code = s.json().get("status_code") if s.ok else None
        if code == "FINISHED":
            break
        if code == "ERROR":
            raise RuntimeError("Instagram : conteneur en erreur")
        time.sleep(3)
    p = requests.post(f"{base}/{IG_ID}/media_publish", data={
        "creation_id": cid, "access_token": META_TOKEN}, timeout=60)
    if not p.ok:
        raise RuntimeError(f"Instagram publish HTTP {p.status_code}: {p.text[:200]}")
    return p.json().get("id")


def publish_all(jpeg, caption):
    results = []
    if PAGE_ID and META_TOKEN:
        try:
            results.append(("Facebook", publish_facebook(jpeg, caption)))
            ok(f"Facebook : publié ({results[-1][1]})")
        except Exception as e:  # noqa: BLE001
            err(f"Facebook : {e}")
    else:
        warn("Facebook ignoré (PAGE_ID / PAGE_ACCESS_TOKEN absent)")
    if IG_ID and META_TOKEN:
        try:
            url = cloudinary_upload(jpeg)
            results.append(("Instagram", publish_instagram(url, caption)))
            ok(f"Instagram : publié ({results[-1][1]})")
        except Exception as e:  # noqa: BLE001
            err(f"Instagram : {e}")
    else:
        warn("Instagram ignoré (INSTAGRAM_BUSINESS_ID absent)")
    return results


# ════════════════════════════════════════════════════════════════════
# Programme principal
# ════════════════════════════════════════════════════════════════════
def check_env():
    banner("ÉTAPE 1/6 — Configuration et environnement")
    log(f"Python {sys.version.split()[0]} | Pillow {Image.__version__} | GitHub Actions : "
        f"{'oui' if env('GITHUB_ACTIONS') else 'non'}")
    log(f"Script : {os.path.basename(__file__)} | version {VERSION}")
    log(f"Heure UTC : {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}")
    log(f"Réglages : ordre images={'>'.join(IMAGE_ORDER)} | carte Pillow=secours final | "
        f"min={MIN_IMG_W}px | sens>={MIN_RELEVANCE}/10 | qualité>={MIN_QUALITY}/10 | "
        f"vision={'oui' if VISION_CHECK else 'non'} (obligatoire={'oui' if VISION_REQUIRED else 'non'}) | "
        f"max posts={MAX_POSTS} | max tentatives={MAX_ATTEMPTS} | âge max={MAX_AGE_H}h | Graph={GRAPH}")
    rows = [
        ("TOKEN_AI_DESCRYB_IMG", GROQ_KEY, "Groq (texte + vision)"),
        ("PAGE_ACCESS_TOKEN", META_TOKEN, "token Meta (Facebook + Instagram)"),
        ("CF_ACCOUNT_ID", CF_ACCOUNT, "Cloudflare Workers AI"),
        ("CF_API_TOKEN", CF_TOKEN, "Cloudflare Workers AI"),
        ("GEMINI_API_KEY", GEMINI_KEY, "Gemini (texte, vision, image) -> secours"),
        ("PEXELS_API_KEY", PEXELS_KEY, "Pexels (photos libres) -> secours"),
        ("PIXABAY_API_KEY", PIXABAY_KEY, "Pixabay (photos libres) -> secours"),
        ("PAGE_ID", PAGE_ID, "ID page Facebook"),
        ("INSTAGRAM_BUSINESS_ID", IG_ID, "ID Instagram"),
        ("CLOUDINARY_CLOUD_NAME", CLD_NAME, "Cloudinary (requis pour Instagram)"),
        ("CLOUDINARY_API_KEY", CLD_KEY, "Cloudinary (requis pour Instagram)"),
        ("CLOUDINARY_API_SECRET", CLD_SECRET, "Cloudinary (requis pour Instagram)"),
    ]
    for name, val, role in rows:
        log(f"{'✅' if val else '⚠️ '} {name:<24}{'présent' if val else 'ABSENT/VIDE':<12} | {role}")
    if not GROQ_KEY and not GEMINI_KEY:
        err("Aucun LLM disponible (Groq ou Gemini requis)")
        sys.exit(1)


def main():
    print(f"▶ Python démarré (version {VERSION})", flush=True)
    check_env()

    banner("ÉTAPE 2/6 — Lecture des flux")
    news, ok_n, fail_n = fetch_news()
    published = load_state()

    banner("ÉTAPE 3/6 — Sélection")
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=MAX_AGE_H)
    fresh = [n for n in news if n["when"] >= cutoff]
    todo = [n for n in fresh if n["id"] not in published]
    log(f"Flux OK : {ok_n} | Flux en échec : {fail_n} | News : {len(news)} | "
        f"Déjà publiées : {len(fresh) - len(todo)} | Trop anciennes : {len(news) - len(fresh)} | "
        f"À publier : {len(todo)}")

    banner("ÉTAPE 4/6 — Traitement (analyse + image)")
    done, attempts, failures = 0, 0, []
    for i, n in enumerate(todo[:MAX_ATTEMPTS], 1):
        if done >= MAX_POSTS:
            break
        attempts += 1
        info_log(f"━━ News {i}/{min(len(todo), MAX_ATTEMPTS)} ━━")
        log(f"Source : {n['source']} | Date : {n['when']:%Y-%m-%d %H:%M}")
        log(f"Titre  : {n['title']}")
        info = analyze_news(n)
        if not info:
            failures.append((n["title"], "analyse LLM impossible"))
            continue
        # Le code garde la main sur l'objet visuel si le LLM est vague
        if info["visual_object"].lower() in ("", "none", "n/a"):
            info["visual_object"] = fallback_object(n)
        log(f"Titre arabe : {info['title_ar']}")
        jpeg, src = get_image(info, n)
        ok(f"Image retenue : {src}")

        banner("ÉTAPE 5/6 — Publication")
        res = publish_all(jpeg, build_caption(info, n))
        if res:
            published.add(n["id"])
            save_state(published)
            done += 1
        else:
            failures.append((n["title"], "publication impossible"))

    banner("ÉTAPE 6/6 — RÉSUMÉ")
    log(f"Tentatives : {attempts} | PUBLIÉES : {done}")
    for t, why in failures:
        err(f"échec : {t} -> {why}")
    log(f"Durée totale : {time.time() - T0:.0f}s")
    if done == 0:
        err("Résultat : ÉCHEC (rien n'a été publié)")
        sys.exit(1)
    ok("Résultat : SUCCÈS")


if __name__ == "__main__":
    main()
