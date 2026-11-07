import sys
import faulthandler
SCRIPT_VERSION = "2026-11-07-money-impact-v5-photos-web-commentaire"
print(f"▶ Python démarré (version {SCRIPT_VERSION}), chargement des modules…", flush=True)
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)

import os
import io
import ast
import base64
import traceback
import json
import re
import time
import random
import zlib
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace

import requests
import feedparser
import cloudinary
import cloudinary.uploader
from bs4 import BeautifulSoup
import PIL
from PIL import (
    Image, ImageDraw, ImageFont, ImageOps, ImageEnhance,
    ImageFilter, ImageChops, ImageStat, features,
)
from huggingface_hub import InferenceClient

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_OK = True
except ImportError:
    ARABIC_OK = False
    print("ATTENTION : pip install arabic-reshaper python-bidi (sinon l'arabe sera mal affiché)")
print("▶ Modules chargés.", flush=True)

# ======================================================================
# CONFIGURATION
# ======================================================================

TRUSTED_WEB_SOURCES = {
    # Sources actuelles
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "InvestingLive Macro": "https://investinglive.com/rss/news",
    "ActionForex Fundamental": "https://www.actionforex.com/category/fundamental-analysis/feed/",

    # Marchés financiers
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    "Bloomberg Economics": "https://feeds.bloomberg.com/economics/news.rss",
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Financial Times Markets": "https://www.ft.com/markets?format=rss",
    "Financial Times": "https://www.ft.com/rss/home",
    "Investing.com": "https://www.investing.com/rss/news.rss",
    "Benzinga": "https://www.benzinga.com/feed",
    "Seeking Alpha": "https://seekingalpha.com/feed.xml",

    # Banques centrales et macroéconomie
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "Bank of England": "https://www.bankofengland.co.uk/rss/news",

    # Matières premières et marchés à terme
    "CME Group": "https://www.cmegroup.com/rss/commentary-home-insights-analysis.rss",
    "OilPrice": "https://oilprice.com/rss/main",

    # Analyse économique
    "The Economist": "https://www.economist.com/finance-and-economics/rss.xml",
    "NPR Economy": "https://feeds.npr.org/1017/rss.xml",
}


# Une actualité doit être directement liée à la finance de marché ou avoir un impact
# financier concret. La simple mention d'argent ne suffit pas.
ALLOWED_NEWS_CATEGORIES = (
    "Forex", "Gold", "Indices", "Crypto", "Central banks",
    "Economic news", "Stocks", "Oil", "Bonds", "Market impact",
)

# Note minimale (sur 100) donnée par le LLM pour qu'une actualité soit publiée.
MIN_NEWS_SCORE = 75
# Types d'article acceptés (le LLM doit en choisir un).
ACCEPTED_ARTICLE_TYPES = ("market_news", "macro_data", "central_bank", "company_market_event")
# Seuil de similarité (0-1) au-delà duquel deux titres sont considérés comme doublons.
DUPLICATE_TITLE_SIMILARITY = 0.70

# Hashtags ajoutés après la ligne « المصدر » (de base + selon la catégorie validée).
HASHTAGS_BASE = "#الأسواق_المالية #اقتصاد #أخبار_اقتصادية #استثمار #تداول #money_impact"
HASHTAGS_CATEGORIE = {
    "Forex": "#فوركس #forex #العملات #الدولار #fx",
    "Gold": "#الذهب #gold #XAUUSD #المعادن_الثمينة",
    "Indices": "#البورصة #المؤشرات #wallstreet #stockmarket",
    "Crypto": "#بيتكوين #العملات_الرقمية #bitcoin #crypto",
    "Central banks": "#البنوك_المركزية #الفيدرالي #أسعار_الفائدة #fed",
    "Economic news": "#الاقتصاد_العالمي #التضخم #economy #macro",
    "Stocks": "#الأسهم #stocks #وول_ستريت #nasdaq",
    "Oil": "#النفط #oil #برنت #الطاقة",
    "Bonds": "#السندات #bonds #treasury #العوائد",
    "Market impact": "#الأسواق #markets #trading #تحليل_السوق",
}

# ---------------------------------------------------------------------
# COMPTES MULTIPLES : même nom de secret + suffixe 1, 2, 3, 4, 5
#   Cloudflare : CF_ACCOUNT_ID / CF_API_TOKEN, puis CF_ACCOUNT_ID1 / CF_API_TOKEN1, ...
#   Hugging Face : HF_TOKEN, HF_TOKEN1, HF_TOKEN2, HF_TOKEN3
#   Groq : TOKEN_AI_DESCRYB_IMG, TOKEN_AI_DESCRYB_IMG1, ...
# Si un compte est épuisé / refusé, le script passe automatiquement au suivant.
# ---------------------------------------------------------------------
KEY_SUFFIXES = ("", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13")


def _csv_env(name, default):
    v = os.getenv(name, "").strip()
    return tuple(x.strip() for x in v.split(",") if x.strip()) if v else tuple(default)


def cf_accounts():
    out = []
    for s in KEY_SUFFIXES:
        acc, tok = os.getenv("CF_ACCOUNT_ID" + s, ""), os.getenv("CF_API_TOKEN" + s, "")
        if acc and tok:
            out.append((s or "0", acc, tok))
    return out


def hf_tokens():
    return [(s or "0", os.getenv("HF_TOKEN" + s, "")) for s in KEY_SUFFIXES
            if os.getenv("HF_TOKEN" + s, "")]


def groq_keys():
    return [(s or "0", os.getenv("TOKEN_AI_DESCRYB_IMG" + s, "")) for s in KEY_SUFFIXES
            if os.getenv("TOKEN_AI_DESCRYB_IMG" + s, "")]


# ---- Modèles (modifiables par variable d'environnement, séparés par des virgules) ----
CF_TEXT_MODELS = _csv_env("CF_TEXT_MODELS", (
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/meta/llama-3.1-8b-instruct",
))
HF_TEXT_MODELS = _csv_env("HF_TEXT_MODELS", (
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen/Qwen3-32B",
))
GROQ_MODELS = _csv_env("GROQ_MODELS", (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
))
CF_IMAGE_MODELS = _csv_env("CF_IMAGE_MODELS", (
    "@cf/black-forest-labs/flux-1-schnell",
    "@cf/bytedance/stable-diffusion-xl-lightning",
    "@cf/stabilityai/stable-diffusion-xl-base-1.0",
))
HF_IMAGE_MODELS = _csv_env("HF_IMAGE_MODELS", (
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-3.5-large-turbo",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0",
))
GRAPH_URL = "https://graph.facebook.com/" + os.getenv("GRAPH_VERSION", "v26.0")

# Pour publier sur la 2e page, mets META_SUFFIX=_MONEY_IMPAC dans le workflow.
META_SUFFIX = os.getenv("META_SUFFIX", "")


def menv(name):
    return os.getenv(name + META_SUFFIX, "")


IMG_W, IMG_H = 1024, 1280
OUT_W, OUT_H = 1080, 1350

IMAGE_CANDIDATES = int(os.getenv("IMAGE_CANDIDATES", "2"))
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "300"))
MAX_ATTEMPTS_PER_RUN = int(os.getenv("MAX_ATTEMPTS_PER_RUN", "330"))
MAX_NEWS_AGE_HOURS = int(os.getenv("MAX_NEWS_AGE_HOURS", "72"))
IG_CAPTION_LIMIT = 2200

# ---- Avertissement : posté en 1er commentaire de chaque publication (image, Reel, Short) ----
DISCLAIMER = "⚠️ هذا محتوى إخباري تعليمي، وليس نصيحة استثمارية."
# ---- Règle « وماذا بعد؟ » : sans suite concrète, la news n'est pas publiée ----
REQUIRE_WHAT_NEXT = os.getenv("REQUIRE_WHAT_NEXT", "1") == "1"
# ---- Contrôle anti-écriture (OCR) des images générées par IA ----
OCR_CHECK = os.getenv("OCR_CHECK", "1") == "1"
OCR_MIN_CONF = int(os.getenv("OCR_MIN_CONF", "60"))
IMAGE_MAX_TRIES = int(os.getenv("IMAGE_MAX_TRIES", "4"))
# ---- Vraies photos du web (sans clé API : Wikimedia Commons + Openverse) ----
WEB_IMAGES_ENABLED = os.getenv("WEB_IMAGES_ENABLED", "1") == "1"
WEB_MIN_SIDE = int(os.getenv("WEB_MIN_SIDE", "900"))
WEB_MAX_DOWNLOADS = int(os.getenv("WEB_MAX_DOWNLOADS", "6"))
WEB_HEADERS = {"User-Agent": "MoneyImpactBot/1.0 (https://instagram.com/money.impact.today; "
                             "news page, editorial use)"}
# Sujets qui ont toujours une vraie photo (argent / papier / marques : l'IA les dessine mal).
WEB_FORCE = ("Forex", "Bonds")
WEB_CATEGORIES = ("Forex", "Bonds", "Stocks", "Economic news")     # repli si le LLM ne répond pas
WEB_QUERIES_DEFAUT = {
    "Forex": ("us dollar banknotes", "euro banknotes", "currency exchange"),
    "Gold": ("gold bars", "gold nuggets"),
    "Indices": ("stock exchange building", "stock market trading floor"),
    "Crypto": ("bitcoin coin", "cryptocurrency coin"),
    "Central banks": ("central bank building", "federal reserve building"),
    "Economic news": ("financial district skyline", "container port cranes"),
    "Stocks": ("stock exchange building", "business district skyscrapers"),
    "Oil": ("oil refinery", "oil pumpjack"),
    "Bonds": ("treasury building", "government building columns"),
    "Market impact": ("stock exchange building", "financial district skyline"),
}

BRAND_NAME = ("Money", ".impact")
BRAND_HANDLE = "@money.impact.today"
BRAND_ACCENT = (255, 196, 0)
AR_FONT_ENV = os.getenv("AR_FONT_PATH", "")

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _find_logo():
    env = os.getenv("LOGO_PATH", "")
    if env:
        return Path(env)
    for name in ("logo_money_impact.png", "logo_money_impact.jpg", "logo_money_impact.jpeg"):
        p = BASE_DIR / "imgs" / name
        if p.is_file():
            return p
    return BASE_DIR / "imgs" / "logo_money_impact.png"


LOGO_PATH = _find_logo()
LOGO_BACKDROP = os.getenv("LOGO_BACKDROP", "0") == "1"

# Palettes variées : évite que toutes les images aient exactement la même couleur.
PALETTES = (
    "deep navy blue and warm gold",
    "deep emerald green and soft silver",
    "dark charcoal and electric teal",
    "deep burgundy and warm amber",
    "midnight purple and copper",
)

STYLE_SUFFIX = (
    "warm cinematic side lighting, shot on Canon EOS R5, 85mm f/1.4 lens, "
    "shallow depth of field, creamy bokeh, tack-sharp focus on the hero subject, "
    "editorial photography, professional color grading in deep navy blue and warm gold, "
    "photorealistic, ultra detailed, 8k, hero subject in the upper 60% of the frame, "
    "lower 40% of the frame dark, smooth and uncluttered, plain smooth surfaces"
)

NEGATIVE_PROMPT = ("text, letters, numbers, words, writing, typography, logo, watermark, signature, "
                   "banknote, currency, money, coin, sign, label, poster, screen, document, "
                   "people, face, hands, cartoon, illustration, painting, 3d render, neon, "
                   "blurry, low quality")

# Scènes de secours (liste blanche) : matières naturelles ou architecture, aucune écriture possible.
SAFE_SCENES = {
    "Gold": ("raw gold nuggets resting on dark slate rock, one large nugget in sharp focus",
             "unrefined gold ore veins in a dark rock face lit by a warm side light",
             "scattered rough gold nuggets and fine gold dust on black volcanic stone"),
    "Oil": ("a steel oil pumpjack silhouetted against a glowing dusk sky",
            "an oil refinery with steel pipes and tall distillation towers at dusk",
            "a row of oil pumpjacks on a desert plain at golden hour",
            "a drilling rig on open plains under a twilight sky"),
    "Central banks": ("the neoclassical stone facade of a grand institutional building with tall marble columns at dusk",
                      "wide stone steps leading to massive marble columns of a classical building at blue hour"),
    "Forex": ("the dome of a classical capitol-style government building at dusk",
              "a deep-water harbor with tall cranes and plain steel containers at blue hour",
              "a neoclassical government building with marble columns and an empty plaza at dusk"),
    "Indices": ("a bronze bull statue on a quiet city street at dusk",
                "the stone columns of a classical exchange building at blue hour",
                "a skyline of tall glass skyscrapers at blue hour seen across calm water"),
    "Stocks": ("a skyline of modern glass skyscrapers reflecting a warm sunset",
               "a bare silicon wafer close-up with a soft rainbow sheen on a dark surface",
               "the plain glass facade of a modern tower at blue hour"),
    "Crypto": ("a smooth polished chrome sphere resting on reflective black glass",
               "a polished titanium hexagonal prism on a dark reflective surface",
               "a dark hall of tall plain metal cabinets with small glowing lights"),
    "Bonds": ("a long marble hall with tall polished columns and a stone floor",
              "marble columns of a neoclassical building with long shadows at dusk"),
    "Economic news": ("a busy harbor with plain stacked steel containers and cranes at dusk",
                      "a steel factory interior with glowing furnaces and metal beams",
                      "a suspension bridge over calm water at blue hour"),
    "Market impact": ("a city skyline at blue hour with soft reflections on the water",
                      "a harbor with cargo ships and cranes at dusk"),
}

# Tout objet qui porte (ou attire) de l'écriture : interdit dans un prompt d'image IA.
_UNSAFE_PROMPT_RE = re.compile(
    r"\b(banknotes?|bank\s*notes?|bills?|currency|cash|money|coins?|ingots?|gold\s+bars?|bullion|"
    r"signs?|signage|billboards?|posters?|banners?|newspapers?|magazines?|documents?|papers?|"
    r"contracts?|labels?|packag(?:e|es|ing)|box|boxes|bottles?|brands?|logos?|screens?|monitors?|"
    r"tickers?|charts?|graphs?|keyboards?|phones?|smartphones?|laptops?|credit\s+cards?|cards?|"
    r"books?|text|letters?|numbers?|words?|writing|typography|plates?|flags?|stamps?|plaques?|"
    r"tablets?|clocks?|dials?|gauges?|calendars?|stickers?|tags?|tickets?|envelopes?|passports?)\b",
    re.I)
_NOMS_OK = frozenset(("the", "and", "for", "with", "after", "amid", "over", "new", "wall",
                      "street", "gold", "oil"))

# Alphabets étrangers (chinois, cyrillique, thaï, coréen...) : la police ne les dessine pas (□□□).
_FOREIGN_RE = re.compile(r"[\u0400-\u04FF\u0E00-\u0E7F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF\uFF00-\uFFEF]")
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff\u00ad]")
_GLYPH_OK_RE = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF"
                          r"A-Za-z0-9\s.,;:!?%$€£&+\-–—/()\[\]'\"«»“”’…،؛؟#@*=_|]")


def nettoyer_pour_image(s):
    """Supprime tout caractère que la police ne sait pas dessiner (sinon : carrés)."""
    s = _INVISIBLE_RE.sub("", s or "")
    return " ".join(_GLYPH_OK_RE.sub("", s).split())


def sujet_neutre(titre):
    """Titre sans noms propres ni chiffres : le LLM d'image ne voit aucune société ni marque."""
    mots = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", titre or "") if not w[0].isupper()]
    return " ".join(mots)[:140]


# ---------------------------------------------------------------------
# Expressions régulières du parseur de prompt d'image (niveau module).
# ---------------------------------------------------------------------
_LABEL_POS_RE = re.compile(
    r"(?:positive\s*prompt|prompt\s*positif|positive)\s*(?:\([^)]*\))?\s*[:：\-–]\s*", re.I)
_LABEL_NEG_RE = re.compile(
    r"\n?\s*(?:negative\s*prompt|prompt\s*n[ée]gatif|negative)\s*(?:\([^)]*\))?\s*[:：\-–]", re.I)
_REFUSAL_RE = re.compile(
    r"\b(i can(?:['’]t|not)|i['’]m sorry|i am sorry|as an ai)\b", re.I)

# ---------------------------------------------------------------------
# Pré-filtre éditorial (gratuit, sans LLM) : écarte tout de suite les contenus
# qui ne doivent jamais être publiés (transcriptions d'appels de résultats,
# contenu promotionnel, etc.).
# ---------------------------------------------------------------------
_BAD_TITLE_RE = re.compile(
    r"(earnings\s+call|conference\s+call|call\s+transcript|earnings\s+transcript|"
    r"webinar|sponsored|promo\s*code|coupon|giveaway|discount|"
    r"how\s+to\s+(?:buy|invest|trade)|best\s+(?:forex\s+|crypto\s+)?(?:brokers?|exchanges?|apps?)|"
    r"top\s+\d+\s+(?:stocks|brokers|crypto)|horoscope|podcast|newsletter|"
    r"press\s+release\s*:|advertorial|partner\s+content)", re.I)
_BAD_BODY_RE = re.compile(
    r"(welcome\s+to\s+(?:the\s+)?.{0,80}(?:earnings|conference)\s+call|"
    r"question[\s\-]+and[\s\-]+answer\s+session|"
    r"\boperator\b.{0,60}\bgood\s+(?:morning|afternoon|evening)|"
    r"this\s+(?:is\s+a\s+)?(?:paid|sponsored)\s+(?:post|content|promotion)|"
    r"affiliate\s+(?:link|disclosure)|use\s+(?:my\s+)?(?:code|link)\s+to\s+(?:get|save|sign))",
    re.I | re.S)
_STOPWORDS = frozenset((
    "the a an of to in on for and or as at by with from is are was were be been it its "
    "this that these those after before over under into amid says say said new"
).split())


# ======================================================================
# LOGS
# ======================================================================
try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except Exception:
    pass

IS_GHA = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
T0 = time.time()
SECRET_ENV_NAMES = tuple(
    [n + s for n in ("TOKEN_AI_DESCRYB_IMG", "HF_TOKEN", "CF_API_TOKEN", "CF_ACCOUNT_ID")
     for s in KEY_SUFFIXES]
    + ["CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET",
       "PAGE_ACCESS_TOKEN", "PAGE_ACCESS_TOKEN_MONEY_IMPAC",
       "ID_CLIENT_FB", "CODE_SECRET_DU_CLIENT_FB", "YOUTUBE_REFRESH_TOKEN"]
)
_ICONS = {"INFO": "  ", "STEP": "➡️ ", "OK": "✅", "WARN": "⚠️ ", "ERROR": "❌"}

STATS = {
    "flux_ok": 0, "flux_ko": 0, "news_total": 0, "deja": 0, "anciennes": 0,
    "candidates": 0, "tentatives": 0, "publies": 0, "rejetees": 0, "echecs": [],
}


def _redact(text):
    s = str(text)
    for name in SECRET_ENV_NAMES:
        v = os.getenv(name, "")
        if v and len(v) >= 6:
            s = s.replace(v, "***")
    return re.sub(r"(access_token=)[^&\s\"']+", r"\1***", s)


def log(*args, level="INFO"):
    msg = _redact(" ".join(str(a) for a in args))
    print(f"[{time.time() - T0:6.1f}s] {_ICONS.get(level, '  ')} {msg}", flush=True)
    if IS_GHA and level == "ERROR":
        print("::error::" + msg.replace("\r", " ").replace("\n", " ")[:300], flush=True)


def banner(num, total, title):
    print("", flush=True)
    print("═" * 68, flush=True)
    print(f" ÉTAPE {num}/{total} — {title}", flush=True)
    print("═" * 68, flush=True)


def step_summary(markdown):
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
    except Exception:
        pass


def explain_graph_error(data):
    err = data.get("error", data) if isinstance(data, dict) else {}
    if not isinstance(err, dict):
        err = {}
    msg = err.get("message") or str(data)[:300]
    code, sub = err.get("code"), err.get("error_subcode")
    hints = {
        190: "Token invalide ou expiré : régénère un token longue durée de la Page.",
        200: "Permission manquante (pages_manage_posts, pages_read_engagement, "
             "pages_manage_engagement, instagram_content_publish, instagram_manage_comments...).",
        10: "Permission refusée pour cette action.",
        100: "Paramètre invalide : vérifie PAGE_ID / INSTAGRAM_BUSINESS_ID / URL image "
             "(ou la version GRAPH_VERSION).",
        368: "Blocage temporaire anti-spam de Meta : réduis la fréquence.",
        4: "Limite d'appels API atteinte.", 17: "Limite d'appels API atteinte.",
        32: "Limite d'appels API atteinte.", 613: "Limite d'appels API atteinte.",
        9004: "Instagram n'arrive pas à télécharger l'image (URL).",
        36000: "Image refusée (taille ou ratio non supporté).",
    }
    hint = hints.get(code)
    return f"{msg} (code={code}, subcode={sub})" + (f"  →  {hint}" if hint else "")


def poster_commentaire(objet_id, texte, token, nom=""):
    """Poste `texte` en commentaire (1er commentaire de la Page) sur un post / média / vidéo.

    Ne lève JAMAIS d'exception et ne bloque jamais la publication : en cas d'échec, un WARN
    explique la cause (souvent : permission pages_manage_engagement / instagram_manage_comments)."""
    if not (objet_id and texte and token):
        return False
    for essai in range(1, 4):
        try:
            r = requests.post(f"{GRAPH_URL}/{objet_id}/comments",
                              data={"message": texte, "access_token": token}, timeout=40)
            try:
                data = r.json()
            except Exception:
                data = {"message": r.text[:300]}
            if r.ok and isinstance(data, dict) and "error" not in data:
                log(f"{nom} : avertissement posté en 1er commentaire (id={data.get('id')})",
                    level="OK")
                return True
            log(f"{nom} : commentaire refusé (essai {essai}/3) -> {explain_graph_error(data)}",
                level="WARN")
            code = (data.get("error") or {}).get("code") if isinstance(data, dict) else None
            if code in (10, 190, 200, 283):
                log(f"{nom} : permission/token manquant : ajoute pages_manage_engagement "
                    "(Facebook) / instagram_manage_comments (Instagram) puis régénère le token.",
                    level="WARN")
                return False
        except Exception as e:
            log(f"{nom} : commentaire exception {type(e).__name__} : {str(e)[:200]}", level="WARN")
        time.sleep(4 * essai)
    return False


# ======================================================================
# SERVICES DÉSACTIVÉS (partagé LLM + images) : "cf:1", "hf:2", "groq:0"...
# ======================================================================
_DEAD = set()
_CACHE = {}


def _clean_llm(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()


# ----------------------------- Cloudflare -----------------------------
def _quota_text(txt):
    t = (txt or "").lower()
    return any(k in t for k in ("4006", "daily free allocation", "neurons", "quota"))


def _llm_cloudflare(messages, temperature, max_tokens):
    for label, acc, tok in cf_accounts():
        dk = f"cf:{label}"
        if dk in _DEAD:
            continue
        for model in CF_TEXT_MODELS:
            try:
                r = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{model}",
                    headers={"Authorization": f"Bearer {tok}"}, timeout=90,
                    json={"messages": messages, "temperature": temperature,
                          "max_tokens": max_tokens})
                if r.status_code in (401, 403):
                    log(f"Cloudflare compte {label} : accès refusé -> compte ignoré.", level="WARN")
                    _DEAD.add(dk)
                    break
                if r.status_code == 429 or (r.status_code >= 400 and _quota_text(r.text)):
                    log(f"Cloudflare compte {label} : quota atteint -> compte suivant.", level="WARN")
                    _DEAD.add(dk)
                    break
                r.raise_for_status()
                res = r.json().get("result") or {}
                out = res.get("response")
                if not out and res.get("choices"):
                    out = res["choices"][0].get("message", {}).get("content")
                # Cloudflare renvoie parfois un JSON déjà parsé (dict/list).
                if isinstance(out, (dict, list)):
                    out = json.dumps(out, ensure_ascii=False)
                if out and str(out).strip():
                    log(f"LLM : Cloudflare compte {label} | {model}", level="OK")
                    return str(out)
                log(f"Cloudflare {label} {model} : réponse vide", level="WARN")
            except Exception as e:
                log(f"Cloudflare {label} {model} : {type(e).__name__} {str(e)[:120]}", level="WARN")
    return None


# --------------------------- Hugging Face ----------------------------
def _llm_hf(messages, temperature, max_tokens):
    for label, token in hf_tokens():
        dk = f"hf:{label}"
        if dk in _DEAD:
            continue
        client = InferenceClient(api_key=token)
        for model in HF_TEXT_MODELS:
            try:
                res = client.chat_completion(
                    model=model, messages=messages,
                    temperature=max(temperature, 0.01), max_tokens=max_tokens)
                out = res.choices[0].message.content
                if out and out.strip():
                    log(f"LLM : Hugging Face compte {label} | {model}", level="OK")
                    return out
            except Exception as e:
                s = str(e)
                log(f"HF compte {label} {model} : {type(e).__name__} {s[:110]}", level="WARN")
                if "402" in s or "Payment Required" in s:
                    log(f"HF compte {label} : crédits épuisés -> compte suivant.", level="WARN")
                    _DEAD.add(dk)
                    break
                if "401" in s or "403" in s:
                    log(f"HF compte {label} : token refusé -> compte suivant.", level="WARN")
                    _DEAD.add(dk)
                    break
    return None


# ------------------------------- Groq --------------------------------
def _groq_models(key):
    """Modèles réellement accessibles avec cette clé (appel /models)."""
    ck = "groq_models:" + key[-6:]
    if ck not in _CACHE:
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=20)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
        skip = ("whisper", "guard", "tts", "orpheus", "playai", "embed", "safeguard")
        ids = [i for i in ids if not any(s in i.lower() for s in skip)]
        pref = ("llama-3.3", "llama-4", "gpt-oss-120b", "qwen", "gpt-oss", "llama-3.1")
        ids.sort(key=lambda i: next((n for n, p in enumerate(pref) if p in i.lower()), 99))
        _CACHE[ck] = ids
        log(f"Groq : modèles disponibles pour cette clé -> {', '.join(ids[:8])}")
    return _CACHE[ck]


def _llm_groq(messages, temperature, max_tokens):
    for label, key in groq_keys():
        dk = f"groq:{label}"
        if dk in _DEAD:
            continue
        queue, seen, auto_done = list(GROQ_MODELS), set(), False
        while queue:
            model = queue.pop(0)
            if model in seen:
                continue
            seen.add(model)
            mt = max_tokens + (1500 if any(k in model for k in ("gpt-oss", "qwen3")) else 0)
            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"}, timeout=90,
                    json={"model": model, "messages": messages,
                          "temperature": temperature, "max_tokens": mt})
                if r.status_code == 401:
                    log(f"Groq clé {label} : refusée -> clé ignorée.", level="WARN")
                    _DEAD.add(dk)
                    break
                if r.status_code == 404 or (r.status_code == 400 and "model" in r.text.lower()):
                    log(f"Groq clé {label} : modèle {model} indisponible.", level="WARN")
                    if not queue and not auto_done:
                        auto_done = True
                        try:
                            queue = [m for m in _groq_models(key) if m not in seen][:5]
                        except Exception as e:
                            log(f"Groq /models : {str(e)[:100]}", level="WARN")
                    continue
                if r.status_code == 429:
                    log(f"Groq clé {label} {model} : limite atteinte -> modèle suivant.", level="WARN")
                    continue
                r.raise_for_status()
                out = _clean_llm(r.json()["choices"][0]["message"]["content"])
                if out:
                    log(f"LLM : Groq clé {label} | {model}", level="OK")
                    return out
            except Exception as e:
                log(f"Groq clé {label} {model} : {type(e).__name__} {str(e)[:110]}", level="WARN")
    return None


def llm_chat(messages, temperature=0.2, max_tokens=1800, order=None):
    """Cloudflare -> Hugging Face -> Groq (ordre modifiable via `order`)."""
    for fn in (order or (_llm_cloudflare, _llm_hf, _llm_groq)):
        try:
            out = fn(messages, temperature, max_tokens)
            if out and str(out).strip():
                return _clean_llm(out)
        except Exception as e:
            log(f"LLM {fn.__name__} : {type(e).__name__} {str(e)[:150]}", level="WARN")
    return None


# ======================================================================
# GÉNÉRATION DE POST (image + publication)
# ======================================================================
class _FontPair:
    """Police arabe + police latine de secours (même taille visuelle).

    Beaucoup de polices arabes (ex. Noto Sans Arabic) n'ont PAS les lettres latines :
    « DoorDash » s'affichait alors en carrés □□□□. Les mots non arabes sont donc
    dessinés avec la police latine, les mots arabes avec la police arabe."""

    def __init__(self, ar, lat):
        self.ar = ar
        self.lat = lat

    def pick(self, is_ar):
        return self.ar if is_ar else self.lat


class TextPost:
    def __init__(self):
        # URL Cloudinary de la dernière image publiée (conservée dans l'historique des posts).
        self.last_image_url = None
        # Origine de la dernière image : "web" | "ia" | "local" (+ crédit et URL source si "web").
        self.last_image_origine = None
        self.last_image_source = None
        self.last_credit = None
        # URLs de photos web déjà utilisées (chargées depuis l'historique par main()).
        self.urls_web_utilisees = set()

    # ------------------------------------------------------------------
    # 1a) Plan : vraie photo web ou image IA ?
    # ------------------------------------------------------------------
    def _planifier_image(self, categorie, titre, news):
        """Retourne (mode, requêtes) ; mode = "web" (vraie photo) ou "ai" (image générée)."""
        defaut = WEB_QUERIES_DEFAUT.get(categorie) or WEB_QUERIES_DEFAUT["Market impact"]
        system = (
            "You are a photo editor for a financial-news page. Decide how to illustrate ONE news "
            "post. Reply with mode \"web\" (a REAL photograph found on the web) when the natural "
            "picture shows money or objects that carry writing: banknotes, coins, bonds or "
            "treasury papers, bank cards, a company's products or shops, stock-exchange floors. "
            "Reply with mode \"ai\" only when the subject is a neutral natural or architectural "
            "scene: raw gold, oil rigs or refineries, a central-bank building, a city skyline, a "
            "port. Also write 3 photo-search queries: English, 2 to 4 words each, generic names of "
            "real objects or places that exist as photographs (e.g. \"euro banknotes\", \"stock "
            "exchange building\", \"pharmacy shelves\"). NEVER use company names, brand names or "
            "people's names in the queries. "
            'Return ONLY JSON: {"mode": "web", "queries": ["...", "...", "..."]}')
        user = f"CATEGORY: {categorie}\nHEADLINE: {titre}\nTEXT: {(news or '')[:500]}"
        mode, queries = None, []
        raw = llm_chat([{"role": "system", "content": system},
                        {"role": "user", "content": user}], temperature=0.1, max_tokens=300)
        if raw:
            try:
                d = Extract_post._extract_json(raw)
                mode = str(d.get("mode") or "").strip().lower()
                queries = [" ".join(str(q).split())[:60] for q in (d.get("queries") or [])
                           if str(q).strip()]
                queries = [q for q in queries if not _FOREIGN_RE.search(q)][:3]
            except Exception as e:
                log(f"Plan d'image : JSON invalide ({e}) -> règles par catégorie.", level="WARN")
        if mode not in ("web", "ai"):
            mode = "web" if categorie in WEB_CATEGORIES else "ai"
        if categorie in WEB_FORCE:
            mode = "web"
        for q in defaut:                       # complète avec les requêtes par défaut
            if q not in queries and len(queries) < 4:
                queries.append(q)
        return mode, queries

    # ------------------------------------------------------------------
    # 1b) Vraies photos du web : Wikimedia Commons puis Openverse (SANS clé API)
    # ------------------------------------------------------------------
    @staticmethod
    def _texte_brut(html):
        return re.sub(r"\s+", " ", BeautifulSoup(html or "", "html.parser")
                      .get_text(" ", strip=True)).strip()

    @staticmethod
    def _licence_ok(lic):
        """Domaine public, CC0, CC BY. Refuse SA (œuvre dérivée), NC, ND, GFDL, usage équitable."""
        l = " ".join((lic or "").lower().replace("-", " ").split())
        if not l:
            return False
        if re.search(r"\b(nc|nd|sa|gfdl|fair|non)\b", l):
            return False
        return bool(re.match(r"(cc0|public domain|pd\b|pdm|cc by)", l))

    def _chercher_wikimedia(self, query):
        out = []
        try:
            r = requests.get(
                "https://commons.wikimedia.org/w/api.php", headers=WEB_HEADERS, timeout=30,
                params={"action": "query", "format": "json", "generator": "search",
                        "gsrnamespace": 6, "gsrsearch": f"{query} filetype:bitmap",
                        "gsrlimit": 20, "prop": "imageinfo",
                        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1600})
            r.raise_for_status()
            pages = (r.json().get("query") or {}).get("pages") or {}
        except Exception as e:
            log(f"Wikimedia « {query} » : {type(e).__name__} {str(e)[:120]}", level="WARN")
            return out
        for p in sorted(pages.values(), key=lambda x: x.get("index", 99)):
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata") or {}
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if min(ii.get("width", 0), ii.get("height", 0)) < WEB_MIN_SIDE:
                continue
            licence = (meta.get("LicenseShortName") or {}).get("value", "")
            if not self._licence_ok(licence):
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            auteur = self._texte_brut((meta.get("Artist") or {}).get("value", ""))[:70]
            credit = f"{auteur or 'auteur inconnu'} / Wikimedia Commons / {licence}"
            out.append({"url": url, "page": ii.get("descriptionurl") or url,
                        "credit": credit, "source": "Wikimedia Commons"})
        return out

    def _chercher_openverse(self, query):
        out = []
        try:
            r = requests.get(
                "https://api.openverse.org/v1/images/", headers=WEB_HEADERS, timeout=30,
                params={"q": query, "license": "cc0,pdm,by", "category": "photograph",
                        "page_size": 20, "mature": "false"})
            if r.status_code in (401, 403, 429):
                log(f"Openverse : accès limité (HTTP {r.status_code}) -> source ignorée.",
                    level="WARN")
                return out
            r.raise_for_status()
            resultats = r.json().get("results") or []
        except Exception as e:
            log(f"Openverse « {query} » : {type(e).__name__} {str(e)[:120]}", level="WARN")
            return out
        for it in resultats:
            url = it.get("url")
            w, h = it.get("width") or 0, it.get("height") or 0
            if not url or (w and h and min(w, h) < WEB_MIN_SIDE):
                continue
            lic = f"{it.get('license', '')} {it.get('license_version', '')}".strip()
            if not self._licence_ok("cc " + lic if lic and not lic.startswith(("cc", "pdm")) else lic):
                continue
            credit = " ".join(str(it.get("attribution") or
                                  f"{it.get('creator') or 'auteur inconnu'} / Openverse / {lic}")
                              .split())[:150]
            out.append({"url": url, "page": it.get("foreign_landing_url") or url,
                        "credit": credit, "source": "Openverse"})
        return out

    def _telecharger_web(self, cand):
        try:
            r = requests.get(cand["url"], headers=WEB_HEADERS, timeout=45)
            r.raise_for_status()
            if len(r.content) > 25_000_000:
                return None
            img = Image.open(io.BytesIO(r.content))
            img.load()
            img = img.convert("RGB")
        except Exception as e:
            log(f"Photo web illisible ({type(e).__name__} {str(e)[:80]}) : {cand['url'][:90]}",
                level="WARN")
            return None
        w, h = img.size
        if min(w, h) < WEB_MIN_SIDE or not (0.6 <= w / h <= 2.0):
            return None
        return img

    def _image_web(self, queries):
        """Retourne (image, infos) ou (None, None). Aucune clé API nécessaire."""
        if not WEB_IMAGES_ENABLED:
            return None, None
        candidats, vus = [], set()
        for q in queries[:4]:
            for nom, fn in (("Wikimedia", self._chercher_wikimedia),
                            ("Openverse", self._chercher_openverse)):
                trouves = [c for c in fn(q) if c["url"] not in vus
                           and c["url"] not in self.urls_web_utilisees]
                for c in trouves:
                    vus.add(c["url"])
                candidats.extend(trouves)
                log(f"Photo web « {q} » via {nom} : {len(trouves)} candidat(s) libre(s) de droits")
            if len(candidats) >= WEB_MAX_DOWNLOADS * 2:
                break
        if not candidats:
            return None, None
        meilleurs = []
        for c in candidats[:WEB_MAX_DOWNLOADS]:
            img = self._telecharger_web(c)
            if img is None:
                continue
            score = self._score_image(img)
            log(f"Photo web : {c['source']} | {img.size[0]}x{img.size[1]} | score {score:.1f} | "
                f"{c['credit'][:70]}")
            meilleurs.append((score, img, c))
        if not meilleurs:
            return None, None
        score, img, infos = max(meilleurs, key=lambda m: m[0])
        log(f"Photo web retenue (score {score:.1f}) : {infos['page'][:100]}", level="OK")
        return img, infos

    # ------------------------------------------------------------------
    # 1c) Prompt d'image IA (sujet neutre, sans objet portant de l'écriture)
    # ------------------------------------------------------------------
    def donne_moi_prompt_images(self, categorie, sujet, attempt=0, order=None):
        scenes = " | ".join(SAFE_SCENES.get(categorie) or SAFE_SCENES["Market impact"])
        system_prompt = (
            "You are an editorial photo director. You describe ONE photorealistic scene made only "
            "of natural subjects or architecture that carry NO writing of any kind."
        )
        user_prompt = f"""
        MARKET TOPIC: {categorie or 'Market impact'}{(' - ' + sujet) if sujet else ''}

        Write ONE photorealistic editorial photo description (max 55 words) for this topic.
        Safe scene families (adapt freely, do not copy): {scenes}

        Rules:
        1. Start with the exact physical hero subject, in the UPPER 60% of the frame, slightly
           off-center, sharp focus.
        2. Background: relevant but blurred (bokeh), specific natural lighting.
        3. The lower 40% of the frame is dark, smooth and uncluttered.
        4. Use ONLY natural materials or architecture: stone, rock, marble, steel pipes, glass
           towers, bronze statues, cranes, fields, water.
        5. NEVER mention a company, brand, product, person, or any object that carries writing:
           money, banknotes, coins, bars, signs, boxes, packaging, bottles, labels, screens,
           charts, documents, flags, books, clocks.
        6. Do not mention colors. Do not use negations ("no text"): describe only what is visible.
        OUTPUT ONLY one plain paragraph: the scene description. No label, no markdown.
        """

        class _Reponse:
            content = ""

        r = _Reponse()
        temp = (0.5, 0.4, 0.7, 0.3)[min(attempt, 3)]
        r.content = llm_chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=temp, order=order,
        ) or ""
        return r

    @staticmethod
    def _parse_prompts(text):
        """Tolérant : markdown, casse, labels absents, JSON, intro parasite."""
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
        text = text.replace("**", "").replace("__", "").replace("`", "")
        text = re.sub(r"^\s*#+\s*", "", text, flags=re.M).strip()
        if not text or _REFUSAL_RE.search(text[:200]):
            return None, None

        # Cas JSON : {"positive_prompt": "...", "negative_prompt": "..."}
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            try:
                d = json.loads(text[a:b + 1])
                if isinstance(d, dict):
                    pos = next((str(v) for k, v in d.items() if "pos" in k.lower()
                                or k.lower() in ("prompt", "description")), "")
                    neg = next((str(v) for k, v in d.items() if "neg" in k.lower()), "")
                    if pos.strip():
                        return " ".join(pos.split()), neg.strip()
            except Exception:
                pass

        m_neg = _LABEL_NEG_RE.search(text)
        head = text[:m_neg.start()] if m_neg else text
        negative = text[m_neg.end():].strip() if m_neg else ""
        m_pos = _LABEL_POS_RE.search(head)
        if m_pos:
            head = head[m_pos.end():]

        # retire les lignes d'intro du type "Here is the description:"
        lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not ln.endswith(":")]
        positive = " ".join(" ".join(lines).split())
        positive = positive.strip(' "\'“”«»[]')
        return (positive or None), negative

    @staticmethod
    def _valid_positive(positive):
        return bool(positive) and len(positive.split()) >= 6

    @staticmethod
    def _prompt_sur(positive, titre):
        """(True, '') si le prompt ne contient ni objet écrit ni nom propre du titre."""
        m = _UNSAFE_PROMPT_RE.search(positive)
        if m:
            return False, f"objet portant de l'écriture : « {m.group(0)} »"
        noms = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", titre or "")
                if w[0].isupper()} - _NOMS_OK
        low = positive.lower()
        for n in noms:
            if re.search(rf"\b{re.escape(n)}\b", low):
                return False, f"nom propre du titre : « {n} »"
        return True, ""

    @staticmethod
    def _assembler(scene):
        palette = random.choice(PALETTES)
        suffix = STYLE_SUFFIX.replace("deep navy blue and warm gold", palette)
        return f"{' '.join(scene.split()[:60])}, {suffix}"

    def _build_prompt(self, categorie, titre):
        sujet = sujet_neutre(titre)
        orders = (None,
                  (_llm_hf, _llm_groq, _llm_cloudflare),
                  (_llm_groq, _llm_cloudflare, _llm_hf))
        for attempt, order in enumerate(orders):
            t = time.time()
            text = self.donne_moi_prompt_images(categorie, sujet, attempt, order).content
            log(f"LLM : réponse prompt image (essai {attempt + 1}/{len(orders)}) "
                f"en {time.time() - t:.1f}s ({len(text or '')} caractères)")
            if not text or not text.strip():
                log("Réponse vide, nouvel essai avec un autre fournisseur.", level="WARN")
                continue
            try:
                positive, _ = self._parse_prompts(text)
            except Exception as e:
                log(f"Parseur de prompt : {type(e).__name__} {str(e)[:150]} -> essai suivant.",
                    level="WARN")
                positive = None
            if not self._valid_positive(positive):
                log(f"Prompt invalide (essai {attempt + 1}) : "
                    f"{re.sub(chr(10), ' ', text)[:200]!r}", level="WARN")
                continue
            sur, pourquoi = self._prompt_sur(positive, titre)
            if sur:
                log("LLM : prompt photo reçu, validé, sans objet écrit.", level="OK")
                return self._assembler(positive)
            log(f"Prompt REJETÉ ({pourquoi}) : {positive[:160]!r}", level="WARN")

        scene = random.choice(SAFE_SCENES.get(categorie) or SAFE_SCENES["Market impact"])
        log("Aucun prompt LLM sûr : scène de la liste blanche utilisée.", level="WARN")
        return self._assembler(scene)

    # ------------------------------------------------------------------
    # 2) Génération d'image IA : Cloudflare (comptes x modèles) -> HF (tokens x modèles)
    # ------------------------------------------------------------------
    def _image_cloudflare(self, prompt, seed=None):
        accounts = cf_accounts()
        if not accounts:
            if "cf_absent" not in _DEAD:
                log("Aucun compte Cloudflare configuré -> Cloudflare ignoré.", level="WARN")
                _DEAD.add("cf_absent")
            return None

        for label, account, token in accounts:
            dk = f"cf:{label}"
            if dk in _DEAD:
                continue
            headers = {"Authorization": f"Bearer {token}"}
            for model in CF_IMAGE_MODELS:
                url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
                if "flux" in model:
                    body = {"prompt": prompt[:2000], "steps": 6}
                else:   # Stable Diffusion XL : taille explicite, retour binaire
                    body = {"prompt": prompt[:2000], "negative_prompt": NEGATIVE_PROMPT,
                            "width": 1024, "height": 1280,
                            "num_steps": 8 if "lightning" in model else 20}
                if seed is not None:
                    body["seed"] = int(seed) % 2_000_000_000
                t = time.time()
                try:
                    log(f"Cloudflare compte {label} : {model}…")
                    r = requests.post(url, headers=headers, json=body, timeout=120)
                    if r.status_code == 400 and "seed" in body and not _quota_text(r.text):
                        body.pop("seed")
                        r = requests.post(url, headers=headers, json=body, timeout=120)
                    if r.status_code in (401, 403):
                        log(f"Cloudflare compte {label} : accès refusé (HTTP {r.status_code}) "
                            "-> compte suivant.", level="WARN")
                        _DEAD.add(dk)
                        break
                    if r.status_code == 429 or (r.status_code >= 400 and _quota_text(r.text)):
                        log(f"Cloudflare compte {label} : quota atteint -> compte suivant.",
                            level="WARN")
                        _DEAD.add(dk)
                        break
                    r.raise_for_status()
                    if r.headers.get("content-type", "").startswith("image/"):
                        raw = r.content
                    else:
                        b64 = (r.json().get("result") or {}).get("image")
                        if not b64:
                            log(f"Cloudflare {model} : réponse sans image.", level="WARN")
                            continue
                        raw = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    log(f"Cloudflare compte {label} OK ({model}) en {time.time() - t:.1f}s -> "
                        f"{img.size[0]}x{img.size[1]}", level="OK")
                    return img
                except Exception as e:
                    log(f"Cloudflare {label} {model} : {type(e).__name__} {str(e)[:200]}",
                        level="WARN")
        return None

    def _image_hf(self, prompt, seed=None):
        tokens = hf_tokens()
        if not tokens:
            if "hf_absent" not in _DEAD:
                log("Aucun HF_TOKEN configuré -> Hugging Face ignoré.", level="WARN")
                _DEAD.add("hf_absent")
            return None

        for label, token in tokens:
            dk = f"hf:{label}"
            if dk in _DEAD:
                continue
            client = InferenceClient(api_key=token)
            for model in HF_IMAGE_MODELS:
                is_fast = "schnell" in model.lower() or "turbo" in model.lower()
                params = {"width": IMG_W, "height": IMG_H,
                          "num_inference_steps": 4 if is_fast else 30}
                if not is_fast:
                    params["guidance_scale"] = 3.5
                    params["negative_prompt"] = NEGATIVE_PROMPT
                if seed is not None:
                    params["seed"] = seed
                attempts = [params, {"width": IMG_W, "height": IMG_H}]
                for i, p in enumerate(attempts):
                    t = time.time()
                    try:
                        log(f"HF compte {label} : {model} (essai {i + 1}/{len(attempts)})")
                        image = client.text_to_image(prompt, model=model, **p)
                        log(f"HF compte {label} OK ({model}) en {time.time() - t:.1f}s -> "
                            f"{image.size[0]}x{image.size[1]}", level="OK")
                        return image.convert("RGB")
                    except Exception as e:
                        s = str(e)
                        log(f"HF {label} {model} essai {i + 1} : {type(e).__name__} {s[:160]}",
                            level="WARN")
                        if "402" in s or "Payment Required" in s:
                            log(f"HF compte {label} : crédits épuisés -> compte suivant.",
                                level="WARN")
                            _DEAD.add(dk)
                            break
                        if "401" in s or "403" in s:
                            log(f"HF compte {label} : token refusé -> compte suivant.",
                                level="WARN")
                            _DEAD.add(dk)
                            break
                        time.sleep(1)
                if dk in _DEAD:
                    break
        return None

    def _generate_one(self, prompt, seed):
        services = (
            ("Cloudflare", self._image_cloudflare),
            ("Hugging Face", self._image_hf),
        )
        for nom, fonction in services:
            img = fonction(prompt, seed)
            if img is not None:
                return img, nom
            log(f"{nom} : pas d'image, service suivant…", level="WARN")
        return None, None

    # ------------------------------------------------------------------
    # 3) Contrôle qualité + anti-écriture (OCR) + sélection du meilleur candidat
    # ------------------------------------------------------------------
    @staticmethod
    def _score_image(img):
        small = img.resize((512, int(512 * img.height / img.width)))
        gray = small.convert("L")
        edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
        sharp = edges.var[0] ** 0.5
        contrast = ImageStat.Stat(gray).stddev[0]
        mean = ImageStat.Stat(gray).mean[0]
        penalty = 0
        if mean < 45:
            penalty = (45 - mean) * 1.5
        elif mean > 200:
            penalty = (mean - 200) * 1.5
        return sharp * 1.0 + contrast * 0.6 - penalty

    _ocr_alerte = False

    @classmethod
    def _ecriture_detectee(cls, img):
        """Mots lisibles trouvés dans l'image IA (liste vide = image propre). Nécessite tesseract."""
        if not OCR_CHECK:
            return []
        try:
            import pytesseract
            gray = ImageOps.autocontrast(img.convert("L"))
            data = pytesseract.image_to_data(gray, config="--psm 11", timeout=40,
                                             output_type=pytesseract.Output.DICT)
            mots = []
            for t, c in zip(data["text"], data["conf"]):
                try:
                    conf = float(c)
                except (TypeError, ValueError):
                    continue
                t = (t or "").strip()
                if conf >= OCR_MIN_CONF and sum(ch.isalnum() for ch in t) >= 3:
                    mots.append(t)
            return mots
        except Exception as e:
            if not cls._ocr_alerte:
                cls._ocr_alerte = True
                log(f"OCR indisponible ({type(e).__name__}) : contrôle d'écriture IGNORÉ. "
                    "Installe tesseract-ocr et pytesseract.", level="WARN")
            return []

    def _best_image(self, prompt):
        """Retourne (image, statut) ; statut : 'ok' | 'texte' (toutes écrites) | 'aucune'."""
        propres, rejetees, essais = [], 0, 0
        voulu = max(1, IMAGE_CANDIDATES)
        while len(propres) < voulu and essais < IMAGE_MAX_TRIES:
            essais += 1
            log(f"Candidat {essais}/{IMAGE_MAX_TRIES} : génération…")
            seed = int(time.time() * 1000) % 2_000_000_000 + essais * 7919
            img, source = self._generate_one(prompt, seed)
            if img is None or img.size[0] < 512:
                log(f"Candidat {essais} : aucune image exploitable.", level="WARN")
                if essais == 1:      # aucun service ne répond : inutile de réessayer
                    break
                continue
            mots = self._ecriture_detectee(img)
            if mots:
                rejetees += 1
                log(f"Candidat {essais} REJETÉ : écriture détectée par OCR {mots[:6]}", level="WARN")
                continue
            score = self._score_image(img)
            log(f"Candidat {essais} propre : {source} | {img.size[0]}x{img.size[1]} | "
                f"score {score:.1f}", level="OK")
            propres.append((score, img))
        if propres:
            best = max(propres, key=lambda c: c[0])
            log(f"Meilleur candidat retenu (score {best[0]:.1f}).")
            return best[1], "ok"
        return None, ("texte" if rejetees else "aucune")

    @staticmethod
    def _fond_local(titre):
        """Dernier recours (aucune photo web, aucune image IA propre) : fond dessiné par Pillow
        (dégradé + grille + chandeliers). Sans écriture par construction."""
        w, h = OUT_W, OUT_H
        pals = (((5, 14, 46), (14, 52, 120), (255, 196, 0)),
                ((4, 30, 24), (12, 84, 62), (205, 214, 224)),
                ((28, 8, 52), (82, 34, 120), (222, 140, 72)))
        seed = zlib.crc32((titre or "").encode("utf-8"))
        top, bottom, accent = pals[seed % 3]
        v = Image.linear_gradient("L").resize((w, h), Image.BILINEAR)
        img = Image.composite(Image.new("RGB", (w, h), bottom), Image.new("RGB", (w, h), top), v)
        ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for x in range(0, w, 90):
            d.line([(x, 0), (x, h)], fill=(255, 255, 255, 12), width=1)
        for y in range(0, h, 90):
            d.line([(0, y), (w, y)], fill=(255, 255, 255, 12), width=1)
        rnd = random.Random(seed)
        y, x, step = rnd.randint(480, 620), -10, 66
        while x < w + step:
            o = y
            c = max(220, min(760, y + rnd.uniform(-52, 38)))
            hi = min(o, c) - rnd.randint(8, 40)
            lo = max(o, c) + rnd.randint(8, 40)
            col = accent + (46,) if c < o else (255, 255, 255, 28)
            d.line([(x + 16, hi), (x + 16, lo)], fill=col, width=3)
            d.rectangle([x, min(o, c), x + 32, max(o, c) + 2], fill=col)
            y, x = c, x + step
        return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")

    # ------------------------------------------------------------------
    # 4) Post-traitement
    # ------------------------------------------------------------------
    @staticmethod
    def _enhance_image(img):
        img = img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=60, threshold=3))
        img = ImageEnhance.Contrast(img).enhance(1.07)
        img = ImageEnhance.Color(img).enhance(1.05)
        img = ImageEnhance.Sharpness(img).enhance(1.15)

        w, h = img.size
        mask = Image.radial_gradient("L").resize((w, h), Image.LANCZOS)
        mask = mask.point(lambda p: int(max(0, p - 90) * 0.55))
        img = Image.composite(Image.new("RGB", (w, h), (3, 8, 22)), img, mask)

        noise = Image.effect_noise((w, h), 14).convert("RGB")
        img = Image.blend(img, ImageChops.soft_light(img, noise), 0.30)
        return img

    @staticmethod
    def _load_font(size):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
        for chemin in candidates:
            if os.path.exists(chemin):
                return ImageFont.truetype(chemin, size)
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    # ------------------------------------------------------------------
    # Titre arabe + branding
    # ------------------------------------------------------------------
    _ar_mode = None
    _AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
    _NUM_RE = re.compile(r"([0-9\u0660-\u0669][0-9\u0660-\u0669.,%]*)")

    @staticmethod
    def _ar_font_candidates():
        return [
            AR_FONT_ENV,
            str(BASE_DIR / "fonts" / "Cairo-Bold.ttf"),
            str(BASE_DIR / "fonts" / "Tajawal-Bold.ttf"),
            str(BASE_DIR / "fonts" / "NotoSansArabic-Bold.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]

    @classmethod
    def _open_ar_font(cls, size, mode):
        layout_cls = getattr(ImageFont, "Layout", None)
        layout = None
        if layout_cls is not None:
            layout = layout_cls.RAQM if mode == "raqm" else layout_cls.BASIC
        for chemin in cls._ar_font_candidates():
            if chemin and os.path.exists(chemin):
                try:
                    if layout is not None:
                        return ImageFont.truetype(chemin, size, layout_engine=layout)
                    return ImageFont.truetype(chemin, size)
                except Exception:
                    continue
        raise RuntimeError("Aucune police arabe trouvée")

    @classmethod
    def _load_arabic_font(cls, size):
        mode = cls._detect_ar_mode()
        # Police latine un peu plus petite : même taille visuelle que l'arabe (Cairo/Noto).
        latin = TextPost._load_font(int(size * 0.92))
        try:
            return _FontPair(cls._open_ar_font(size, mode), latin)
        except Exception:
            log("ATTENTION : aucune police arabe trouvée (mets Cairo-Bold.ttf dans ./fonts/)")
            return _FontPair(latin, latin)

    @classmethod
    def _ar_word_visual(cls, word, mode):
        if mode == "raqm" or not ARABIC_OK:
            return word
        shaped = arabic_reshaper.reshape(word)
        if mode == "bidi":
            return get_display(shaped, base_dir="R")
        parts = cls._NUM_RE.split(shaped)
        pieces = [p if i % 2 else p[::-1] for i, p in enumerate(parts)]
        return "".join(reversed(pieces))

    @classmethod
    def _ar_selftest(cls, mode):
        font = cls._open_ar_font(100, mode)
        kw = {"direction": "rtl", "language": "ar"} if mode == "raqm" else {}
        img = Image.new("L", (400, 260), 0)
        ImageDraw.Draw(img).text(
            (60, 40), cls._ar_word_visual("اب", mode), font=font, fill=255, **kw
        )
        bbox = img.getbbox()
        if not bbox:
            return False
        x0, y0, x1, y1 = bbox
        strip = max(4, (x1 - x0) // 5)
        left = img.crop((x0, y0, x0 + strip, y1)).getbbox()
        right = img.crop((x1 - strip, y0, x1, y1)).getbbox()
        if not left or not right:
            return False
        return right[1] < left[1] - 5

    @classmethod
    def _detect_ar_mode(cls):
        if cls._ar_mode:
            return cls._ar_mode
        candidates = []
        try:
            if features.check("raqm") and features.check("fribidi") and features.check("harfbuzz"):
                candidates.append("raqm")
        except Exception:
            pass
        if ARABIC_OK:
            candidates += ["bidi", "manual"]

        chosen = None
        for mode in candidates:
            try:
                if cls._ar_selftest(mode):
                    chosen = mode
                    break
                log(f"Mode arabe '{mode}' : rendu incorrect, essai suivant.")
            except Exception as e:
                log(f"Mode arabe '{mode}' indisponible : {e}")
        if chosen is None:
            chosen = "bidi" if ARABIC_OK else "plain"
            log("ATTENTION : aucun mode arabe validé, l'affichage peut être incorrect.")
        cls._ar_mode = chosen
        log(f"Mode d'écriture arabe retenu : {chosen}")
        return chosen

    @classmethod
    def _ar_runs(cls, text):
        runs = []
        # Les caractères que la police ne sait pas dessiner sont supprimés (plus de carrés □□□).
        for tok in nettoyer_pour_image(text).split():
            is_ar = bool(cls._AR_RE.search(tok))
            if runs and not is_ar and not runs[-1][1]:
                runs[-1] = (runs[-1][0] + " " + tok, False)
            else:
                runs.append((tok, is_ar))
        return runs

    @classmethod
    def _run_args(cls, run, is_ar):
        mode = cls._detect_ar_mode()
        if is_ar:
            kw = {"direction": "rtl", "language": "ar"} if mode == "raqm" else {}
            return cls._ar_word_visual(run, mode), kw
        return run, ({"direction": "ltr"} if mode == "raqm" else {})

    def _line_width(self, draw, text, font):
        runs = self._ar_runs(text)
        space = draw.textlength(" ", font=font.ar)
        total = 0
        for run, is_ar in runs:
            s, kw = self._run_args(run, is_ar)
            total += draw.textlength(s, font=font.pick(is_ar), **kw)
        return total + space * max(0, len(runs) - 1)

    def _draw_rtl_line(self, draw, right_x, y, text, font, fill):
        """Tous les mots (arabes, latins, chiffres) sont posés sur la MÊME ligne de base."""
        space = draw.textlength(" ", font=font.ar)
        baseline = y + font.ar.getmetrics()[0]       # ascent de la police arabe = ligne de base commune
        x = right_x
        for run, is_ar in self._ar_runs(text):
            s, kw = self._run_args(run, is_ar)
            f = font.pick(is_ar)
            rw = draw.textlength(s, font=f, **kw)
            x -= rw
            draw.text((x, baseline), s, font=f, fill=fill, anchor="ls", **kw)
            x -= space

    def _wrap(self, draw, text, font, max_w):
        lines, cur = [], []
        for word in text.split():
            test = " ".join(cur + [word])
            if cur and self._line_width(draw, test, font) > max_w:
                lines.append(" ".join(cur))
                cur = [word]
            else:
                cur.append(word)
        if cur:
            lines.append(" ".join(cur))
        return lines

    def _fit_headline(self, draw, text, max_w, max_h, w):
        size, min_size, max_lines = int(w * 0.078), int(w * 0.05), 4
        while True:
            font = self._load_arabic_font(size)
            lines = self._wrap(draw, text, font, max_w)
            line_h = int(size * 1.4)
            fits = len(lines) <= max_lines and len(lines) * line_h <= max_h
            if fits or size <= min_size:
                break
            size -= 2
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip(" ،,.") + "…"
        return font, lines, line_h

    @staticmethod
    def _load_logo(w, h):
        try:
            if not LOGO_PATH.is_file():
                log(f"Logo introuvable ({LOGO_PATH}) : logo texte utilisé à la place.")
                return None
            logo = Image.open(LOGO_PATH).convert("RGBA")
            ratio = min(int(w * 0.24) / logo.width, int(h * 0.11) / logo.height)
            size = (max(1, int(logo.width * ratio)), max(1, int(logo.height * ratio)))
            return logo.resize(size, Image.LANCZOS)
        except Exception as e:
            log(f"Logo illisible : {e}")
            return None

    def _add_branding(self, img, headline=None):
        img = img.convert("RGBA")
        w, h = img.size
        margin = int(w * 0.07)
        has_title = bool(headline and headline.strip())
        logo = self._load_logo(w, h)

        grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grad)
        bottom_h = int(h * (0.50 if has_title else 0.28))
        for i in range(bottom_h):
            a = int(235 * (i / bottom_h) ** 1.5)
            gd.line([(0, h - bottom_h + i), (w, h - bottom_h + i)], fill=(4, 10, 28, a))
        top_h = int(h * 0.16)
        for i in range(top_h):
            a = int(160 * (1 - i / top_h) ** 1.6)
            gd.line([(0, i), (w, i)], fill=(4, 10, 28, a))
        img = Image.alpha_composite(img, grad)
        probe = ImageDraw.Draw(img)

        lx, ly = int(w * 0.04), int(h * 0.03)
        part_a, part_b = BRAND_NAME

        if logo is not None:
            logo_box = [lx, ly, lx + logo.width, ly + logo.height]
        else:
            f_logo = self._load_font(int(w * 0.046))
            wa = probe.textlength(part_a, font=f_logo)
            wb = probe.textlength(part_b, font=f_logo)
            logo_bb = f_logo.getbbox(part_a + part_b)
            pad_x, pad_y = int(w * 0.022), int(w * 0.013)
            logo_box = [lx, ly, lx + wa + wb + 2 * pad_x, ly + logo_bb[3] + 2 * pad_y]

        f_sub = self._load_font(int(w * 0.027))
        handle_y = h - int(h * 0.055)

        title_data = None
        if has_title:
            bottom = handle_y - int(h * 0.028)
            font, lines, line_h = self._fit_headline(
                probe, headline.strip(), w - 2 * margin, int(h * 0.30), w
            )
            top = bottom - len(lines) * line_h

            f_tag = self._load_font(int(w * 0.038))
            ta = probe.textlength(part_a, font=f_tag)
            tb_w = probe.textlength(part_b, font=f_tag)
            tbb = f_tag.getbbox(part_a + part_b)
            tp_x, tp_y = int(w * 0.022), int(w * 0.011)
            bar_w = max(8, int(w * 0.009))
            box_w = int(ta + tb_w) + 2 * tp_x + bar_w
            box_h = tbb[3] + 2 * tp_y
            tag_right = w - margin
            gap = int(h * 0.022)
            tag_box = [tag_right - box_w, top - gap - box_h, tag_right, top - gap]
            title_data = (font, lines, line_h, top, f_tag, ta, tag_box, tp_x, tp_y, bar_w)

        soft = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(soft)
        if logo is not None:
            shadow = Image.new("RGBA", logo.size, (0, 0, 0, 0))
            shadow.putalpha(logo.getchannel("A").point(lambda a: int(a * 0.75)))
            soft.alpha_composite(shadow, (lx + 3, ly + 4))
        if title_data:
            font, lines, line_h, top = title_data[:4]
            for i, line in enumerate(lines):
                self._draw_rtl_line(sd, w - margin, top + i * line_h + 4, line, font,
                                    (0, 0, 0, 220))
        soft = soft.filter(ImageFilter.GaussianBlur(5))
        img = Image.alpha_composite(img, soft)

        crisp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        cd = ImageDraw.Draw(crisp)
        if logo is None:
            cd.rectangle(logo_box, fill=(4, 10, 28, 190), outline=(255, 255, 255, 220), width=2)
        elif LOGO_BACKDROP:
            p = int(w * 0.012)
            cd.rectangle([logo_box[0] - p, logo_box[1] - p, logo_box[2] + p, logo_box[3] + p],
                         fill=(4, 10, 28, 190), outline=(255, 255, 255, 220), width=2)
        img = Image.alpha_composite(img, crisp)

        if logo is not None:
            img.alpha_composite(logo, (lx, ly))
        draw = ImageDraw.Draw(img)

        if logo is None:
            draw.text((lx + pad_x, ly + pad_y), part_a, font=f_logo, fill=(255, 255, 255, 255))
            draw.text((lx + pad_x + wa, ly + pad_y), part_b, font=f_logo,
                      fill=BRAND_ACCENT + (255,))
            draw.rectangle([logo_box[0], logo_box[3] - 5, logo_box[2], logo_box[3]],
                           fill=BRAND_ACCENT + (255,))

        if title_data:
            font, lines, line_h, top, f_tag, ta, tag_box, tp_x, tp_y, bar_w = title_data
            draw.rectangle(tag_box, fill=(255, 255, 255, 255))
            draw.rectangle([tag_box[2] - bar_w, tag_box[1], tag_box[2], tag_box[3]],
                           fill=BRAND_ACCENT + (255,))
            draw.text((tag_box[0] + tp_x, tag_box[1] + tp_y), part_a, font=f_tag,
                      fill=(5, 12, 30, 255))
            draw.text((tag_box[0] + tp_x + ta, tag_box[1] + tp_y), part_b, font=f_tag,
                      fill=(214, 148, 0, 255))
            for i, line in enumerate(lines):
                self._draw_rtl_line(draw, w - margin, top + i * line_h, line, font,
                                    (255, 255, 255, 255))

        draw.text((int(w * 0.04), handle_y), BRAND_HANDLE, font=f_sub,
                  fill=(225, 230, 240, 235))

        return img.convert("RGB")

    # ------------------------------------------------------------------
    # 5) Pipeline image complet : photo web -> image IA vérifiée -> fond local
    # ------------------------------------------------------------------
    def generer_image(self, news_original=None, headline=None, categorie=None, titre=""):
        t0 = time.time()
        self.last_image_origine, self.last_image_source, self.last_credit = None, None, None
        img = None

        mode, queries = self._planifier_image(categorie, titre, news_original)
        log(f"Image 1/5 : plan = {mode.upper()} | requêtes photo : {queries}")

        if mode == "web":
            img, infos = self._image_web(queries)
            if img is not None:
                self.last_image_origine = "web"
                self.last_image_source = infos["url"]
                self.last_credit = infos["credit"]
                log(f"Image 2/5 : VRAIE photo du web ({infos['source']}).", level="OK")
            else:
                log("Aucune vraie photo web exploitable : repli sur l'image IA.", level="WARN")

        if img is None:
            log("Image 2/5 : prompt IA (sujet neutre + validation anti-écriture)…")
            prompt = self._build_prompt(categorie, titre)
            log(f"Image 3/5 : prompt final ({len(prompt.split())} mots) :")
            log("      " + prompt)
            img, statut = self._best_image(prompt)
            if img is not None:
                self.last_image_origine = "ia"
            else:
                log("Aucune image IA propre"
                    + (" (toutes contenaient de l'écriture)" if statut == "texte"
                       else " (aucun service ne répond)")
                    + " : fond local sans écriture utilisé, la publication continue.",
                    level="WARN")
                img = self._fond_local(titre or headline)
                self.last_image_origine = "local"

        log("Image 4/5 : recadrage 4:5, netteté, couleurs, vignette, grain…")
        img = ImageOps.fit(img, (OUT_W, OUT_H), Image.LANCZOS, centering=(0.5, 0.4))
        img = self._enhance_image(img)

        log("Image 5/5 : logo + titre arabe"
            + (f" : « {headline} »" if headline else " (aucun titre)"))
        img = self._add_branding(img, headline)

        img.save(OUTPUT_DIR / "img.png")
        output_jpg = OUTPUT_DIR / "img.jpg"
        img.save(output_jpg, format="JPEG", quality=95, optimize=True, subsampling=0)
        log(f"Image prête ({self.last_image_origine}) : {output_jpg.name} "
            f"({output_jpg.stat().st_size // 1024} Ko, {img.size[0]}x{img.size[1]}) "
            f"en {time.time() - t0:.0f}s", level="OK")
        return output_jpg

    def _ajouter_credit(self, caption):
        """Ajoute « 📷 auteur / licence » en fin de légende si l'image vient du web."""
        if not self.last_credit:
            return caption
        ligne = "\n📷 " + " ".join(self.last_credit.split())[:150]
        if len(caption) + len(ligne) > IG_CAPTION_LIMIT:
            caption = caption[:IG_CAPTION_LIMIT - len(ligne) - 1].rstrip() + "…"
        return caption + ligne

    # ------------------------------------------------------------------
    # 6) Publication Facebook + Instagram (+ avertissement en 1er commentaire)
    # ------------------------------------------------------------------
    @staticmethod
    def _json_or_text(response):
        if "json" in response.headers.get("content-type", ""):
            try:
                return response.json()
            except Exception:
                pass
        return {"message": response.text[:500]}

    def _publier_facebook(self, chemin, caption, page_id, token):
        log(f"Facebook : envoi de la photo ({chemin.stat().st_size // 1024} Ko)…")
        try:
            with chemin.open("rb") as fichier:
                response = requests.post(
                    f"{GRAPH_URL}/{page_id}/photos",
                    data={"message": caption, "access_token": token},
                    files={"source": (chemin.name, fichier, "image/jpeg")},
                    timeout=120,
                )
            data = self._json_or_text(response)
            log(f"Facebook : réponse HTTP {response.status_code}")
            if response.ok and "error" not in data:
                pid = data.get("post_id") or data.get("id")
                log(f"Facebook : PUBLIÉ (id={pid})", level="OK")
                poster_commentaire(pid, DISCLAIMER, token, "Facebook")
                return pid or True
            log("Facebook : ÉCHEC -> " + explain_graph_error(data), level="ERROR")
        except Exception as e:
            log(f"Facebook : exception {type(e).__name__} : {str(e)[:300]}", level="ERROR")
        return False

    def _publier_instagram(self, chemin, caption, ig_id, token, cloud):
        try:
            cloudinary.config(cloud_name=cloud[0], api_key=cloud[1], api_secret=cloud[2],
                              secure=True)
            log("Instagram 1/4 : envoi de l'image sur Cloudinary…")
            upload = cloudinary.uploader.upload(
                str(chemin), resource_type="image", folder="actualites_financieres"
            )
            image_url = upload.get("secure_url")
            if not (image_url and image_url.startswith("https://")):
                log(f"Instagram : Cloudinary n'a pas renvoyé d'URL valide : {upload}", level="ERROR")
                return False
            log(f"Instagram 1/4 : image hébergée -> {image_url}", level="OK")
            self.last_image_url = image_url

            log("Instagram 2/4 : création du conteneur média…")
            r = requests.post(
                f"{GRAPH_URL}/{ig_id}/media",
                data={"image_url": image_url, "caption": caption[:IG_CAPTION_LIMIT],
                      "access_token": token},
                timeout=90,
            )
            data = self._json_or_text(r)
            log(f"Instagram 2/4 : HTTP {r.status_code}")
            if not (r.ok and "error" not in data and data.get("id")):
                log("Instagram : ÉCHEC création -> " + explain_graph_error(data), level="ERROR")
                return False
            creation_id = data["id"]
            log(f"Instagram 2/4 : conteneur créé (id={creation_id})", level="OK")

            log("Instagram 3/4 : attente du traitement par Instagram…")
            statut = None
            for i in range(12):
                sr = requests.get(
                    f"{GRAPH_URL}/{creation_id}",
                    params={"fields": "status_code", "access_token": token},
                    timeout=30,
                )
                statut = sr.json().get("status_code")
                log(f"      contrôle {i + 1}/12 : {statut}")
                if statut in ("FINISHED", "ERROR"):
                    break
                time.sleep(5)
            if statut != "FINISHED":
                log(f"Instagram : le média n'est pas prêt (statut final : {statut})", level="ERROR")
                return False

            log("Instagram 4/4 : publication…")
            pr = requests.post(
                f"{GRAPH_URL}/{ig_id}/media_publish",
                data={"creation_id": creation_id, "access_token": token},
                timeout=90,
            )
            pdata = self._json_or_text(pr)
            log(f"Instagram 4/4 : HTTP {pr.status_code}")
            if pr.ok and "error" not in pdata:
                media_id = pdata.get("id")
                log(f"Instagram : PUBLIÉ (id={media_id})", level="OK")
                poster_commentaire(media_id, DISCLAIMER, token, "Instagram")
                return media_id or True
            log("Instagram : ÉCHEC publication -> " + explain_graph_error(pdata), level="ERROR")
        except Exception as e:
            log(f"Instagram : exception {type(e).__name__} : {str(e)[:300]}", level="ERROR")
        return False

    def publier_texte_partout(self, news_original, news_translited, headline=None,
                              categorie=None, titre=""):
        resultats = {"image": None, "facebook": False, "instagram": False, "erreur": None}

        try:
            self.last_image_url = None
            log("── Génération de l'image ──", level="STEP")
            chemin = Path(self.generer_image(news_original=news_original, headline=headline,
                                             categorie=categorie, titre=titre))
            if not chemin.is_file():
                raise FileNotFoundError(f"Image générée introuvable : {chemin.resolve()}")
            resultats["image"] = str(chemin.resolve())
            legende = self._ajouter_credit(news_translited)

            page_id = menv("PAGE_ID")
            token = menv("PAGE_ACCESS_TOKEN")
            ig_id = menv("INSTAGRAM_BUSINESS_ID")
            cloud = (os.getenv("CLOUDINARY_CLOUD_NAME", ""), os.getenv("CLOUDINARY_API_KEY", ""),
                     os.getenv("CLOUDINARY_API_SECRET", ""))
            if not token:
                raise ValueError(f"PAGE_ACCESS_TOKEN{META_SUFFIX} est absent "
                                 "(secret non transmis au job ?)")

            log("── Publication Facebook ──", level="STEP")
            if page_id:
                resultats["facebook"] = self._publier_facebook(chemin, legende, page_id, token)
            else:
                log(f"Facebook IGNORÉ : PAGE_ID{META_SUFFIX} est vide.", level="WARN")

            log("── Publication Instagram ──", level="STEP")
            manque = [n for n, v in ((f"INSTAGRAM_BUSINESS_ID{META_SUFFIX}", ig_id),
                                     ("CLOUDINARY_CLOUD_NAME", cloud[0]),
                                     ("CLOUDINARY_API_KEY", cloud[1]),
                                     ("CLOUDINARY_API_SECRET", cloud[2])) if not v]
            if manque:
                log("Instagram IGNORÉ : variable(s) vide(s) -> " + ", ".join(manque), level="WARN")
            else:
                resultats["instagram"] = self._publier_instagram(chemin, legende,
                                                                 ig_id, token, cloud)
        except Exception as e:
            resultats["erreur"] = f"{type(e).__name__}: {e}"
            log(f"Échec de la génération/préparation : {resultats['erreur']}", level="ERROR")
            log(traceback.format_exc(limit=4))

        return resultats


# ======================================================================
# EXTRACTION DES NEWS + FILTRAGE ÉDITORIAL + TRADUCTION
# ======================================================================
class Extract_post:
    def __init__(self):
        self.TRUSTED_WEB_SOURCES = TRUSTED_WEB_SOURCES

    def clean_html(self, text):
        if not text:
            return ""
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        return soup.get_text(separator=" ", strip=True)

    def get_publication_date(self, entry):
        date_struct = entry.get("published_parsed") or entry.get("updated_parsed")
        if not date_struct:
            return None, None
        date_utc = datetime(*date_struct[:6], tzinfo=timezone.utc)
        return date_utc.strftime("%Y-%m-%d %H:%M"), date_utc

    def fetch_all_news(self):
        articles = []
        seen_links = set()
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }

        for source_name, feed_url in self.TRUSTED_WEB_SOURCES.items():
            t = time.time()
            log(f"[{source_name}] lecture de {feed_url}")
            try:
                resp = requests.get(feed_url, headers=headers, timeout=30)
                log(f"[{source_name}] HTTP {resp.status_code} | {len(resp.content) // 1024} Ko | "
                    f"{resp.headers.get('content-type', '?')} | {time.time() - t:.1f}s")
                if resp.status_code >= 400:
                    log(f"[{source_name}] flux INACCESSIBLE. Début de la réponse : "
                        f"{resp.text[:150]!r}", level="WARN")
                    STATS["flux_ko"] += 1
                    continue

                feed = feedparser.parse(resp.content)
                if getattr(feed, "bozo", 0):
                    log(f"[{source_name}] XML imparfait : "
                        f"{str(getattr(feed, 'bozo_exception', ''))[:150]}", level="WARN")
                if not feed.entries:
                    log(f"[{source_name}] 0 article dans le flux (page de blocage ?). "
                        f"Début de la réponse : {resp.text[:150]!r}", level="WARN")
                    STATS["flux_ko"] += 1
                    continue

                ajoutes = 0
                for entry in feed.entries:
                    link = entry.get("link", "").strip()
                    title = entry.get("title", "").strip()

                    if not title or not link or link in seen_links:
                        continue

                    raw_text = (
                        entry.get("content", [{}])[0].get("value", "")
                        if entry.get("content")
                        else entry.get("summary", "")
                    )

                    clean_text = self.clean_html(raw_text)
                    clean_text = re.sub(r"\s*The post .*? appeared first on .*$", "",
                                        clean_text, flags=re.S)
                    clean_text = clean_text.replace("[...]", "…").strip()
                    publication_text, publication_datetime = self.get_publication_date(entry)

                    if publication_text is None:
                        continue

                    seen_links.add(link)
                    ajoutes += 1
                    articles.append({
                        "source": source_name,
                        "title": self.clean_html(title),
                        "date": publication_text,
                        "continue": clean_text,
                        "link": link,
                        "datetime": publication_datetime,
                    })
                log(f"[{source_name}] {ajoutes} article(s) retenu(s) sur {len(feed.entries)}",
                    level="OK")
                STATS["flux_ok"] += 1
            except Exception as error:
                log(f"[{source_name}] erreur {type(error).__name__} : {str(error)[:300]}",
                    level="ERROR")
                STATS["flux_ko"] += 1

        articles.sort(key=lambda item: item["datetime"], reverse=True)
        return articles

    def get_article_content(self, news):
        """Récupère le corps de l'article lorsque le site l'autorise; sinon retourne l'extrait RSS."""
        rss_text = (news.get("continue") or "").strip()
        link = (news.get("link") or "").strip()
        if not link.startswith(("http://", "https://")):
            return rss_text
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MoneyImpactBot/1.0)",
                   "Accept": "text/html,application/xhtml+xml"}
        try:
            response = requests.get(link, headers=headers, timeout=18)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                element.decompose()
            article = soup.find("article") or soup.find(attrs={"itemprop": "articleBody"}) or soup.find("main")
            if article:
                body = re.sub(r"\s+", " ", article.get_text(" ", strip=True)).strip()
                if len(body) >= 300:
                    log(f"Corps de l'article récupéré ({len(body)} caractères).", level="OK")
                    return body[:12000]
        except Exception as e:
            log(f"Lecture du corps de l'article impossible; validation sur l'extrait RSS seulement ({type(e).__name__}).",
                level="WARN")
        return rss_text

    # ------------------------------------------------------------------
    # Doublons : comparaison de titres (mots significatifs, similarité de Jaccard)
    # ------------------------------------------------------------------
    @staticmethod
    def _title_tokens(title):
        words = re.findall(r"[a-z0-9]+", (title or "").lower())
        return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)

    def titre_en_doublon(self, title, titres_connus):
        """Retourne le titre déjà connu (publié ou accepté pendant ce run) trop proche, sinon None."""
        a = self._title_tokens(title)
        if len(a) < 3:
            return None
        for autre in titres_connus:
            b = self._title_tokens(autre)
            if len(b) < 3:
                continue
            inter = len(a & b)
            union = len(a | b)
            if union and inter / union >= DUPLICATE_TITLE_SIMILARITY:
                return autre
        return None

    # ------------------------------------------------------------------
    # Pré-filtre gratuit (sans LLM)
    # ------------------------------------------------------------------
    @staticmethod
    def prefiltre_actualite(news, article_content):
        """Retourne (ok, raison). Rejette les contenus manifestement inutilisables."""
        title = news.get("title", "") or ""
        rss = (news.get("continue") or "").strip()
        body = (article_content or "").strip()
        head = (title + " " + body[:2500])

        if _BAD_TITLE_RE.search(title):
            return False, "titre de type appel de résultats / promotionnel / hors sujet"
        if _BAD_BODY_RE.search(head):
            return False, "texte de type transcription d'appel de résultats ou contenu sponsorisé"
        # Beaucoup de flux (Fed, FT, CNBC, Bloomberg) ne donnent qu'un court extrait :
        # seuil bas, le LLM juge ensuite (note >= MIN_NEWS_SCORE).
        if len(body) < 100 and len(rss) < 60:
            return False, "contenu trop mince pour établir un fait financier"
        return True, ""

    # ------------------------------------------------------------------
    # Validation éditoriale stricte par LLM (note /100, seuil MIN_NEWS_SCORE)
    # ------------------------------------------------------------------
    def valider_actualite(self, news, article_content):
        """Décision LLM stricte avant toute traduction, image ou publication.

        Retourne (accepte, categorie, raison, score)."""
        categories = ", ".join(ALLOWED_NEWS_CATEGORIES)
        types = ", ".join(ACCEPTED_ARTICLE_TYPES)
        prompt = f"""
You are the strict editorial gatekeeper for Money.impact, a trading and financial-markets news page.
Your job is to PROTECT the page quality: when in doubt, reject.

Allowed categories: {categories}.
Accepted article types: {types}.

ACCEPT only articles that report a CONCRETE, NEW financial event with a clear and direct market
connection: a price move, a data release (CPI, NFP, GDP, PMI...), a central-bank decision or
statement, an official company announcement that moves its stock (guidance, results, M&A, offering),
a commodity/bond/currency/crypto development, or a regulatory/geopolitical event with stated market impact.

REJECT (relevant=false) all of the following:
- Earnings-call transcripts, conference-call transcripts, Q&A sessions, investor-day or webinar texts.
- Opinion pieces, generic analysis without a new event, educational or "how to" content.
- Personal stories, lifestyle, celebrity, real estate anecdotes, consumer tips, human-interest stories.
- Promotional, sponsored, broker/product advertising or affiliate content.
- Articles that merely mention money, prices or a company without a meaningful market impact.
- Articles whose text is too thin, vague or incomplete to establish the financial event.
- Duplicated or recycled news that adds no new information.
Do NOT infer facts that are not in the provided text. A financial RSS source does NOT qualify an article by itself.

Scoring (0-100): 90-100 = major, clear, directly market-moving event; 75-89 = solid market news with a
clear event; below 75 = weak, indirect, vague or promotional. Be severe: the page only publishes >= {MIN_NEWS_SCORE}.

Return ONLY this JSON object, nothing else (no markdown):
{{"relevant": true, "score": 0, "category": "one allowed category", "article_type": "one accepted type or rejected type (e.g. earnings_call_transcript, promotional, opinion, lifestyle, other)", "financial_event": "one short sentence naming the concrete financial event", "reason": "short reason"}}
Use relevant=false and financial_event="" when there is no clear financial event.
"""
        content = (article_content or "").strip()
        user = (f"SOURCE: {news.get('source','')}\nTITLE: {news.get('title','')}\n"
                f"DATE: {news.get('date','')}\nURL: {news.get('link','')}\n"
                f"RSS EXCERPT: {news.get('continue','')}\nARTICLE TEXT: {content[:12000]}")
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
        raw = llm_chat(messages, temperature=0.0, max_tokens=500)
        if not raw:
            log("Aucune API LLM ne répond pendant la validation éditoriale : arrêt du workflow.", level="ERROR")
            terminer(1, "ÉCHEC (LLM indisponible pour valider les actualités)")
        try:
            data = self._extract_json(raw)

            relevant = data.get("relevant")
            if isinstance(relevant, str):
                low = relevant.strip().lower()
                if low in ("true", "false"):
                    relevant = (low == "true")
            if not isinstance(relevant, bool):
                raise ValueError("champ relevant absent ou non booléen")

            try:
                score = int(round(float(data.get("score"))))
            except (TypeError, ValueError):
                raise ValueError("champ score absent ou non numérique")
            score = max(0, min(100, score))

            category = str(data.get("category") or "").strip()
            article_type = str(data.get("article_type") or "").strip().lower()
            event = str(data.get("financial_event") or "").strip()
            reason = str(data.get("reason") or "aucune justification fournie").strip()

            if category:
                match = next((c for c in ALLOWED_NEWS_CATEGORIES
                              if c.lower() == category.lower()), None)
                category = match or ""

            # Décision finale : TOUTES les conditions doivent être réunies.
            motifs = []
            if not relevant:
                motifs.append("jugée non pertinente par le LLM")
            if score < MIN_NEWS_SCORE:
                motifs.append(f"note {score}/100 < {MIN_NEWS_SCORE}")
            if not category:
                motifs.append("catégorie absente ou non autorisée")
            if article_type not in ACCEPTED_ARTICLE_TYPES:
                motifs.append(f"type d'article refusé ({article_type or 'inconnu'})")
            if len(event) < 12:
                motifs.append("aucun événement financier concret identifié")

            accepte = not motifs
            if not accepte:
                reason = f"{reason} | " + "; ".join(motifs)

            log(f"Validation éditoriale : {'ACCEPTÉE' if accepte else 'REJETÉE'} | "
                f"note={score}/100 | catégorie={category or 'aucune'} | type={article_type or '?'} | "
                f"raison={reason[:220]}",
                level="OK" if accepte else "WARN")
            if accepte:
                log(f"Événement financier : {event[:200]}")
            return accepte, category, reason, score
        except Exception as e:
            log(f"Réponse LLM de validation inexploitable ({type(e).__name__}: {e}) : article rejeté par sécurité.",
                level="WARN")
            log(f"Réponse brute du LLM : {str(raw)[:300]!r}", level="WARN")
            return False, "", "validation LLM inexploitable", 0

    @staticmethod
    def _extract_json(text):
        """Extrait un objet JSON d'une réponse LLM, même entourée de Markdown/texte."""
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text[match.start():])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue

        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = ast.literal_eval(text[start:end + 1])
                if isinstance(data, dict):
                    return data
            except (ValueError, SyntaxError):
                pass

        raise ValueError("Aucun objet JSON valide trouvé dans la réponse du modèle")

    @staticmethod
    def hashtags_pour(category):
        """Hashtags de base + hashtags propres à la catégorie validée."""
        return (HASHTAGS_BASE + " " + HASHTAGS_CATEGORIE.get(category, "")).strip()

    @staticmethod
    def ajouter_source(caption, source, hashtags="", date="", reserve=0):
        """Légende + date + « المصدر: <source> » + hashtags, sans dépasser la limite Instagram.

        `reserve` laisse de la place pour la ligne de crédit photo ajoutée ensuite.
        L'avertissement n'est PAS dans la légende : il est posté en 1er commentaire."""
        caption = (caption or "").strip()
        # retire une source / des hashtags déjà écrits par le LLM
        caption = re.sub(r"\n*\s*(?:المصدر|Source)\s*[:：].*$", "", caption,
                         flags=re.I | re.S).rstrip()
        caption = re.sub(r"(?:\s*#\S+)+\s*$", "", caption).rstrip()
        suffixe = (f"\n\n📅 {str(date)[:10]} | المصدر: {source}" if date
                   else f"\n\nالمصدر: {source}")
        if hashtags:
            suffixe += f"\n\n{hashtags}"
        place = IG_CAPTION_LIMIT - len(suffixe) - 1 - reserve
        if len(caption) > place:
            caption = caption[:place].rstrip(" ،,.\n") + "…"
        return caption + suffixe

    def tradication_of_text(self, text):
        prompt = """
        You are a professional Arabic financial editor writing for a Moroccan / Arab audience of
        beginner traders and savers. Translate and adapt the news into natural, SIMPLE Modern
        Standard Arabic. Keep every fact faithful. Do not add facts, numbers or dates that are not
        in the source. Do not write the source name, the date or any disclaimer (added separately).
        Write ONLY in Arabic (Latin letters only for brand names and tickers). Never use Chinese,
        Cyrillic or any other script.
        Caption structure (about 1300 characters max):
        1) First line: a hook giving the key fact and why it matters (not a dry statement).
        2) Two or three short sentences with the key facts.
        3) A line starting with "💡 ماذا يعني لك؟" explaining in plain words the likely impact on
           ordinary people or traders. If the news clearly involves the US dollar, gold or Bitcoin,
           say in one clause what it means for it. Use cautious wording (قد / من المحتمل). Mention
           the dirham or Morocco ONLY if the news clearly involves the US dollar, euro, oil, gold
           or global interest rates, and never invent figures.
        4) A line starting with "👀 وماذا بعد؟" saying what to watch next. Use ONLY what the source
           text supports (a scheduled decision, a data release, a stated next step). If the text
           gives no concrete basis, leave "what_next" empty and do NOT write this line.
        5) A last line with a short question inviting comments, for example:
           "هل تتوقع ...؟ اكتب رأيك في التعليق 👇".
        Explain jargon in a few simple words. No investment advice, no buy/sell recommendation.
        Return ONLY one valid JSON object, with double quotes around every key and string.
        Do not use Markdown fences, comments, or text before/after the JSON.
        Required format:
        {"caption": "full adapted text", "headline": "short Arabic headline", "what_next": "the watch-next sentence, or empty string"}
        Headline: 5 to 9 Arabic words, one strong idea. Start with a question, a number or a
        warning when that is faithful to the facts. NEVER start with a dry verb such as انخفاض /
        ارتفاع / توسع / يعلن / تعلن. No emojis, hashtags, quotation marks or source name.
        """
        t = time.time()
        log("Traduction en arabe (Cloudflare -> Hugging Face -> Groq, tous comptes)…")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"SOURCE TEXT TO TRANSLATE:\n{text}"},
        ]

        providers = (
            ("Cloudflare", _llm_cloudflare),
            ("Hugging Face", _llm_hf),
            ("Groq", _llm_groq),
        )
        for provider_name, provider_fn in providers:
            try:
                raw = provider_fn(messages, temperature=0.0, max_tokens=2500)
            except Exception as e:
                log(f"Traduction {provider_name} : {type(e).__name__} {str(e)[:180]}", level="WARN")
                continue

            if not raw or not str(raw).strip():
                log(f"Traduction {provider_name} : réponse vide, fournisseur suivant.", level="WARN")
                continue

            try:
                data = self._extract_json(raw)
                caption = str(data.get("caption") or "").strip()
                headline = str(data.get("headline") or "").strip()
                if _FOREIGN_RE.search(caption + " " + headline):
                    raise ValueError("alphabet étranger (chinois/cyrillique...) dans la réponse")
                if not caption:
                    raise ValueError("champ 'caption' absent ou vide")
                if not headline:
                    headline = caption[:100].strip()
                    data["headline"] = headline
                data["caption"] = caption
                if REQUIRE_WHAT_NEXT and "what_next" in data \
                        and not str(data.get("what_next") or "").strip():
                    log("Règle « وماذا بعد؟ » : aucune suite concrète -> news non publiée.",
                        level="WARN")
                    return {"skip": True, "reason": "aucune suite concrète (« وماذا بعد؟ »)"}
                log(f"Traduction reçue via {provider_name} en {time.time() - t:.1f}s", level="OK")
                return data
            except Exception as e:
                excerpt = re.sub(r"\s+", " ", str(raw))[:350]
                log(f"Traduction {provider_name} : réponse refusée ({e}); extrait={excerpt!r}",
                    level="WARN")
                log(f"Nouvel essai avec le fournisseur suivant ({provider_name} a échoué).",
                    level="WARN")

        log("Traduction ÉCHOUÉE : aucun fournisseur n'a renvoyé un JSON exploitable.", level="ERROR")
        return None


# ======================================================================
# MAIN
# ======================================================================
TOTAL_STEPS = 7


def charger_historique(fichier_json):
    if os.path.exists(fichier_json):
        try:
            with open(fichier_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            data = data if isinstance(data, dict) else {}
            log(f"Historique : {len(data)} news déjà publiées ({os.path.basename(fichier_json)})")
            if IS_GHA and not data:
                log("Historique VIDE sur GitHub Actions : le fichier doit être commité après "
                    "chaque run, sinon les mêmes news seront republiées.", level="WARN")
            return data
        except Exception as e:
            log(f"Historique illisible ({e}) : repart de zéro.", level="WARN")
            return {}
    log("Historique : fichier absent -> création (premier lancement).", level="WARN")
    with open(fichier_json, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)
    return {}


def diagnostic_environnement():
    log(f"Python {sys.version.split()[0]} | Pillow {PIL.__version__} | "
        f"GitHub Actions : {'oui' if IS_GHA else 'non'}")
    log(f"Script : {Path(__file__).name} | version {SCRIPT_VERSION}")
    log(f"Heure UTC : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}")
    log(f"Comptes : Cloudflare={len(cf_accounts())} | Hugging Face={len(hf_tokens())} | "
        f"Groq={len(groq_keys())}")
    log(f"Modèles LLM : CF={CF_TEXT_MODELS[0]}+{len(CF_TEXT_MODELS) - 1} | "
        f"HF={HF_TEXT_MODELS[0]}+{len(HF_TEXT_MODELS) - 1} | Groq={GROQ_MODELS[0]}+{len(GROQ_MODELS) - 1}")
    log(f"Modèles image : CF={CF_IMAGE_MODELS[0]}+{len(CF_IMAGE_MODELS) - 1} | "
        f"HF={HF_IMAGE_MODELS[0]}+{len(HF_IMAGE_MODELS) - 1}")
    log(f"Réglages : candidats={IMAGE_CANDIDATES} | max posts={MAX_POSTS_PER_RUN} | "
        f"max tentatives={MAX_ATTEMPTS_PER_RUN} | âge max={MAX_NEWS_AGE_HOURS}h | "
        f"note min={MIN_NEWS_SCORE}/100 | "
        f"Graph={GRAPH_URL.rsplit('/', 1)[-1]} | page={META_SUFFIX or 'principale'}")
    log(f"Photos web : {'activées' if WEB_IMAGES_ENABLED else 'désactivées'} "
        "(Wikimedia Commons + Openverse, sans clé) | "
        f"règle « وماذا بعد؟ » : {'oui' if REQUIRE_WHAT_NEXT else 'non'} | "
        f"OCR anti-écriture : {'oui' if OCR_CHECK else 'non'}")
    if OCR_CHECK:
        try:
            import pytesseract
            log(f"OCR : tesseract {pytesseract.get_tesseract_version()} disponible", level="OK")
        except Exception as e:
            log(f"OCR indisponible ({type(e).__name__}) : installe tesseract-ocr + pytesseract "
                "(sinon le contrôle d'écriture des images IA est ignoré).", level="WARN")

    s = META_SUFFIX
    variables = [
        ("PAGE_ACCESS_TOKEN" + s, True, "token Meta (Facebook + Instagram)"),
        ("PAGE_ID" + s, False, "ID de la page Facebook -> sinon Facebook ignoré"),
        ("INSTAGRAM_BUSINESS_ID" + s, False, "ID Instagram -> sinon Instagram ignoré"),
        ("CLOUDINARY_CLOUD_NAME", False, "Cloudinary (requis pour Instagram)"),
        ("CLOUDINARY_API_KEY", False, "Cloudinary (requis pour Instagram)"),
        ("CLOUDINARY_API_SECRET", False, "Cloudinary (requis pour Instagram)"),
    ]
    manquants = []
    for nom, obligatoire, role in variables:
        if os.getenv(nom, ""):
            log(f"{nom:<34} présent        | {role}", level="OK")
        else:
            log(f"{nom:<34} ABSENT/VIDE     | {role}", level="ERROR" if obligatoire else "WARN")
            if obligatoire:
                manquants.append(nom)

    for label, acc, _ in cf_accounts():
        suffixe = "" if label == "0" else label
        log(f"Cloudflare compte {label:<2} présent (CF_ACCOUNT_ID{suffixe}/CF_API_TOKEN{suffixe})",
            level="OK")
    for label, _ in hf_tokens():
        log(f"Hugging Face jeton {label:<2} présent (HF_TOKEN{'' if label == '0' else label})",
            level="OK")
    for label, _ in groq_keys():
        log(f"Groq clé {label:<2} présente (TOKEN_AI_DESCRYB_IMG{'' if label == '0' else label})",
            level="OK")

    # YouTube (optionnel : sert seulement pour les Shorts, géré dans publie_video_youtub.py)
    yt = [n for n in ("ID_CLIENT_FB", "CODE_SECRET_DU_CLIENT_FB", "YOUTUBE_REFRESH_TOKEN")
          if not os.getenv(n, "")]
    if yt:
        log("YouTube Shorts : secret(s) absent(s) -> " + ", ".join(yt) + " (YouTube sera ignoré)",
            level="WARN")
    else:
        log("YouTube Shorts : secrets présents (ID_CLIENT_FB, CODE_SECRET_DU_CLIENT_FB, "
            "YOUTUBE_REFRESH_TOKEN)", level="OK")

    if not (cf_accounts() or hf_tokens() or groq_keys()):
        log("Aucun service LLM configuré (Cloudflare, Hugging Face ou Groq).", level="ERROR")
        manquants.append("CF_ACCOUNT_ID+CF_API_TOKEN ou HF_TOKEN ou TOKEN_AI_DESCRYB_IMG")
    if IS_GHA:
        log("   (un secret n'est visible du script que s'il est listé dans 'env:' du workflow)")

    log(f"Logo : {'trouvé' if LOGO_PATH.is_file() else 'ABSENT'} ({LOGO_PATH})",
        level="OK" if LOGO_PATH.is_file() else "WARN")
    police = next((p for p in TextPost._ar_font_candidates() if p and os.path.exists(p)), None)
    log(f"Police arabe : {police or 'AUCUNE (arabe illisible !)'}",
        level="OK" if police else "ERROR")
    log(f"Arabe : bibliothèques {'présentes' if ARABIC_OK else 'ABSENTES (pip install arabic-reshaper python-bidi)'}",
        level="OK" if ARABIC_OK else "WARN")
    TextPost._detect_ar_mode()
    return manquants


def verifier_meta():
    token = menv("PAGE_ACCESS_TOKEN")
    page_id, ig_id = menv("PAGE_ID"), menv("INSTAGRAM_BUSINESS_ID")
    fb_ok = ig_ok = False

    def appel(obj_id, champs):
        r = requests.get(f"{GRAPH_URL}/{obj_id}", params={"fields": champs, "access_token": token},
                         timeout=30)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"message": r.text[:300]}

    if page_id:
        try:
            code, data = appel(page_id, "id,name")
            if code == 200 and "error" not in data:
                log(f"Facebook : accès OK à la page « {data.get('name')} » (id {data.get('id')})",
                    level="OK")
                fb_ok = True
            else:
                log("Facebook : accès REFUSÉ -> " + explain_graph_error(data), level="ERROR")
        except Exception as e:
            log(f"Facebook : test impossible ({type(e).__name__}) : {str(e)[:200]}", level="ERROR")
    if ig_id:
        try:
            code, data = appel(ig_id, "id,username")
            if code == 200 and "error" not in data:
                log(f"Instagram : accès OK au compte @{data.get('username')}", level="OK")
                ig_ok = True
            else:
                log("Instagram : accès REFUSÉ -> " + explain_graph_error(data), level="ERROR")
        except Exception as e:
            log(f"Instagram : test impossible ({type(e).__name__}) : {str(e)[:200]}", level="ERROR")
    return fb_ok, ig_ok


def ecrire_resume(statut):
    lignes = [
        f"## Agent Facebook / Instagram — {statut}",
        "",
        "| Indicateur | Valeur |", "|---|---|",
        f"| Flux RSS OK / en échec | {STATS['flux_ok']} / {STATS['flux_ko']} |",
        f"| News récupérées | {STATS['news_total']} |",
        f"| Déjà publiées | {STATS['deja']} |",
        f"| Trop anciennes (> {MAX_NEWS_AGE_HOURS} h) | {STATS['anciennes']} |",
        f"| À examiner | {STATS['candidates']} |",
        f"| Tentatives | {STATS['tentatives']} |",
        f"| Rejetées (filtre éditorial) | {STATS['rejetees']} |",
        f"| **Publiées** | **{STATS['publies']}** |",
    ]
    if STATS["echecs"]:
        lignes += ["", "### Rejets / échecs"]
        lignes += [f"- {t[:80]} — {raison}" for t, raison in STATS["echecs"]]
    step_summary("\n".join(lignes))


_VIDEO_DONE = False


def lancer_video_une_fois():
    """Lance la vidéo Reel une seule fois par run (indépendante des news)."""
    global _VIDEO_DONE
    if _VIDEO_DONE:
        return
    _VIDEO_DONE = True
    publier_video_reel({})


def terminer(code, statut):
    """Point de sortie unique du script.

    La vidéo Reel est INDÉPENDANTE des news : elle est lancée ici, une seule fois, quel que
    soit le résultat des news (publiées, rejetées, rien de nouveau, panne LLM...)."""
    faulthandler.cancel_dump_traceback_later()
    lancer_video_une_fois()      # filet de sécurité si elle n'a pas encore été lancée
    banner(7, TOTAL_STEPS, "RÉSUMÉ")
    log(f"Flux OK : {STATS['flux_ok']} | Flux en échec : {STATS['flux_ko']} | "
        f"News : {STATS['news_total']} | Déjà publiées : {STATS['deja']} | "
        f"Trop anciennes : {STATS['anciennes']} | À examiner : {STATS['candidates']}")
    log(f"Tentatives : {STATS['tentatives']} | Rejetées : {STATS['rejetees']} | "
        f"PUBLIÉES : {STATS['publies']}",
        level="OK" if STATS["publies"] else "WARN")
    for titre, raison in STATS["echecs"]:
        log(f"   rejet/échec : {titre[:70]} -> {raison}", level="WARN" if "rejet" in raison.lower() or "hors" in raison.lower() else "ERROR")
    if _DEAD:
        log("Comptes désactivés pendant ce run : " + ", ".join(sorted(_DEAD)))
    log(f"Durée totale : {time.time() - T0:.0f}s | Résultat : {statut} (code de sortie {code})",
        level="OK" if code == 0 else "ERROR")
    ecrire_resume(statut)
    sys.exit(code)


def publier_video_reel(historique):
    """Appelle le module séparé video_reel.py (lit uniquement la Page Facebook).
    Ne lève jamais d'exception et ne change pas le code de sortie du script."""
    try:
        from video_reel import publier_video_reel as _run
    except Exception as e:
        log(f"Module video_reel.py introuvable ou invalide ({type(e).__name__}: {str(e)[:200]}) : "
            "vidéo ignorée.", level="ERROR")
        return
    ctx = SimpleNamespace(
        log=log, llm_chat=llm_chat, menv=menv, explain_graph_error=explain_graph_error,
        json_or_text=TextPost._json_or_text, step_summary=step_summary,
        graph_url=GRAPH_URL, ig_caption_limit=IG_CAPTION_LIMIT, brand_handle=BRAND_HANDLE,
        meta_suffix=META_SUFFIX, base_dir=BASE_DIR, output_dir=OUTPUT_DIR,
        poster_commentaire=poster_commentaire, disclaimer=DISCLAIMER,
    )
    try:
        _run(historique, ctx)
    except Exception as e:
        log(f"Vidéo Reel : erreur {type(e).__name__} : {str(e)[:300]}", level="ERROR")
        log(traceback.format_exc(limit=4))


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nom_hist = "historique_posts.json" if not META_SUFFIX else f"historique_posts{META_SUFFIX.lower()}.json"
    fichier_json = os.path.join(base_dir, nom_hist)

    banner(1, TOTAL_STEPS, "Configuration et environnement")
    manquants = diagnostic_environnement()
    if manquants:
        log("Variables obligatoires manquantes : " + ", ".join(manquants), level="ERROR")
        log("   -> ajoute-les dans Settings > Secrets ET dans la section 'env:' du workflow.")
        STATS["echecs"].append(("Démarrage", "secrets manquants : " + ", ".join(manquants)))
        terminer(1, "ÉCHEC (configuration)")

    banner(2, TOTAL_STEPS, "Vérification des accès (LLM, Facebook, Instagram)")
    if llm_chat([{"role": "user", "content": "Reply with the word: ok"}], max_tokens=40) is None:
        log("Aucun service LLM ne répond (tous comptes Cloudflare, Hugging Face, Groq) : arrêt.",
            level="ERROR")
        STATS["echecs"].append(("LLM", "aucun service LLM disponible"))
        terminer(1, "ÉCHEC (LLM)")
    fb_ok, ig_ok = verifier_meta()
    if not (fb_ok or ig_ok):
        log("Aucune destination utilisable (Facebook et Instagram refusés) : arrêt, inutile de "
            "générer des images.", level="ERROR")
        STATS["echecs"].append(("Meta", "token/ID refusés par l'API Graph"))
        terminer(1, "ÉCHEC (Meta)")

    # VIDÉO REEL : lancée AVANT les news, donc indépendante de leur durée, de leurs rejets
    # et d'un éventuel dépassement de délai. Elle utilise les photos déjà publiées sur la Page.
    banner(3, TOTAL_STEPS, "Vidéo Reel (images de la Page, 12 dernières heures)")
    lancer_video_une_fois()

    extract_post = Extract_post()
    text_post = TextPost()

    banner(4, TOTAL_STEPS, "Lecture des flux RSS")
    all_news = extract_post.fetch_all_news()
    STATS["news_total"] = len(all_news)
    if not all_news:
        log("AUCUNE news récupérée : les flux sont bloqués ou vides (voir lignes ci-dessus).",
            level="ERROR")
        STATS["echecs"].append(("Flux RSS", "aucune news récupérée"))
        terminer(1, "ÉCHEC (aucune news)")

    banner(5, TOTAL_STEPS, "Sélection des news à examiner")
    historique = charger_historique(fichier_json)
    # Photos web déjà utilisées : jamais deux fois la même photo.
    text_post.urls_web_utilisees = {v.get("image_source") for v in historique.values()
                                    if isinstance(v, dict) and v.get("image_source")}
    now = datetime.now(timezone.utc)
    candidates = []
    for i, news in enumerate(all_news):
        post_id = f"{news['source']}_{news['title']}_{news['date']}"
        age_h = (now - news["datetime"]).total_seconds() / 3600
        if post_id in historique:
            statut = "déjà publiée"
            STATS["deja"] += 1
        elif age_h > MAX_NEWS_AGE_HOURS:
            statut = f"trop ancienne ({age_h:.0f} h)"
            STATS["anciennes"] += 1
        else:
            statut = "À EXAMINER"
            candidates.append((news, post_id))
        if i < 15:
            log(f"[{i + 1:02d}] {news['source'][:22]:<22} {news['date']} ({age_h:5.1f} h) "
                f"{statut:<22} | {news['title'][:60]}")
    STATS["candidates"] = len(candidates)
    log(f"Bilan : {len(all_news)} news | {STATS['deja']} déjà publiées | "
        f"{STATS['anciennes']} trop anciennes | {len(candidates)} à examiner",
        level="OK" if candidates else "WARN")

    if not candidates:
        if STATS["deja"] and not STATS["anciennes"]:
            log("Rien de nouveau : toutes les news récentes sont déjà publiées.")
        elif STATS["anciennes"]:
            log(f"Toutes les news non publiées ont plus de {MAX_NEWS_AGE_HOURS} h : "
                "augmente MAX_NEWS_AGE_HOURS si tu veux les publier.", level="WARN")
        terminer(0, "RIEN À PUBLIER")      # la vidéo est lancée dans terminer()

    # Titres déjà connus : historique + news acceptées pendant ce run (anti-doublons).
    titres_connus = [v.get("titre", "") for v in historique.values()
                     if isinstance(v, dict) and v.get("titre")]

    banner(6, TOTAL_STEPS, "Filtrage, traduction, image et publication")
    for k, (news, post_id) in enumerate(candidates, 1):
        if STATS["publies"] >= MAX_POSTS_PER_RUN:
            log(f"Limite atteinte ({MAX_POSTS_PER_RUN} publication(s) par exécution).")
            break
        if STATS["tentatives"] >= MAX_ATTEMPTS_PER_RUN:
            log(f"Limite de tentatives atteinte ({MAX_ATTEMPTS_PER_RUN}).", level="WARN")
            break

        STATS["tentatives"] += 1
        title, content, date, src = news["title"], news["continue"], news["date"], news["source"]
        log(f"━━ News {k}/{len(candidates)} (tentative {STATS['tentatives']}/{MAX_ATTEMPTS_PER_RUN}) ━━",
            level="STEP")
        log(f"Source : {src} | Date : {date}")
        log(f"Titre  : {title}")
        log(f"Texte  : {len(content)} caractères")

        # --- Filtre 1 : doublon de titre (gratuit) ---
        doublon = extract_post.titre_en_doublon(title, titres_connus)
        if doublon:
            log(f"REJETÉE (doublon) : proche de « {doublon[:80]} »", level="WARN")
            STATS["rejetees"] += 1
            STATS["echecs"].append((title, f"rejet doublon : proche de « {doublon[:60]} »"))
            continue

        article_content = extract_post.get_article_content(news)

        # --- Filtre 2 : pré-filtre gratuit (appels de résultats, promo, contenu mince) ---
        pre_ok, pre_raison = extract_post.prefiltre_actualite(news, article_content)
        if not pre_ok:
            log(f"REJETÉE (pré-filtre) : {pre_raison}", level="WARN")
            STATS["rejetees"] += 1
            STATS["echecs"].append((title, f"rejet pré-filtre : {pre_raison}"))
            continue

        # --- Filtre 3 : validation LLM stricte (note >= MIN_NEWS_SCORE) ---
        editorial_ok, editorial_category, editorial_reason, editorial_score = \
            extract_post.valider_actualite(news, article_content)
        if not editorial_ok:
            STATS["rejetees"] += 1
            STATS["echecs"].append((title, f"rejet éditorial (note {editorial_score}) : {editorial_reason}"))
            continue   # aucune traduction, aucune image, aucune publication

        text_original = (f"{title}\n{content}\n\nCorps de l'article : {article_content}"
                         f"\nsource: {src}, date: {date}, catégorie: {editorial_category}")
        text_translated = extract_post.tradication_of_text(text_original)
        if not isinstance(text_translated, dict):
            log("Aucune traduction LLM exploitable : arrêt du workflow, aucune publication sans traduction.", level="ERROR")
            terminer(1, "ÉCHEC (LLM indisponible pour la traduction)")

        # --- Filtre 4 : règle « وماذا بعد؟ » (pas de suite concrète = pas de publication) ---
        if text_translated.get("skip"):
            raison = text_translated.get("reason", "rejet « وماذا بعد؟ »")
            log(f"REJETÉE : {raison}", level="WARN")
            STATS["rejetees"] += 1
            STATS["echecs"].append((title, f"rejet {raison}"))
            continue

        caption = str(text_translated.get("caption") or "").strip()
        if not caption:
            log(f"Traduction sans 'caption' : {str(text_translated)[:200]}", level="ERROR")
            STATS["echecs"].append((title, "traduction vide"))
            continue
        headline = str(text_translated.get("headline") or "").strip()
        if not headline:
            headline = " ".join(caption.split()[:14])
            log("Pas de titre court renvoyé : début de la légende utilisé.", level="WARN")

        # Date + source (« المصدر: ... ») + hashtags dans la légende ; l'avertissement est posté
        # en 1er commentaire (pas dans la légende ni dans l'image). On laisse de la place au crédit photo.
        caption = extract_post.ajouter_source(
            caption, src, extract_post.hashtags_pour(editorial_category), date, reserve=170)

        log(f"Titre arabe : {headline}")
        log(f"Légende     : {len(caption)} caractères (date + source {src} + hashtags inclus)")

        resultat = text_post.publier_texte_partout(text_original, caption, headline,
                                                   editorial_category, title)

        if resultat.get("facebook") or resultat.get("instagram"):
            historique[post_id] = {"source": src, "titre": title, "date": date,
                                   "note": editorial_score, "categorie": editorial_category,
                                   "image_url": text_post.last_image_url,
                                   "image_origine": text_post.last_image_origine,
                                   "image_source": text_post.last_image_source,
                                   "publie_le": datetime.now(timezone.utc).isoformat()}
            with open(fichier_json, "w", encoding="utf-8") as f:
                json.dump(historique, f, ensure_ascii=False, indent=4)
            titres_connus.append(title)
            if text_post.last_image_source:
                text_post.urls_web_utilisees.add(text_post.last_image_source)
            STATS["publies"] += 1
            log(f"PUBLIÉ (Facebook : {'oui' if resultat['facebook'] else 'non'} | "
                f"Instagram : {'oui' if resultat['instagram'] else 'non'} | "
                f"image : {text_post.last_image_origine}) -> historique mis à jour",
                level="OK")
        else:
            raison = resultat.get("erreur") or "Facebook et Instagram n'ont pas confirmé (voir ❌ ci-dessus)"
            STATS["echecs"].append((title, raison))
            log(f"NON PUBLIÉ : {raison}", level="ERROR")

    # (La vidéo Reel a déjà été lancée avant les news ; terminer() sert de filet de sécurité.)
    if STATS["publies"] > 0:
        terminer(0, "SUCCÈS")
    if STATS["rejetees"] > 0 and STATS["rejetees"] >= STATS["tentatives"]:
        # Toutes les news examinées ont été rejetées par le filtre : ce n'est PAS une panne.
        terminer(0, "RIEN DE PUBLIABLE (toutes les news ont été rejetées par le filtre éditorial)")
    terminer(1, "ÉCHEC (rien n'a été publié)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        log(f"ERREUR INATTENDUE : {type(exc).__name__} : {exc}", level="ERROR")
        print(traceback.format_exc(), flush=True)
        STATS["echecs"].append(("Erreur inattendue", f"{type(exc).__name__}: {exc}"))
        ecrire_resume("CRASH")
        sys.exit(1)
