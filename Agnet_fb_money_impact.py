import sys
import faulthandler
SCRIPT_VERSION = "2026-10-26-money-impact-strict-llm-fix2"
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
from pathlib import Path
from datetime import datetime, timezone

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
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "InvestingLive Macro": "https://investinglive.com/rss/news",
    "ActionForex Fundamental": "https://www.actionforex.com/category/fundamental-analysis/feed/",
}

# Une actualité doit être directement liée à la finance de marché ou avoir un impact
# financier concret. La simple mention d'argent ne suffit pas.
ALLOWED_NEWS_CATEGORIES = (
    "Forex", "Gold", "Indices", "Crypto", "Central banks",
    "Economic news", "Stocks", "Oil", "Bonds", "Market impact",
)

# ---------------------------------------------------------------------
# COMPTES MULTIPLES : même nom de secret + suffixe 1, 2, 3, 4, 5
#   Cloudflare : CF_ACCOUNT_ID / CF_API_TOKEN, puis CF_ACCOUNT_ID1 / CF_API_TOKEN1, ...
#   Hugging Face : HF_TOKEN, HF_TOKEN1, HF_TOKEN2, HF_TOKEN3
#   Groq : TOKEN_AI_DESCRYB_IMG, TOKEN_AI_DESCRYB_IMG1, ...
# Si un compte est épuisé / refusé, le script passe automatiquement au suivant.
# ---------------------------------------------------------------------
KEY_SUFFIXES = ("", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")


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
# Cloudflare LLM
CF_TEXT_MODELS = _csv_env("CF_TEXT_MODELS", (
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/meta/llama-3.1-8b-instruct",
))
# Hugging Face LLM : modèles équivalents, essayés dans l'ordre
HF_TEXT_MODELS = _csv_env("HF_TEXT_MODELS", (
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen/Qwen3-32B",
))
# Groq : modèles actuels de la plateforme (si tous sont refusés, le script
# interroge /models et utilise ceux que ta clé peut réellement employer)
GROQ_MODELS = _csv_env("GROQ_MODELS", (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
))
# Images
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

FALLBACK_SCENES = [
    (("gold", "xau", "bullion"),
     "Stack of gold bars on dark polished surface inside a bank vault, blurred safety deposit boxes behind"),
    (("oil", "crude", "wti", "brent", "opec"),
     "Oil refinery towers at dusk with glowing flare stacks, blurred pipelines and storage tanks in background"),
    (("bitcoin", "crypto", "btc", "ethereum"),
     "Single physical gold bitcoin coin standing on dark reflective glass, blurred server racks behind"),
    (("dollar", "usd", "fed ", "fomc", "powell", "treasury", "yield"),
     "Stack of US hundred-dollar bills on dark marble desk, blurred columns of a federal building behind"),
    (("euro", "eur", "ecb", "lagarde"),
     "Euro banknotes fanned out on dark stone surface, blurred European central bank facade behind"),
    (("yen", "jpy", "boj"),
     "Japanese yen banknotes and coins on dark lacquered surface, blurred Tokyo skyline at dusk behind"),
    (("pound", "gbp", "boe", "sterling"),
     "British pound notes on dark slate surface, blurred London financial district behind"),
    (("stock", "nasdaq", "s&p", "dow", "equities", "wall street"),
     "New York Stock Exchange facade with large columns and flags, dramatic evening light, blurred street below"),
]
DEFAULT_SCENE = (
    "Stack of gold bars on dark marble surface, blurred bank vault door in background"
)

STYLE_SUFFIX = (
    "warm cinematic side lighting, shot on Canon EOS R5, 85mm f/1.4 lens, "
    "shallow depth of field, creamy bokeh, tack-sharp focus on the hero subject, "
    "editorial photography, professional color grading in deep navy blue and warm gold, "
    "photorealistic, ultra detailed, 8k, hero subject in the upper 60% of the frame, lower 40% of the frame dark, smooth and uncluttered, "
    "no text, no letters, no logos, no people"
)


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
       "PAGE_ACCESS_TOKEN", "PAGE_ACCESS_TOKEN_MONEY_IMPAC"]
)
_ICONS = {"INFO": "  ", "STEP": "➡️ ", "OK": "✅", "WARN": "⚠️ ", "ERROR": "❌"}

STATS = {
    "flux_ok": 0, "flux_ko": 0, "news_total": 0, "deja": 0, "anciennes": 0,
    "candidates": 0, "tentatives": 0, "publies": 0, "echecs": [],
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
             "instagram_content_publish...).",
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


# ======================================================================
# SERVICES DÉSACTIVÉS (partagé LLM + images) : "cf:1", "hf:2", "groq:0"...
# Un compte épuisé pour le LLM l'est aussi pour les images (quota commun).
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
                # CORRECTIF : Cloudflare renvoie parfois un JSON déjà parsé (dict/list).
                # str(dict) donnerait du pseudo-JSON Python invalide ('True', apostrophes).
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
                # autre erreur (modèle indisponible...) : on tente le modèle équivalent suivant
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
                    if not queue and not auto_done:   # tous refusés : on demande la liste réelle
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


def llm_chat(messages, temperature=0.2, max_tokens=1800):
    """Cloudflare (tous les comptes) -> Hugging Face (tous les tokens) -> Groq (toutes les clés)."""
    for fn in (_llm_cloudflare, _llm_hf, _llm_groq):
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
class TextPost:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 1) Prompt d'image
    # ------------------------------------------------------------------
    def donne_moi_prompt_images(self, news):
        system_prompt = (
            "You are an award-winning Reuters financial photojournalist and editorial "
            "art director. You write vivid, specific, physically realistic photo "
            "descriptions. You NEVER use vague terms like 'trading screens' or "
            "'financial charts'. You always name exact objects, materials, lighting "
            "and depth of field. You forbid abstract designs, 3D renders, neon, "
            "wireframes and futuristic shapes."
        )

        user_prompt = f"""
        FINANCIAL NEWS:
        "{(news or '')[:800]}"

        Identify the MAIN SUBJECT of this news and write ONE photorealistic editorial
        photo description that represents it visually.

        Subject mapping examples:
        - bond yields / Treasuries -> stack of Treasury bonds on a desk
        - gold -> gold bars in a vault
        - oil -> refinery at dusk or oil barrels
        - central bank -> central bank building facade
        - US dollar -> dollar banknotes
        - stock selloff / rally -> stock exchange facade
        - crypto -> physical coin on reflective surface

        Rules:
        1. The FIRST WORDS must name the exact physical hero subject.
        2. Hero subject sits in the UPPER 60% of the frame, sharp focus, slightly off-center.
        3. Background: relevant but blurred (bokeh). Specific lighting (e.g. golden
           hour side-light, tungsten vault light, cold blue dawn light).
        4. The lower 40% of the frame must be dark, smooth and uncluttered (a headline will be printed there).
        5. Palette: deep navy blue and warm gold.
        6. FORBIDDEN: people, faces, hands, text, letters, numbers, logos, watermarks,
           abstract shapes, 3D renders, neon, computer screens with charts.
        7. Max 60 words. End with: photorealistic, 85mm lens, sharp focus.

        OUTPUT ONLY this format:
        POSITIVE PROMPT:
        [description]

        NEGATIVE PROMPT:
        people, faces, hands, text, letters, numbers, logos, watermark, abstract,
        3d render, neon, cartoon, illustration, painting, blurry, low quality
        """

        class _Reponse:
            content = ""

        r = _Reponse()
        r.content = llm_chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            temperature=0.4,
        ) or ""
        return r

    @staticmethod
    def _parse_prompts(text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
        pos = re.search(r"POSITIVE PROMPT:\s*(.*?)\s*NEGATIVE PROMPT:", text, re.S | re.I)
        neg = re.search(r"NEGATIVE PROMPT:\s*(.*)", text, re.S | re.I)
        if not pos or not neg:
            return None, None
        positive = pos.group(1).strip().strip('"').strip("[]")
        return positive, neg.group(1).strip()

    def _build_prompt(self, news):
        t = time.time()
        text = self.donne_moi_prompt_images(news).content
        log(f"LLM : réponse en {time.time() - t:.1f}s ({len(text or '')} caractères)")
        if not text or not text.strip():
            log("Aucune API LLM n'a fourni le prompt d'image : arrêt obligatoire.", level="ERROR")
            terminer(1, "ÉCHEC (LLM indisponible pour le prompt image)")
        positive, _ = self._parse_prompts(text)
        if not positive or len(positive.split()) < 6:
            raise RuntimeError("Le LLM a répondu, mais son prompt d'image est invalide : aucun prompt de secours autorisé.")
        log("LLM : prompt photo reçu et validé.", level="OK")
        positive = " ".join(positive.split()[:70])
        return f"{positive}, {STYLE_SUFFIX}"

    # ------------------------------------------------------------------
    # 2) Génération d'image : Cloudflare (comptes x modèles) -> HF (tokens x modèles) -> fond
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
                    body = {"prompt": prompt[:2000], "width": 1024, "height": 1280,
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

    @staticmethod
    def _image_fallback_design():
        w, h = IMG_W, IMG_H
        img = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(img)
        top, bot = (10, 30, 70), (3, 8, 22)
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)],
                   fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
        glow = Image.new("RGB", (w, h), (0, 0, 0))
        ImageDraw.Draw(glow).ellipse([w * 0.15, h * 0.05, w * 0.85, h * 0.55],
                                     fill=(120, 85, 10))
        glow = glow.filter(ImageFilter.GaussianBlur(120))
        return ImageChops.add(img, glow)

    # ------------------------------------------------------------------
    # 3) Sélection du meilleur candidat
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

    def _best_image(self, prompt):
        candidates = []
        n = max(1, IMAGE_CANDIDATES)
        for i in range(n):
            log(f"Candidat {i + 1}/{n} : génération…")
            seed = int(time.time() * 1000) % 2_000_000_000 + i * 7919
            img, source = self._generate_one(prompt, seed)
            if img is not None and img.size[0] >= 512:
                score = self._score_image(img)
                log(f"Candidat {i + 1}/{n} : {source} | {img.size[0]}x{img.size[1]} | "
                    f"score {score:.1f}", level="OK")
                candidates.append((score, img))
            else:
                log(f"Candidat {i + 1}/{n} : aucune image exploitable.", level="WARN")
                if i == 0:   # aucun service ne répond : inutile de réessayer
                    break
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c[0])
        log(f"Meilleur candidat retenu (score {best[0]:.1f}).")
        return best[1]

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
        try:
            return cls._open_ar_font(size, mode)
        except Exception:
            log("ATTENTION : aucune police arabe trouvée (mets Cairo-Bold.ttf dans ./fonts/)")
            return TextPost._load_font(size)

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
        for tok in text.split():
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
        space = draw.textlength(" ", font=font)
        total = 0
        for run, is_ar in runs:
            s, kw = self._run_args(run, is_ar)
            total += draw.textlength(s, font=font, **kw)
        return total + space * max(0, len(runs) - 1)

    def _draw_rtl_line(self, draw, right_x, y, text, font, fill):
        space = draw.textlength(" ", font=font)
        x = right_x
        for run, is_ar in self._ar_runs(text):
            s, kw = self._run_args(run, is_ar)
            rw = draw.textlength(s, font=font, **kw)
            x -= rw
            draw.text((x, y), s, font=font, fill=fill, **kw)
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
    # 5) Pipeline image complet
    # ------------------------------------------------------------------
    def generer_image(self, news_original=None, headline=None):
        t0 = time.time()
        log("Image 1/5 : demande du prompt photo au LLM…")
        prompt = self._build_prompt(news_original)
        log(f"Image 2/5 : prompt final ({len(prompt.split())} mots) :")
        log("      " + prompt)
        log(f"Image 3/5 : génération de {max(1, IMAGE_CANDIDATES)} candidat(s)…")
        img = self._best_image(prompt)
        if img is None:
            log("Aucune image réellement générée par API : publication interdite et workflow arrêté.", level="ERROR")
            terminer(1, "ÉCHEC (aucune image générée par API)")

        log("Image 4/5 : recadrage 4:5, netteté, couleurs, vignette, grain…")
        img = ImageOps.fit(img, (OUT_W, OUT_H), Image.LANCZOS, centering=(0.5, 0.4))
        img = self._enhance_image(img)

        log("Image 5/5 : logo + titre arabe"
            + (f" : « {headline} »" if headline else " (aucun titre)"))
        img = self._add_branding(img, headline)

        img.save(OUTPUT_DIR / "img.png")
        output_jpg = OUTPUT_DIR / "img.jpg"
        img.save(output_jpg, format="JPEG", quality=95, optimize=True, subsampling=0)
        log(f"Image prête : {output_jpg.name} ({output_jpg.stat().st_size // 1024} Ko, "
            f"{img.size[0]}x{img.size[1]}) en {time.time() - t0:.0f}s", level="OK")
        return output_jpg

    # ------------------------------------------------------------------
    # 6) Publication Facebook + Instagram
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
                log(f"Facebook : PUBLIÉ (id={data.get('post_id') or data.get('id')})", level="OK")
                return True
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
                log(f"Instagram : PUBLIÉ (id={pdata.get('id')})", level="OK")
                return True
            log("Instagram : ÉCHEC publication -> " + explain_graph_error(pdata), level="ERROR")
        except Exception as e:
            log(f"Instagram : exception {type(e).__name__} : {str(e)[:300]}", level="ERROR")
        return False

    def publier_texte_partout(self, news_original, news_translited, headline=None):
        resultats = {"image": None, "facebook": False, "instagram": False, "erreur": None}

        try:
            log("── Génération de l'image ──", level="STEP")
            chemin = Path(self.generer_image(news_original=news_original, headline=headline))
            if not chemin.is_file():
                raise FileNotFoundError(f"Image générée introuvable : {chemin.resolve()}")
            resultats["image"] = str(chemin.resolve())

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
                resultats["facebook"] = self._publier_facebook(chemin, news_translited,
                                                               page_id, token)
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
                resultats["instagram"] = self._publier_instagram(chemin, news_translited,
                                                                 ig_id, token, cloud)
        except Exception as e:
            resultats["erreur"] = f"{type(e).__name__}: {e}"
            log(f"Échec de la génération/préparation : {resultats['erreur']}", level="ERROR")
            log(traceback.format_exc(limit=4))

        return resultats


# ======================================================================
# EXTRACTION DES NEWS + TRADUCTION
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

    def valider_actualite(self, news, article_content):
        """Décision LLM stricte avant toute traduction, image ou publication."""
        categories = ", ".join(ALLOWED_NEWS_CATEGORIES)
        prompt = f"""
You are the strict editorial gatekeeper for Money.impact, a trading and financial-markets news page.
Decide whether this article is directly relevant to at least one allowed category: {categories}.
Accept only concrete market, trading, asset-price, macroeconomic, central-bank, company-stock,
commodity, bond, crypto or currency news with a clear financial-market connection.
Reject personal stories, family/property anecdotes, lifestyle, celebrity, general human-interest,
consumer stories, and articles that merely mention money without meaningful market impact.
Reject if the provided article text is too thin or does not establish a concrete financial connection.
Do not infer facts that are not present. The RSS source being financial does NOT automatically qualify an article.
Return ONLY JSON with this exact shape:
{{"relevant": true, "category": "one allowed category", "reason": "short reason"}}
Use relevant=false when uncertain. No markdown.
"""
        content = (article_content or "").strip()
        user = (f"SOURCE: {news.get('source','')}\nTITLE: {news.get('title','')}\n"
                f"DATE: {news.get('date','')}\nURL: {news.get('link','')}\n"
                f"RSS EXCERPT: {news.get('continue','')}\nARTICLE TEXT: {content[:12000]}")
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
        raw = llm_chat(messages, temperature=0.0, max_tokens=350)
        if not raw:
            log("Aucune API LLM ne répond pendant la validation éditoriale : arrêt du workflow.", level="ERROR")
            terminer(1, "ÉCHEC (LLM indisponible pour valider les actualités)")
        try:
            data = self._extract_json(raw)
            relevant = data.get("relevant")
            # CORRECTIF : tolère "true"/"false" renvoyés sous forme de chaîne.
            if isinstance(relevant, str):
                low = relevant.strip().lower()
                if low in ("true", "false"):
                    relevant = (low == "true")
            category = str(data.get("category") or "").strip()
            reason = str(data.get("reason") or "aucune justification fournie").strip()
            if not isinstance(relevant, bool):
                raise ValueError("champ relevant absent ou non booléen")
            if relevant and category not in ALLOWED_NEWS_CATEGORIES:
                # tolérance sur la casse (ex. "forex" -> "Forex")
                match = next((c for c in ALLOWED_NEWS_CATEGORIES
                              if c.lower() == category.lower()), None)
                if match is None:
                    raise ValueError(f"catégorie non autorisée: {category!r}")
                category = match
            log(f"Validation éditoriale : {'ACCEPTÉE' if relevant else 'REJETÉE'} | "
                f"catégorie={category or 'aucune'} | raison={reason[:180]}",
                level="OK" if relevant else "WARN")
            return relevant, category, reason
        except Exception as e:
            log(f"Réponse LLM de validation inexploitable ({type(e).__name__}: {e}) : article rejeté par sécurité.",
                level="WARN")
            log(f"Réponse brute du LLM : {str(raw)[:300]!r}", level="WARN")
            return False, "", "validation LLM inexploitable"

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

        # Cherche chaque accolade ouvrante et essaie de décoder un objet complet.
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text[match.start():])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue

        # CORRECTIF : filet de sécurité pour un dict au format Python
        # ({'relevant': True, ...}) issu d'un str(dict).
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = ast.literal_eval(text[start:end + 1])
                if isinstance(data, dict):
                    return data
            except (ValueError, SyntaxError):
                pass

        raise ValueError("Aucun objet JSON valide trouvé dans la réponse du modèle")

    def tradication_of_text(self, text):
        prompt = """
        You are a professional Arabic translator specializing in financial news.
        Translate into natural Modern Standard Arabic. Preserve financial meaning and facts.
        Return ONLY one valid JSON object, with double quotes around every key and string.
        Do not use Markdown fences, comments, or text before/after the JSON.
        Required format:
        {"caption": "full translation of the text", "headline": "short Arabic news headline"}
        The headline should be 8 to 15 Arabic words, newsroom style, capture the key fact
        and market impact if any, with no emojis, hashtags, quotation marks, or source name.
        """
        t = time.time()
        log("Traduction en arabe (Cloudflare -> Hugging Face -> Groq, tous comptes)…")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"SOURCE TEXT TO TRANSLATE:\n{text}"},
        ]

        # Important : si un fournisseur renvoie du JSON invalide, essayer le fournisseur
        # suivant au lieu de s'arrêter après la première réponse non vide.
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
                if not caption:
                    raise ValueError("champ 'caption' absent ou vide")
                if not headline:
                    # Une légende valide reste exploitable; le titre sera généré à partir de celle-ci.
                    headline = caption[:100].strip()
                    data["headline"] = headline
                data["caption"] = caption
                log(f"Traduction reçue via {provider_name} en {time.time() - t:.1f}s", level="OK")
                return data
            except Exception as e:
                # Journaliser un extrait limité, sans afficher d'identifiants API.
                excerpt = re.sub(r"\s+", " ", str(raw))[:350]
                log(f"Traduction {provider_name} : JSON invalide ({e}); extrait={excerpt!r}", level="WARN")
                log(f"Nouvel essai avec le fournisseur suivant ({provider_name} a échoué).", level="WARN")

        log("Traduction ÉCHOUÉE : aucun fournisseur n'a renvoyé un JSON exploitable.", level="ERROR")
        return None


# ======================================================================
# MAIN
# ======================================================================
TOTAL_STEPS = 6


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
        f"Graph={GRAPH_URL.rsplit('/', 1)[-1]} | page={META_SUFFIX or 'principale'}")

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
        f"| À publier | {STATS['candidates']} |",
        f"| Tentatives | {STATS['tentatives']} |",
        f"| **Publiées** | **{STATS['publies']}** |",
    ]
    if STATS["echecs"]:
        lignes += ["", "### Échecs"]
        lignes += [f"- {t[:80]} — {raison}" for t, raison in STATS["echecs"]]
    step_summary("\n".join(lignes))


def terminer(code, statut):
    faulthandler.cancel_dump_traceback_later()
    banner(6, TOTAL_STEPS, "RÉSUMÉ")
    log(f"Flux OK : {STATS['flux_ok']} | Flux en échec : {STATS['flux_ko']} | "
        f"News : {STATS['news_total']} | Déjà publiées : {STATS['deja']} | "
        f"Trop anciennes : {STATS['anciennes']} | À publier : {STATS['candidates']}")
    log(f"Tentatives : {STATS['tentatives']} | PUBLIÉES : {STATS['publies']}",
        level="OK" if STATS["publies"] else "WARN")
    for titre, raison in STATS["echecs"]:
        log(f"   échec : {titre[:70]} -> {raison}", level="ERROR")
    if _DEAD:
        log("Comptes désactivés pendant ce run : " + ", ".join(sorted(_DEAD)))
    log(f"Durée totale : {time.time() - T0:.0f}s | Résultat : {statut} (code de sortie {code})",
        level="OK" if code == 0 else "ERROR")
    ecrire_resume(statut)
    sys.exit(code)


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

    extract_post = Extract_post()
    text_post = TextPost()

    banner(3, TOTAL_STEPS, "Lecture des flux RSS")
    all_news = extract_post.fetch_all_news()
    STATS["news_total"] = len(all_news)
    if not all_news:
        log("AUCUNE news récupérée : les 3 flux sont bloqués ou vides (voir lignes ci-dessus).",
            level="ERROR")
        STATS["echecs"].append(("Flux RSS", "aucune news récupérée"))
        terminer(1, "ÉCHEC (aucune news)")

    banner(4, TOTAL_STEPS, "Sélection des news à publier")
    historique = charger_historique(fichier_json)
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
            statut = "À PUBLIER"
            candidates.append((news, post_id))
        if i < 15:
            log(f"[{i + 1:02d}] {news['source'][:22]:<22} {news['date']} ({age_h:5.1f} h) "
                f"{statut:<22} | {news['title'][:60]}")
    STATS["candidates"] = len(candidates)
    log(f"Bilan : {len(all_news)} news | {STATS['deja']} déjà publiées | "
        f"{STATS['anciennes']} trop anciennes | {len(candidates)} à publier",
        level="OK" if candidates else "WARN")

    if not candidates:
        if STATS["deja"] and not STATS["anciennes"]:
            log("Rien de nouveau : toutes les news récentes sont déjà publiées.")
        elif STATS["anciennes"]:
            log(f"Toutes les news non publiées ont plus de {MAX_NEWS_AGE_HOURS} h : "
                "augmente MAX_NEWS_AGE_HOURS si tu veux les publier.", level="WARN")
        terminer(0, "RIEN À PUBLIER")

    banner(5, TOTAL_STEPS, "Traduction, image et publication")
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

        article_content = extract_post.get_article_content(news)
        editorial_ok, editorial_category, editorial_reason = extract_post.valider_actualite(news, article_content)
        if not editorial_ok:
            STATS["echecs"].append((title, f"actualité hors spécialité/rejetée : {editorial_reason}"))
            continue

        text_original = (f"{title}\n{content}\n\nCorps de l'article : {article_content}"
                         f"\nsource: {src}, date: {date}, catégorie: {editorial_category}")
        text_translated = extract_post.tradication_of_text(text_original)
        if not isinstance(text_translated, dict):
            log("Aucune traduction LLM exploitable : arrêt du workflow, aucune publication sans traduction.", level="ERROR")
            terminer(1, "ÉCHEC (LLM indisponible pour la traduction)")

        caption = str(text_translated.get("caption") or "").strip()
        if not caption:
            log(f"Traduction sans 'caption' : {str(text_translated)[:200]}", level="ERROR")
            STATS["echecs"].append((title, "traduction vide"))
            continue
        headline = str(text_translated.get("headline") or "").strip()
        if not headline:
            headline = " ".join(caption.split()[:14])
            log("Pas de titre court renvoyé : début de la légende utilisé.", level="WARN")
        log(f"Titre arabe : {headline}")
        log(f"Légende     : {len(caption)} caractères")

        resultat = text_post.publier_texte_partout(text_original, caption, headline)

        if resultat.get("facebook") or resultat.get("instagram"):
            historique[post_id] = {"source": src, "titre": title, "date": date}
            with open(fichier_json, "w", encoding="utf-8") as f:
                json.dump(historique, f, ensure_ascii=False, indent=4)
            STATS["publies"] += 1
            log(f"PUBLIÉ (Facebook : {'oui' if resultat['facebook'] else 'non'} | "
                f"Instagram : {'oui' if resultat['instagram'] else 'non'}) -> historique mis à jour",
                level="OK")
        else:
            raison = resultat.get("erreur") or "Facebook et Instagram n'ont pas confirmé (voir ❌ ci-dessus)"
            STATS["echecs"].append((title, raison))
            log(f"NON PUBLIÉ : {raison}", level="ERROR")

    if STATS["publies"] > 0:
        terminer(0, "SUCCÈS")
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
