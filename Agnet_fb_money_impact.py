import sys
import faulthandler
SCRIPT_VERSION = "2026-10-24-cloudflare-logo"
print(f"▶ Python démarré (version {SCRIPT_VERSION}), chargement des modules…", flush=True)
faulthandler.enable()
# Si le script reste figé, affiche la pile d'appel toutes les 90 s (montre OÙ il bloque)
faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)

import os
import io
import base64
import traceback
import json
import re
import time
import urllib.parse
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
from groq import Groq
from langchain_groq import ChatGroq
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

# Modifiables par variables d'environnement (sans toucher au code)
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
# Générateurs d'image, essayés dans cet ordre : Cloudflare -> Hugging Face -> Pollinations
CF_IMAGE_MODEL = os.getenv("CF_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
GRAPH_URL = "https://graph.facebook.com/" + os.getenv("GRAPH_VERSION", "v26.0")

# Génération directement en portrait -> aucun recadrage destructif
IMG_W, IMG_H = 1024, 1280      # multiples de 16 (requis par FLUX)
OUT_W, OUT_H = 1080, 1350      # sortie finale 4:5 (Facebook / Instagram)

# Nombre de candidats générés ; le meilleur est sélectionné automatiquement
IMAGE_CANDIDATES = int(os.getenv("IMAGE_CANDIDATES", "2"))

# Contrôle de publication
MAX_POSTS_PER_RUN = int(os.getenv("MAX_POSTS_PER_RUN", "1"))
MAX_ATTEMPTS_PER_RUN = int(os.getenv("MAX_ATTEMPTS_PER_RUN", "3"))
MAX_NEWS_AGE_HOURS = int(os.getenv("MAX_NEWS_AGE_HOURS", "24"))
IG_CAPTION_LIMIT = 2200

# Identité de la page
BRAND_NAME = ("Money", ".impact")
BRAND_HANDLE = "@money.impact.today"
BRAND_ACCENT = (255, 196, 0)
# Police arabe grasse. Idéal : Cairo-Bold.ttf ou Tajawal-Bold.ttf dans le dossier ./fonts/
AR_FONT_ENV = os.getenv("AR_FONT_PATH", "")

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logo de la page (PNG transparent conseillé), affiché en haut à gauche
def _find_logo():
    env = os.getenv("LOGO_PATH", "")
    if env:
        return Path(env)
    for name in ("logo.png", "logo.jpg", "logo.jpeg"):
        p = BASE_DIR / "imgs" / name
        if p.is_file():
            return p
    return BASE_DIR / "imgs" / "logo.png"


LOGO_PATH = _find_logo()
# Mettre LOGO_BACKDROP=1 si ton logo est sombre (ajoute un fond foncé derrière)
LOGO_BACKDROP = os.getenv("LOGO_BACKDROP", "0") == "1"

# Scènes de secours si le LLM échoue (mots-clés -> scène photo précise)
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
     "British pound notes on dark oak-free slate surface, blurred London financial district behind"),
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
# LOGS : affichage en temps réel, étape par étape (visible dans GitHub Actions)
# ======================================================================
try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except Exception:
    pass

IS_GHA = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
T0 = time.time()
SECRET_ENV_NAMES = ("TOKEN_AI_DESCRYB_IMG", "HF_TOKEN", "PAGE_ACCESS_TOKEN",
                    "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET",
                    "CF_API_TOKEN", "CF_ACCOUNT_ID")
_ICONS = {"INFO": "  ", "STEP": "➡️ ", "OK": "✅", "WARN": "⚠️ ", "ERROR": "❌"}

# Compteurs pour le résumé final
STATS = {
    "flux_ok": 0, "flux_ko": 0, "news_total": 0, "deja": 0, "anciennes": 0,
    "candidates": 0, "tentatives": 0, "publies": 0, "echecs": [],
}


def _redact(text):
    """Masque les secrets : un token ne doit JAMAIS apparaître dans les logs."""
    s = str(text)
    for name in SECRET_ENV_NAMES:
        v = os.getenv(name, "")
        if v and len(v) >= 6:
            s = s.replace(v, "***")
    return re.sub(r"(access_token=)[^&\s\"']+", r"\1***", s)


def log(*args, level="INFO"):
    """Une ligne de log horodatée, affichée immédiatement (flush)."""
    msg = _redact(" ".join(str(a) for a in args))
    print(f"[{time.time() - T0:6.1f}s] {_ICONS.get(level, '  ')} {msg}", flush=True)
    if IS_GHA and level == "ERROR":  # apparaît aussi dans les annotations du run
        print("::error::" + msg.replace("\r", " ").replace("\n", " ")[:300], flush=True)


def banner(num, total, title):
    print("", flush=True)
    print("═" * 68, flush=True)
    print(f" ÉTAPE {num}/{total} — {title}", flush=True)
    print("═" * 68, flush=True)


def step_summary(markdown):
    """Ajoute du texte à la page 'Summary' du run GitHub Actions."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
    except Exception:
        pass


def explain_graph_error(data):
    """Transforme une erreur de l'API Meta en message clair + piste de solution."""
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
# GÉNÉRATION DE POST (image + publication)
# ======================================================================
class TextPost:
    # Services d'image désactivés pour le reste de l'exécution
    # (crédits épuisés, token refusé...) : évite de perdre du temps à chaque image.
    _dead = set()

    def __init__(self):
        self.llm = ChatGroq(
            model_name=GROQ_MODEL,
            temperature=0.4,
            groq_api_key=os.getenv("TOKEN_AI_DESCRYB_IMG", ""),
            timeout=60,
            max_retries=1,
        )

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

        try:
            return self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as e:
            log(f"LLM prompt image : erreur {type(e).__name__} : {str(e)[:300]}", level="ERROR")

            class Fallback:
                content = ""

            return Fallback()

    @staticmethod
    def _parse_prompts(text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
        pos = re.search(r"POSITIVE PROMPT:\s*(.*?)\s*NEGATIVE PROMPT:", text, re.S | re.I)
        neg = re.search(r"NEGATIVE PROMPT:\s*(.*)", text, re.S | re.I)
        if not pos or not neg:
            return None, None
        positive = pos.group(1).strip().strip('"').strip("[]")
        return positive, neg.group(1).strip()

    @staticmethod
    def _fallback_scene(news):
        low = (news or "").lower()
        for keywords, scene in FALLBACK_SCENES:
            if any(k in low for k in keywords):
                return scene
        return DEFAULT_SCENE

    def _build_prompt(self, news):
        t = time.time()
        text = self.donne_moi_prompt_images(news).content
        log(f"LLM : réponse en {time.time() - t:.1f}s ({len(text or '')} caractères)")
        positive, _ = self._parse_prompts(text)
        if not positive or len(positive.split()) < 6:
            positive = self._fallback_scene(news)
            log("LLM : prompt inexploitable -> scène de secours (mots-clés) utilisée.", level="WARN")
        else:
            log("LLM : prompt photo reçu.", level="OK")
        # FLUX ignore le negative prompt : les interdits sont dans le positif
        positive = " ".join(positive.split()[:70])
        return f"{positive}, {STYLE_SUFFIX}"

    # ------------------------------------------------------------------
    # 2) Génération d'image : Cloudflare -> Hugging Face -> Pollinations
    # ------------------------------------------------------------------
    def _image_cloudflare(self, prompt, seed=None):
        """Cloudflare Workers AI (FLUX.1 schnell) : 10 000 neurons gratuits par jour."""
        account = os.getenv("CF_ACCOUNT_ID", "")
        token = os.getenv("CF_API_TOKEN", "")
        if not (account and token):
            if "cloudflare_absent" not in self._dead:
                log("CF_ACCOUNT_ID / CF_API_TOKEN absents -> Cloudflare ignoré.", level="WARN")
                self._dead.add("cloudflare_absent")
            return None
        if "cloudflare" in self._dead:
            return None

        url = (f"https://api.cloudflare.com/client/v4/accounts/{account}"
               f"/ai/run/{CF_IMAGE_MODEL}")
        headers = {"Authorization": f"Bearer {token}"}
        body = {"prompt": prompt[:2000], "steps": 6}   # schnell : 8 steps maximum
        if seed is not None:
            body["seed"] = int(seed) % 2_000_000_000

        t = time.time()
        try:
            log(f"Cloudflare : {CF_IMAGE_MODEL} (steps={body['steps']})…")
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 400 and "seed" in body:   # seed refusé : on réessaie sans
                body.pop("seed")
                r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code in (401, 403):
                log(f"Cloudflare : accès refusé (HTTP {r.status_code}) -> vérifie CF_API_TOKEN "
                    "(permissions Workers AI) et CF_ACCOUNT_ID. Service ignoré.", level="ERROR")
                self._dead.add("cloudflare")
                return None
            if r.status_code == 429:
                log("Cloudflare : limite atteinte (quota du jour ?) -> service ignoré.",
                    level="WARN")
                self._dead.add("cloudflare")
                return None
            r.raise_for_status()
            data = r.json()
            b64 = (data.get("result") or {}).get("image")
            if not b64:
                log(f"Cloudflare : réponse sans image : {str(data)[:300]}", level="WARN")
                return None
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            log(f"Cloudflare OK en {time.time() - t:.1f}s -> image {img.size[0]}x{img.size[1]}",
                level="OK")
            return img
        except Exception as e:
            log(f"Cloudflare indisponible ({type(e).__name__}) : {str(e)[:300]}", level="WARN")
            return None

    def _image_hf(self, prompt, seed=None):
        token = os.getenv("HF_TOKEN", "")
        if not token:
            log("HF_TOKEN absent -> Hugging Face ignoré.", level="WARN")
            return None
        if "hf" in self._dead:
            return None

        model = HF_IMAGE_MODEL
        is_schnell = "schnell" in model.lower()
        # schnell : distillé pour 1-4 steps, ignore la guidance
        # dev : 28-35 steps, guidance ~3.5
        params = {"width": IMG_W, "height": IMG_H,
                  "num_inference_steps": 4 if is_schnell else 30}
        if not is_schnell:
            params["guidance_scale"] = 3.5
        if seed is not None:
            params["seed"] = seed

        client = InferenceClient(api_key=token)

        # Dégradation progressive des paramètres si l'API en refuse certains
        attempts = [
            params,
            {k: v for k, v in params.items() if k != "seed"},
            {"width": IMG_W, "height": IMG_H},
            {},
        ]
        for i, p in enumerate(attempts):
            t = time.time()
            try:
                log(f"HF essai {i + 1}/{len(attempts)} : {model} | {p}")
                image = client.text_to_image(prompt, model=model, **p)
                log(f"HF OK en {time.time() - t:.1f}s -> image {image.size[0]}x{image.size[1]}",
                    level="OK")
                return image.convert("RGB")
            except Exception as e:
                log(f"HF essai {i + 1} échoué en {time.time() - t:.1f}s "
                    f"({type(e).__name__}) : {str(e)[:300]}", level="WARN")
                if "402" in str(e) or "Payment Required" in str(e):
                    log("HF : crédits épuisés -> service ignoré pour cette exécution.",
                        level="WARN")
                    self._dead.add("hf")
                    return None
                time.sleep(2)
        log("HF : tous les essais ont échoué.", level="WARN")
        return None

    def _image_pollinations(self, prompt, seed=None):
        if "pollinations" in self._dead:
            return None
        seed = seed if seed is not None else int(time.time())
        url = (
            "https://image.pollinations.ai/prompt/"
            + urllib.parse.quote(prompt)
            + f"?model=flux&width={IMG_W}&height={IMG_H}&nologo=true&seed={seed}&enhance=false"
        )
        t = time.time()
        try:
            log("Pollinations : requête en cours (peut prendre jusqu'à 3 min)…")
            r = requests.get(url, timeout=180)
            log(f"Pollinations : HTTP {r.status_code} | {len(r.content) // 1024} Ko "
                f"| {time.time() - t:.1f}s")
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            # Retire un éventuel bandeau/filigrane en bas
            w, h = img.size
            log(f"Pollinations OK -> image {w}x{h}", level="OK")
            return img.crop((0, 0, w, int(h * 0.94)))
        except Exception as e:
            log(f"Pollinations indisponible ({type(e).__name__}) : {str(e)[:200]} "
                "-> service ignoré pour cette exécution.", level="WARN")
            self._dead.add("pollinations")
            return None

    def _generate_one(self, prompt, seed):
        services = (
            ("Cloudflare", self._image_cloudflare),
            ("Hugging Face", self._image_hf),
            ("Pollinations", self._image_pollinations),
        )
        for nom, fonction in services:
            img = fonction(prompt, seed)
            if img is not None:
                return img, nom
            log(f"{nom} : pas d'image, service suivant…", level="WARN")
        return None, None

    @staticmethod
    def _image_fallback_design():
        """Dernier recours : fond graphique bleu nuit et or (le titre arabe est ajouté dessus)."""
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
        """Score = netteté + contraste, pénalise images trop sombres/claires."""
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
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c[0])
        log(f"Meilleur candidat retenu (score {best[0]:.1f}).")
        return best[1]

    # ------------------------------------------------------------------
    # 4) Post-traitement (netteté, couleur, vignette, grain)
    # ------------------------------------------------------------------
    @staticmethod
    def _enhance_image(img):
        img = img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=60, threshold=3))
        img = ImageEnhance.Contrast(img).enhance(1.07)
        img = ImageEnhance.Color(img).enhance(1.05)
        img = ImageEnhance.Sharpness(img).enhance(1.15)

        w, h = img.size
        # Vignette douce
        mask = Image.radial_gradient("L").resize((w, h), Image.LANCZOS)
        mask = mask.point(lambda p: int(max(0, p - 90) * 0.55))
        img = Image.composite(Image.new("RGB", (w, h), (3, 8, 22)), img, mask)

        # Grain très léger : casse l'aspect "lisse" typique des images IA
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
    # Titre arabe + branding style "chaîne d'info" (comme Al Arabiya)
    # ------------------------------------------------------------------
    # L'affichage de l'arabe dépend de la version de Pillow (Raqm ou non). Au premier
    # appel, le script teste le rendu réel et choisit un mode qui fonctionne :
    #   "raqm"   : texte brut, Raqm/HarfBuzz fait le shaping + le sens RTL
    #   "bidi"   : arabic_reshaper + python-bidi, moteur BASIC
    #   "manual" : arabic_reshaper + inversion des lettres, moteur BASIC
    # Dans tous les cas, l'ordre des mots est géré ici (un mot = un dessin), donc
    # il ne peut plus être inversé par le moteur de texte.
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
        """Un mot arabe -> chaîne à dessiner (gauche vers droite)."""
        if mode == "raqm" or not ARABIC_OK:
            return word
        shaped = arabic_reshaper.reshape(word)
        if mode == "bidi":
            return get_display(shaped, base_dir="R")
        # "manual" : lettres inversées, les chiffres gardent leur ordre
        parts = cls._NUM_RE.split(shaped)
        pieces = [p if i % 2 else p[::-1] for i, p in enumerate(parts)]
        return "".join(reversed(pieces))

    @classmethod
    def _ar_selftest(cls, mode):
        """Dessine 'اب' : l'alef (grande barre) doit être à DROITE du bêh."""
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
        """Découpe en blocs : mots arabes seuls, mots latins/nombres regroupés (ordre LTR)."""
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
        """Dessine une ligne arabe alignée à droite : 1er mot à droite, suivants vers la gauche."""
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
        """Réduit la taille de police jusqu'à ce que le titre tienne (max 4 lignes)."""
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
        """Charge imgs/logo.png (PNG transparent conseillé). None si absent."""
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

        # ---- 1) Dégradés (bas fort pour le titre, haut léger pour le logo) ----
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

        # ---- 2) Géométrie ----
        lx, ly = int(w * 0.04), int(h * 0.03)
        part_a, part_b = BRAND_NAME

        if logo is not None:
            logo_box = [lx, ly, lx + logo.width, ly + logo.height]
        else:  # secours : logo en texte
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

            # Étiquette = nom de la page (Money.impact)
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

        # ---- 3) Ombres floutées ----
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

        # ---- 4) Fond translucide net (logo texte ou fond du logo image) ----
        crisp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        cd = ImageDraw.Draw(crisp)
        if logo is None:
            cd.rectangle(logo_box, fill=(4, 10, 28, 190), outline=(255, 255, 255, 220), width=2)
        elif LOGO_BACKDROP:
            p = int(w * 0.012)
            cd.rectangle([logo_box[0] - p, logo_box[1] - p, logo_box[2] + p, logo_box[3] + p],
                         fill=(4, 10, 28, 190), outline=(255, 255, 255, 220), width=2)
        img = Image.alpha_composite(img, crisp)

        # ---- 5) Éléments opaques ----
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
        log("Image 1/5 : demande du prompt photo au LLM (Groq)…")
        prompt = self._build_prompt(news_original)
        log(f"Image 2/5 : prompt final ({len(prompt.split())} mots) :")
        log("      " + prompt)
        log(f"Image 3/5 : génération de {max(1, IMAGE_CANDIDATES)} candidat(s)…")
        img = self._best_image(prompt)
        if img is None:
            log("Aucun générateur d'image disponible (Cloudflare, Hugging Face et "
                "Pollinations ont échoué) -> fond graphique de secours.", level="WARN")
            img = self._image_fallback_design()

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

            page_id = os.getenv("PAGE_ID", "")
            token = os.getenv("PAGE_ACCESS_TOKEN", "")
            ig_id = os.getenv("INSTAGRAM_BUSINESS_ID", "")
            cloud = (os.getenv("CLOUDINARY_CLOUD_NAME", ""), os.getenv("CLOUDINARY_API_KEY", ""),
                     os.getenv("CLOUDINARY_API_SECRET", ""))
            if not token:
                raise ValueError("PAGE_ACCESS_TOKEN est absent (secret non transmis au job ?)")

            log("── Publication Facebook ──", level="STEP")
            if page_id:
                resultats["facebook"] = self._publier_facebook(chemin, news_translited,
                                                               page_id, token)
            else:
                log("Facebook IGNORÉ : PAGE_ID est vide.", level="WARN")

            log("── Publication Instagram ──", level="STEP")
            manque = [n for n, v in (("INSTAGRAM_BUSINESS_ID", ig_id),
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
        self.model_groq_name = GROQ_MODEL
        self.client = Groq(api_key=os.getenv("TOKEN_AI_DESCRYB_IMG", ""), timeout=60,
                           max_retries=1)
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
            # Beaucoup de sites bloquent le User-Agent par défaut de feedparser/GitHub
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

    @staticmethod
    def _extract_json(text):
        """Extrait l'objet JSON d'une réponse LLM (gère <think>…</think> et ```json)."""
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                return json.loads(m.group(0))
            raise

    def tradication_of_text(self, text):
        prompt = """
        You are a professional Arabic translator specializing in financial news.
        Translate into natural Modern Standard Arabic. Preserve financial meaning and facts.
        Return ONLY a valid JSON object with two fields:
        {"caption": "full translation of the text",
         "headline": "short punchy Arabic news headline"}
        The headline: 8 to 15 words, newsroom style, captures the key fact (and the
        market impact if any), no emojis, no hashtags, no quotation marks, no source name.
        """
        t = time.time()
        log(f"Traduction en arabe (Groq, modèle {self.model_groq_name})…")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"SOURCE TEXT TO TRANSLATE:\n{text}"},
        ]
        last_error = None
        for mode_json in (True, False):   # 1) mode JSON strict, 2) texte libre puis extraction
            try:
                kwargs = {"response_format": {"type": "json_object"}} if mode_json else {}
                response = self.client.chat.completions.create(
                    model=self.model_groq_name,
                    messages=messages,
                    temperature=0.0,
                    **kwargs,
                )
                data = self._extract_json(response.choices[0].message.content)
                if isinstance(data, dict) and str(data.get("caption") or "").strip():
                    log(f"Traduction reçue en {time.time() - t:.1f}s", level="OK")
                    return data
                last_error = f"réponse sans 'caption' : {str(data)[:150]}"
                log(f"Traduction inexploitable ({last_error})", level="WARN")
            except Exception as e:
                last_error = f"{type(e).__name__} : {str(e)[:300]}"
                log(f"Traduction essai {'JSON' if mode_json else 'texte'} échoué ({last_error})",
                    level="WARN")
        log(f"Traduction ÉCHOUÉE : {last_error}", level="ERROR")
        if last_error and "model" in last_error.lower() and (
                "not" in last_error.lower() or "exist" in last_error.lower()):
            log("   -> le modèle Groq n'existe pas / n'est plus disponible : "
                "change la variable GROQ_MODEL.", level="WARN")
        return None


# ======================================================================
# MAIN
# ======================================================================
GROQ_FALLBACKS = ("llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b",
                  "llama-3.1-8b-instant")
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
    """Affiche la config + la présence des secrets (jamais leur valeur). Renvoie les manquants."""
    log(f"Python {sys.version.split()[0]} | Pillow {PIL.__version__} | "
        f"GitHub Actions : {'oui' if IS_GHA else 'non'}")
    log(f"Script : {Path(__file__).name} | version {SCRIPT_VERSION}")
    log(f"Heure UTC : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}")
    log(f"Réglages : Groq={GROQ_MODEL} | image CF={CF_IMAGE_MODEL} | image HF={HF_IMAGE_MODEL} | "
        f"candidats={IMAGE_CANDIDATES} | max posts={MAX_POSTS_PER_RUN} | "
        f"max tentatives={MAX_ATTEMPTS_PER_RUN} | âge max={MAX_NEWS_AGE_HOURS}h | "
        f"Graph={GRAPH_URL.rsplit('/', 1)[-1]}")

    variables = [
        ("TOKEN_AI_DESCRYB_IMG", True, "Groq (traduction + prompt image)"),
        ("PAGE_ACCESS_TOKEN", True, "token Meta (Facebook + Instagram)"),
        ("CF_ACCOUNT_ID", False, "Cloudflare Workers AI (images FLUX gratuites)"),
        ("CF_API_TOKEN", False, "Cloudflare Workers AI (images FLUX gratuites)"),
        ("HF_TOKEN", False, "Hugging Face (images FLUX) -> secours"),
        ("PAGE_ID", False, "ID de la page Facebook -> sinon Facebook ignoré"),
        ("INSTAGRAM_BUSINESS_ID", False, "ID Instagram -> sinon Instagram ignoré"),
        ("CLOUDINARY_CLOUD_NAME", False, "Cloudinary (requis pour Instagram)"),
        ("CLOUDINARY_API_KEY", False, "Cloudinary (requis pour Instagram)"),
        ("CLOUDINARY_API_SECRET", False, "Cloudinary (requis pour Instagram)"),
    ]
    manquants = []
    for nom, obligatoire, role in variables:
        if os.getenv(nom, ""):
            log(f"{nom:<24} présent        | {role}", level="OK")
        else:
            log(f"{nom:<24} ABSENT/VIDE     | {role}", level="ERROR" if obligatoire else "WARN")
            if obligatoire:
                manquants.append(nom)
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


def resoudre_modele_groq():
    """Vérifie que le modèle Groq existe ; sinon bascule sur un modèle disponible."""
    global GROQ_MODEL
    try:
        client = Groq(api_key=os.getenv("TOKEN_AI_DESCRYB_IMG", ""), timeout=30, max_retries=1)
        ids = sorted(m.id for m in client.models.list().data)
    except Exception as e:
        log(f"Groq inaccessible ({type(e).__name__}) : {str(e)[:300]}", level="ERROR")
        log("   -> clé TOKEN_AI_DESCRYB_IMG invalide, ou réseau bloqué.", level="WARN")
        return False
    log(f"Groq : clé valide, {len(ids)} modèles disponibles", level="OK")
    if GROQ_MODEL in ids:
        log(f"Groq : modèle '{GROQ_MODEL}' disponible", level="OK")
        return True
    log(f"Le modèle Groq '{GROQ_MODEL}' N'EXISTE PAS.", level="ERROR")
    log("   Modèles disponibles : " + ", ".join(ids))
    for cand in GROQ_FALLBACKS:
        if cand in ids:
            GROQ_MODEL = cand
            log(f"   -> bascule automatique sur '{cand}' (fixe GROQ_MODEL pour le garder)",
                level="WARN")
            return True
    return False


def verifier_meta():
    """Teste les tokens Meta AVANT de générer des images. Renvoie (facebook_ok, instagram_ok)."""
    token = os.getenv("PAGE_ACCESS_TOKEN", "")
    page_id, ig_id = os.getenv("PAGE_ID", ""), os.getenv("INSTAGRAM_BUSINESS_ID", "")
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
    log(f"Durée totale : {time.time() - T0:.0f}s | Résultat : {statut} (code de sortie {code})",
        level="OK" if code == 0 else "ERROR")
    ecrire_resume(statut)
    sys.exit(code)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fichier_json = os.path.join(base_dir, "historique_posts.json")

    # ---------------- ÉTAPE 1 : environnement ----------------
    banner(1, TOTAL_STEPS, "Configuration et environnement")
    manquants = diagnostic_environnement()
    if manquants:
        log("Variables obligatoires manquantes : " + ", ".join(manquants), level="ERROR")
        log("   -> ajoute-les dans Settings > Secrets ET dans la section 'env:' du workflow.")
        STATS["echecs"].append(("Démarrage", "secrets manquants : " + ", ".join(manquants)))
        terminer(1, "ÉCHEC (configuration)")

    # ---------------- ÉTAPE 2 : accès aux services ----------------
    banner(2, TOTAL_STEPS, "Vérification des accès (Groq, Facebook, Instagram)")
    if not resoudre_modele_groq():
        STATS["echecs"].append(("Groq", "clé invalide ou aucun modèle utilisable"))
        terminer(1, "ÉCHEC (Groq)")
    fb_ok, ig_ok = verifier_meta()
    if not (fb_ok or ig_ok):
        log("Aucune destination utilisable (Facebook et Instagram refusés) : arrêt, inutile de "
            "générer des images.", level="ERROR")
        STATS["echecs"].append(("Meta", "token/ID refusés par l'API Graph"))
        terminer(1, "ÉCHEC (Meta)")

    extract_post = Extract_post()
    text_post = TextPost()

    # ---------------- ÉTAPE 3 : flux RSS ----------------
    banner(3, TOTAL_STEPS, "Lecture des flux RSS")
    all_news = extract_post.fetch_all_news()
    STATS["news_total"] = len(all_news)
    if not all_news:
        log("AUCUNE news récupérée : les 3 flux sont bloqués ou vides (voir lignes ci-dessus).",
            level="ERROR")
        STATS["echecs"].append(("Flux RSS", "aucune news récupérée"))
        terminer(1, "ÉCHEC (aucune news)")

    # ---------------- ÉTAPE 4 : sélection ----------------
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

    # ---------------- ÉTAPE 5 : publication ----------------
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

        text_original = f"{title}\n{content}\n\nsource: {src}, date: {date}"
        text_translated = extract_post.tradication_of_text(text_original)

        if not isinstance(text_translated, dict):
            STATS["echecs"].append((title, "traduction échouée"))
            continue
        caption = str(text_translated.get("caption") or "").strip()
        if not caption:
            log(f"Traduction sans 'caption' : {str(text_translated)[:200]}", level="ERROR")
            STATS["echecs"].append((title, "traduction vide"))
            continue
        headline = str(text_translated.get("headline") or "").strip()
        if not headline:  # secours : début de la légende
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

    # ---------------- ÉTAPE 6 : résumé ----------------
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
