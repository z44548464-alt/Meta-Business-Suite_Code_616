"""Publication d'une vidéo TikTok pour Money.impact (undetected-chromedriver + session sauvegardée).

Session (créée UNE FOIS en local avec tiktok_login_local.py) lue dans l'ordre :
  1. variable d'environnement TIKTOK_SESSION_B64  (secret GitHub, JSON encodé en base64)
  2. fichier tiktok_session.json à côté du script (test local / dépôt privé)

Options (variables d'environnement) :
  TIKTOK_ENABLED          "1" (défaut) / "0"
  TIKTOK_HEADLESS         "1" (défaut, serveur) / "0" (fenêtre visible, pour tester en local)
  TIKTOK_UPLOAD_TIMEOUT   secondes max d'attente du traitement vidéo (défaut 300)
  TIKTOK_PROXY            ex. http://user:pass@host:port (optionnel)
  TIKTOK_CHROME_VERSION   ex. 126 (optionnel : force la version du driver)

publier_video_tiktok(chemin, titres, legende, ctx) retourne True ou None. NE LÈVE JAMAIS d'exception.
Test local :  python publie_video_tiktok_2.py ma_video.mp4 "légende"
"""
import base64
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

try:
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    _IMPORT_ERR = None
except Exception as _e:                      # le script principal ne doit jamais planter
    uc = By = Keys = None
    _IMPORT_ERR = f"{type(_e).__name__}: {str(_e)[:200]}"

BASE_DIR = Path(__file__).resolve().parent
SESSION_FILE = next((p for p in (BASE_DIR / "data_tiktok" / "tiktok_session.json",
                                 BASE_DIR / "tiktok_session.json") if p.is_file()),
                    BASE_DIR / "data_tiktok" / "tiktok_session.json")
HOME_URL = "https://www.tiktok.com/"
UPLOAD_URLS = ("https://www.tiktok.com/tiktokstudio/upload?from=webapp",
               "https://www.tiktok.com/creator-center/upload?lang=en")
COOKIE_KEYS = ("name", "value", "domain", "path", "secure", "httpOnly", "expiry", "sameSite")
POPUP_TEXTS = {"got it", "not now", "maybe later", "skip", "dismiss", "close"}
CONFIRM_TEXTS = {"post now", "post anyway", "publish now"}
SUCCESS_MARKERS = ("your video has been uploaded", "video published", "manage your posts",
                   "your video is being uploaded", "uploaded to tiktok")


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------
def _log(ctx, msg, level="INFO"):
    fn = getattr(ctx, "log", None)
    if fn:
        fn(msg, level=level)
    else:
        print(f"[{level}] {msg}", flush=True)


def _pause(a=0.6, b=1.6):
    time.sleep(random.uniform(a, b))


def _charger_session(ctx):
    """Retourne le dict de session ou None."""
    raw = os.getenv("TIKTOK_SESSION_B64", "").strip()
    source = "secret TIKTOK_SESSION_B64"
    try:
        if raw:
            data = json.loads(base64.b64decode(raw).decode("utf-8"))
        elif SESSION_FILE.is_file():
            data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            source = SESSION_FILE.name
        else:
            _log(ctx, "TikTok IGNORÉ : aucune session (secret TIKTOK_SESSION_B64 absent et "
                      "tiktok_session.json introuvable). Lancez tiktok_login_local.py.",
                 level="WARN")
            return None
    except Exception as e:
        _log(ctx, f"TikTok : session illisible ({type(e).__name__} {str(e)[:120]}) : "
                  "régénérez-la avec tiktok_login_local.py.", level="ERROR")
        return None
    cookies = data.get("cookies") if isinstance(data, dict) else None
    if not cookies:
        _log(ctx, "TikTok : session sans cookies.", level="ERROR")
        return None
    noms = {c.get("name") for c in cookies}
    if not ({"sessionid", "sessionid_ss"} & noms):
        _log(ctx, "TikTok : cookie 'sessionid' absent de la session (login incomplet).",
             level="ERROR")
        return None
    _log(ctx, f"TikTok : session chargée depuis {source} ({len(cookies)} cookies, "
              f"sauvegardée le {str(data.get('saved_at', '?'))[:19]}).")
    return data


def detecter_version_chrome():
    """Numéro de version majeure du Chrome installé (ex. 150), ou None."""
    import shutil
    import subprocess
    chemins = [shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium",
                                          "chromium-browser", "chrome")]
    chemins += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    for p in chemins:
        if not p or not os.path.exists(p):
            continue
        try:
            if os.name == "nt":
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"(Get-Item '{p}').VersionInfo.ProductVersion"],
                    capture_output=True, text=True, timeout=20).stdout
            else:
                out = subprocess.run([p, "--version"], capture_output=True, text=True,
                                     timeout=20).stdout
            m = re.search(r"(\d+)\.\d+\.\d+\.\d+", out)
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return None


def _creer_driver(headless, ctx):
    if uc is None:
        _log(ctx, f"TikTok : undetected-chromedriver/selenium indisponible ({_IMPORT_ERR}). "
                  "pip install undetected-chromedriver selenium setuptools", level="ERROR")
        return None
    try:
        opts = uc.ChromeOptions()
        for a in ("--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                  "--window-size=1366,900", "--lang=en-US", "--mute-audio",
                  "--disable-notifications", "--no-first-run", "--no-default-browser-check",
                  "--disable-blink-features=AutomationControlled"):
            opts.add_argument(a)
        proxy = os.getenv("TIKTOK_PROXY", "").strip()
        if proxy:
            opts.add_argument(f"--proxy-server={proxy}")
        kw = {"options": opts, "use_subprocess": True, "headless": headless}
        v = os.getenv("TIKTOK_CHROME_VERSION", "").strip()
        major = int(v) if v.isdigit() else detecter_version_chrome()
        if major:
            kw["version_main"] = major      # ChromeDriver compatible avec le Chrome installé
            _log(ctx, f"TikTok : Chrome détecté, version {major}.")
        driver = uc.Chrome(**kw)
        driver.set_page_load_timeout(90)
        # retire "HeadlessChrome" du User-Agent (la plateforme reste celle de la machine)
        try:
            ua = driver.execute_script("return navigator.userAgent")
            driver.execute_cdp_cmd("Network.setUserAgentOverride", {
                "userAgent": ua.replace("HeadlessChrome", "Chrome"),
                "acceptLanguage": "en-US,en;q=0.9"})
        except Exception:
            pass
        _log(ctx, f"TikTok : Chrome démarré (headless={headless}).")
        return driver
    except Exception as e:
        _log(ctx, f"TikTok : impossible de démarrer Chrome ({type(e).__name__} "
                  f"{str(e)[:250]}).", level="ERROR")
        return None


def _injecter_cookies(driver, cookies, ctx):
    driver.get(HOME_URL)                      # il faut être sur le domaine pour poser les cookies
    _pause(2, 3.5)
    ok = 0
    for c in cookies:
        ck = {k: v for k, v in c.items() if k in COOKIE_KEYS}
        if ck.get("sameSite") not in ("Strict", "Lax", "None"):
            ck.pop("sameSite", None)
        if "expiry" in ck:
            try:
                ck["expiry"] = int(ck["expiry"])
            except Exception:
                ck.pop("expiry", None)
        try:
            driver.add_cookie(ck)
            ok += 1
        except Exception:
            continue
    _log(ctx, f"TikTok : {ok}/{len(cookies)} cookies injectés.")
    return ok > 0


def _debug(driver, ctx, tag):
    """Capture d'écran + HTML pour comprendre un échec (dossier output/tiktok_debug)."""
    try:
        base = Path(getattr(ctx, "output_dir", BASE_DIR / "output")) / "tiktok_debug"
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        driver.switch_to.default_content()
        driver.save_screenshot(str(base / f"{stamp}_{tag}.png"))
        (base / f"{stamp}_{tag}.html").write_text(driver.page_source[:400000], encoding="utf-8")
        _log(ctx, f"TikTok : capture de debug -> {base.name}/{stamp}_{tag}.png "
                  f"(url : {driver.current_url[:100]})", level="WARN")
    except Exception:
        pass


def _cliquer(driver, el):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.5)
        el.click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)


def _boutons_par_texte(driver, textes, dans_dialogue=False):
    xp = "//div[@role='dialog']//button" if dans_dialogue else "//button | //div[@role='button']"
    out = []
    try:
        for b in driver.find_elements(By.XPATH, xp):
            try:
                if b.is_displayed() and (b.text or "").strip().lower() in textes:
                    out.append(b)
            except Exception:
                continue
    except Exception:
        pass
    return out


def _fermer_popups(driver, ctx):
    """Ferme les fenêtres d'information (« Got it », « Not now »…)."""
    for b in _boutons_par_texte(driver, POPUP_TEXTS, dans_dialogue=True):
        try:
            _cliquer(driver, b)
            _log(ctx, f"TikTok : fenêtre fermée ({(b.text or '').strip()}).")
            _pause(0.5, 1)
        except Exception:
            pass
    # vérification automatique du contenu : on refuse (sinon bloque la publication)
    try:
        dlg = driver.find_elements(By.XPATH, "//div[@role='dialog']")
        for d in dlg:
            if d.is_displayed() and "content check" in (d.text or "").lower():
                for b in d.find_elements(By.TAG_NAME, "button"):
                    if (b.text or "").strip().lower() == "cancel":
                        _cliquer(driver, b)
                        _pause(0.5, 1)
    except Exception:
        pass


def _chercher_input_fichier(driver):
    driver.switch_to.default_content()
    els = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
    if els:
        return els[0]
    for fr in driver.find_elements(By.TAG_NAME, "iframe"):
        try:
            driver.switch_to.default_content()
            driver.switch_to.frame(fr)
            els = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if els:
                return els[0]                 # on reste dans l'iframe pour la suite
        except Exception:
            continue
    driver.switch_to.default_content()
    return None


def _preparer_legende(legende, titres):
    t = legende or " ".join(x for x in (titres or []) if x)
    t = re.sub("[^\u0000-\uFFFF]", "", t)           # chromedriver : caractères BMP uniquement (pas d'emoji)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 1800:
        t = t[:1800].rsplit(" ", 1)[0]
    return (t or "أهم أخبار الأسواق المالية اليوم #الأسواق_المالية") + " "


def _saisir_legende(driver, texte, ctx):
    editeur = None
    fin = time.time() + 120
    while time.time() < fin and editeur is None:
        for sel in ("div.public-DraftEditor-content[contenteditable='true']",
                    "div[contenteditable='true']"):
            for e in driver.find_elements(By.CSS_SELECTOR, sel):
                try:
                    if e.is_displayed():
                        editeur = e
                        break
                except Exception:
                    continue
            if editeur:
                break
        if editeur is None:
            time.sleep(2)
    if editeur is None:
        _log(ctx, "TikTok : zone de légende introuvable.", level="ERROR")
        return False
    _cliquer(driver, editeur)
    _pause()
    mod = Keys.COMMAND if sys.platform == "darwin" else Keys.CONTROL
    editeur.send_keys(mod, "a")
    editeur.send_keys(Keys.BACKSPACE)               # efface le nom du fichier pré-rempli
    _pause(0.5, 1)
    for i in range(0, len(texte), 25):
        editeur.send_keys(texte[i:i + 25])
        time.sleep(random.uniform(0.03, 0.12))
    _pause(1.0, 2.0)
    _log(ctx, f"TikTok : légende saisie ({len(texte)} caractères).")
    return True


def _trouver_bouton_publier(driver):
    for sel in ("button[data-e2e='post_video_button']", "button[data-e2e='post-button']"):
        for b in driver.find_elements(By.CSS_SELECTOR, sel):
            return b
    for b in driver.find_elements(By.XPATH, "//button"):
        try:
            if (b.text or "").strip().lower() in ("post", "publish"):
                return b
        except Exception:
            continue
    return None


def _bouton_actif(b):
    try:
        return (b.is_displayed() and b.is_enabled()
                and (b.get_attribute("aria-disabled") or "").lower() != "true"
                and (b.get_attribute("data-disabled") or "").lower() != "true")
    except Exception:
        return False


def _succes(driver):
    try:
        url = driver.current_url.lower()
        if "/content" in url and "upload" not in url:
            return True
        driver.switch_to.default_content()
        page = (driver.find_element(By.TAG_NAME, "body").text or "").lower()
        return any(m in page for m in SUCCESS_MARKERS)
    except Exception:
        return False


# ----------------------------------------------------------------------
# Publication
# ----------------------------------------------------------------------
def publier_video_tiktok(chemin_video, titres, legende, ctx=None):
    driver = None
    session = None
    try:
        if os.getenv("TIKTOK_ENABLED", "1") != "1":
            _log(ctx, "TikTok désactivé (TIKTOK_ENABLED != 1).")
            return None
        chemin = Path(chemin_video).resolve()
        if not chemin.is_file():
            _log(ctx, f"TikTok : fichier vidéo introuvable ({chemin})", level="ERROR")
            return None
        session = _charger_session(ctx)
        if not session:
            return None

        _log(ctx, "━━ TikTok ━━", level="STEP")
        headless = os.getenv("TIKTOK_HEADLESS", "1") == "1"
        driver = _creer_driver(headless, ctx)
        if not driver:
            return None
        if not _injecter_cookies(driver, session["cookies"], ctx):
            _log(ctx, "TikTok : aucun cookie accepté par le navigateur.", level="ERROR")
            return None

        # 1) page d'upload + envoi du fichier
        champ = None
        for url in UPLOAD_URLS:
            driver.get(url)
            _pause(4, 6)
            if "login" in driver.current_url.lower():
                _log(ctx, "TikTok : session EXPIRÉE (redirigé vers la connexion). Relancez "
                          "tiktok_login_local.py puis mettez à jour le secret TIKTOK_SESSION_B64.",
                     level="ERROR")
                _debug(driver, ctx, "session_expiree")
                return None
            _fermer_popups(driver, ctx)
            fin = time.time() + 40
            while time.time() < fin and champ is None:
                champ = _chercher_input_fichier(driver)
                if champ is None:
                    time.sleep(2)
            if champ is not None:
                break
        if champ is None:
            _log(ctx, "TikTok : champ d'envoi de fichier introuvable (captcha / vérification "
                      "humaine ou page modifiée ?).", level="ERROR")
            _debug(driver, ctx, "pas_de_champ_fichier")
            return None
        _log(ctx, f"TikTok : envoi de {chemin.name} ({chemin.stat().st_size // 1024} Ko)…")
        champ.send_keys(str(chemin))
        _pause(3, 5)

        # 2) légende
        _fermer_popups(driver, ctx)
        if not _saisir_legende(driver, _preparer_legende(legende, titres), ctx):
            _debug(driver, ctx, "pas_de_legende")
            return None

        # 3) attente de la fin du traitement : le bouton « Post » devient actif
        limite = int(os.getenv("TIKTOK_UPLOAD_TIMEOUT", "300") or 300)
        _log(ctx, f"TikTok : attente du traitement de la vidéo (max {limite}s)…")
        fin = time.time() + limite
        bouton = None
        while time.time() < fin:
            _fermer_popups(driver, ctx)
            b = _trouver_bouton_publier(driver)
            if b is not None and _bouton_actif(b):
                bouton = b
                break
            time.sleep(3)
        if bouton is None:
            _log(ctx, "TikTok : le bouton Publier n'est jamais devenu actif.", level="ERROR")
            _debug(driver, ctx, "bouton_inactif")
            return None

        # 4) publication
        _pause(1.5, 3)
        _cliquer(driver, bouton)
        _log(ctx, "TikTok : clic sur Publier…")
        fin = time.time() + 120
        while time.time() < fin:
            for b in _boutons_par_texte(driver, CONFIRM_TEXTS, dans_dialogue=True):
                _cliquer(driver, b)
                _log(ctx, "TikTok : confirmation « Post now » cliquée.")
                _pause(1, 2)
            if _succes(driver):
                _log(ctx, "TikTok : vidéo PUBLIÉE.", level="OK")
                _sauver_cookies_local(driver, session, ctx)
                return True
            time.sleep(3)
        _log(ctx, "TikTok : publication non confirmée (voir la capture de debug).", level="ERROR")
        _debug(driver, ctx, "non_confirme")
    except Exception as e:
        _log(ctx, f"TikTok : erreur inattendue {type(e).__name__} : {str(e)[:300]}", level="ERROR")
        _log(ctx, traceback.format_exc(limit=3))
        if driver is not None:
            _debug(driver, ctx, "exception")
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
    return None


def _sauver_cookies_local(driver, session, ctx):
    """En local (fichier), rafraîchit les cookies pour prolonger la session."""
    try:
        if os.getenv("TIKTOK_SESSION_B64", "").strip() or not SESSION_FILE.is_file():
            return
        driver.switch_to.default_content()
        cookies = [c for c in driver.get_cookies() if "tiktok" in (c.get("domain") or "")]
        if cookies:
            session["cookies"] = cookies
            SESSION_FILE.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# ----------------------------------------------------------------------
# Test local : python publie_video_tiktok_2.py video.mp4 "légende"
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage : python publie_video_tiktok_2.py ma_video.mp4 "légende"')
    os.environ.setdefault("TIKTOK_HEADLESS", "0")       # fenêtre visible pour le test
    res = publier_video_tiktok(sys.argv[1], [], sys.argv[2] if len(sys.argv) > 2 else "", None)
    print("RÉSULTAT :", "PUBLIÉ" if res else "ÉCHEC")
