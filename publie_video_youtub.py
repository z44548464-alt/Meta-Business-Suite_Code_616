"""Publication d'un YouTube Short pour Money.impact.

Utilise uniquement `requests` (OAuth2 refresh token + upload resumable de l'API YouTube Data v3).
Secrets GitHub :
  ID_CLIENT_FB              -> client ID Google (YouTube)
  CODE_SECRET_DU_CLIENT_FB  -> client secret Google (YouTube)
  YOUTUBE_REFRESH_TOKEN     -> obtenu une fois avec youtube_get_refresh_token.py
Options : YOUTUBE_ENABLED ("1"/"0"), YOUTUBE_PRIVACY (public|unlisted|private), YOUTUBE_CATEGORY_ID.
Ne lève jamais d'exception : retourne l'ID de la vidéo ou None.
"""
import json
import os
import re
import time
from pathlib import Path

import requests

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
YT_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "25")          # 25 = News & Politics
YT_PRIVACY = os.getenv("YOUTUBE_PRIVACY", "public")
YT_HASHTAGS = ("#Shorts #الأسواق_المالية #فوركس #الذهب #البورصة #اقتصاد #أخبار_اقتصادية "
               "#تداول #استثمار #forex #trading #gold #markets #money_impact")
YT_TAGS = ["Shorts", "الأسواق المالية", "فوركس", "الذهب", "البورصة", "اقتصاد", "أخبار اقتصادية",
           "تداول", "استثمار", "forex", "trading", "gold", "markets", "money impact"]


def _log(ctx, msg, level="INFO"):
    fn = getattr(ctx, "log", None)
    if fn:
        fn(msg, level=level)
    else:
        print(f"[{level}] {msg}", flush=True)


def _secret(*names):
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return ""


def _erreur_google(resp):
    """Message lisible à partir d'une réponse d'erreur Google."""
    try:
        err = resp.json().get("error", {})
        if isinstance(err, str):
            return f"{err} : {resp.json().get('error_description', '')}", err
        raison = (err.get("errors") or [{}])[0].get("reason", "")
        return f"{err.get('message', '')[:250]} (reason={raison})", raison
    except Exception:
        return resp.text[:250], ""


def _access_token(client_id, client_secret, refresh, ctx):
    try:
        r = requests.post(TOKEN_URL, data={
            "client_id": client_id, "client_secret": client_secret,
            "refresh_token": refresh, "grant_type": "refresh_token"}, timeout=40)
        if r.ok and r.json().get("access_token"):
            return r.json()["access_token"]
        msg, code = _erreur_google(r)
        _log(ctx, f"YouTube : jeton refusé -> {msg}", level="ERROR")
        if code == "invalid_grant":
            _log(ctx, "   -> refresh token expiré/révoqué : passe l'écran de consentement OAuth "
                      "en « En production » puis régénère le token.", level="ERROR")
    except Exception as e:
        _log(ctx, f"YouTube : jeton impossible ({type(e).__name__} {str(e)[:150]})", level="ERROR")
    return None


def construire_titre(titres):
    """Titre court (<= 100 caractères) terminé par #Shorts, à partir du 1er titre disponible."""
    base = ""
    for t in titres or []:
        t = re.sub(r"\n*\s*المصدر\s*[:：].*$", "", t or "", flags=re.S)
        t = re.sub(r"#\S+", "", t).replace("<", "").replace(">", "")
        t = " ".join(t.split())
        if t:
            base = t
            break
    if not base:
        base = "أهم أخبار الأسواق المالية اليوم 📊"
    if len(base) > 70:
        base = base[:70].rsplit(" ", 1)[0]
    return f"{base} #Shorts"[:100]


def construire_description(legende):
    desc = (legende or "").replace("<", "").replace(">", "").strip()
    desc = re.sub(r"(?:\s*#\S+)+\s*$", "", desc).rstrip()      # évite les hashtags en double
    return (desc[:4300] + "\n\n" + YT_HASHTAGS)[:5000]


def publier_short_youtube(chemin_video, titres, legende, ctx=None):
    """Upload `chemin_video` (MP4 vertical 9:16, < 3 min) comme YouTube Short."""
    try:
        if os.getenv("YOUTUBE_ENABLED", "1") != "1":
            _log(ctx, "YouTube désactivé (YOUTUBE_ENABLED != 1).")
            return None
        client_id = _secret("ID_CLIENT_FB", "YOUTUBE_CLIENT_ID")
        client_secret = _secret("CODE_SECRET_DU_CLIENT_FB", "YOUTUBE_CLIENT_SECRET")
        refresh = _secret("YOUTUBE_REFRESH_TOKEN")
        manque = [n for n, v in (("ID_CLIENT_FB", client_id),
                                 ("CODE_SECRET_DU_CLIENT_FB", client_secret),
                                 ("YOUTUBE_REFRESH_TOKEN", refresh)) if not v]
        if manque:
            _log(ctx, "YouTube IGNORÉ : secret(s) absent(s) -> " + ", ".join(manque), level="WARN")
            return None
        chemin = Path(chemin_video)
        if not chemin.is_file():
            _log(ctx, f"YouTube : fichier vidéo introuvable ({chemin})", level="ERROR")
            return None

        _log(ctx, "━━ YouTube Short ━━", level="STEP")
        token = _access_token(client_id, client_secret, refresh, ctx)
        if not token:
            return None

        titre = construire_titre(titres)
        meta = {
            "snippet": {"title": titre, "description": construire_description(legende),
                        "tags": YT_TAGS, "categoryId": YT_CATEGORY_ID,
                        "defaultLanguage": "ar", "defaultAudioLanguage": "ar"},
            "status": {"privacyStatus": YT_PRIVACY, "selfDeclaredMadeForKids": False,
                       "embeddable": True, "publicStatsViewable": True},
        }
        taille = chemin.stat().st_size
        _log(ctx, f"YouTube : titre « {titre} » | {taille // 1024} Ko | {YT_PRIVACY}")

        for essai in range(1, 4):
            try:
                init = requests.post(
                    UPLOAD_URL, params={"uploadType": "resumable", "part": "snippet,status"},
                    headers={"Authorization": f"Bearer {token}",
                             "Content-Type": "application/json; charset=UTF-8",
                             "X-Upload-Content-Length": str(taille),
                             "X-Upload-Content-Type": "video/mp4"},
                    data=json.dumps(meta, ensure_ascii=False).encode("utf-8"), timeout=60)
                if not init.ok:
                    msg, raison = _erreur_google(init)
                    _log(ctx, f"YouTube : ÉCHEC initialisation (HTTP {init.status_code}) -> {msg}",
                         level="ERROR")
                    if raison in ("quotaExceeded", "dailyLimitExceeded", "uploadLimitExceeded"):
                        _log(ctx, "   -> quota/limite d'upload atteint : réessaie demain.",
                             level="ERROR")
                    if init.status_code < 500:
                        return None            # erreur définitive : inutile de réessayer
                    time.sleep(5 * essai)
                    continue
                location = init.headers.get("Location")
                if not location:
                    _log(ctx, "YouTube : pas d'URL d'upload renvoyée.", level="ERROR")
                    return None

                _log(ctx, f"YouTube : envoi du fichier (essai {essai}/3)…")
                with chemin.open("rb") as f:
                    up = requests.put(location, data=f, timeout=900,
                                      headers={"Content-Type": "video/mp4",
                                               "Content-Length": str(taille)})
                if up.status_code in (200, 201):
                    data = up.json()
                    vid = data.get("id")
                    prive = (data.get("status") or {}).get("privacyStatus")
                    _log(ctx, f"YouTube : Short PUBLIÉ -> https://youtube.com/shorts/{vid}",
                         level="OK")
                    if YT_PRIVACY == "public" and prive and prive != "public":
                        _log(ctx, f"YouTube : la vidéo est en « {prive} » : le projet Google Cloud "
                                  "n'est pas audité (vidéos forcées en privé). Demande l'audit de "
                                  "l'API YouTube Data.", level="WARN")
                    return vid
                msg, _ = _erreur_google(up)
                _log(ctx, f"YouTube : ÉCHEC upload (HTTP {up.status_code}) -> {msg}", level="ERROR")
                if up.status_code < 500:
                    return None
                time.sleep(5 * essai)
            except requests.RequestException as e:
                _log(ctx, f"YouTube : erreur réseau essai {essai}/3 : {type(e).__name__} "
                          f"{str(e)[:150]}", level="WARN")
                time.sleep(5 * essai)
    except Exception as e:
        _log(ctx, f"YouTube : erreur inattendue {type(e).__name__} : {str(e)[:250]}", level="ERROR")
    return None
