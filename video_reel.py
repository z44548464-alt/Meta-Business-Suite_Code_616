"""Module vidéo Reels (Facebook + Instagram) pour Money.impact.

Crée un diaporama 9:16 avec les 3 à 5 dernières images publiées qui n'ont jamais servi dans une
vidéo : Ken Burns + transitions aléatoires + musique aléatoire (music/news_musique*.mp3).
Appelé en dernier par le script principal : publier_video_reel(historique, ctx).
"""
import io
import json
import random
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests
import cloudinary
import cloudinary.uploader
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ----------------------------------------------------------------------
# Réglages
# ----------------------------------------------------------------------
VIDEO_FPS = 30
# Meilleur format pour les Reels Instagram ET Facebook : 9:16, 1080x1920
VIDEO_FORMATS = {"instagram": (1080, 1920), "facebook": (1080, 1920)}
# Durée proportionnelle au nombre d'images : n -> (min s, max s)
VIDEO_DURATIONS = {3: (10, 20), 4: (18, 30), 5: (25, 40)}
VIDEO_TRANSITION_S = 0.8
# Transitions ffmpeg xfade (modifiable : circleopen, wipeleft, smoothleft, ...)
VIDEO_TRANSITIONS = ("fade", "slideleft", "slideright", "slideup", "slidedown")
VIDEO_ZOOM_AMOUNT = 0.12            # Ken Burns : +12 % de zoom sur la durée d'un plan
VIDEO_HASHTAGS = "#الأسواق_المالية #فوركس #الذهب #البورصة #اقتصاد #forex #trading #money_impact"

# Renseignés à chaque appel par publier_video_reel()
C = None
VIDEO_DIR = "video_SIMPLE"
MUSIC_DIR = "music_NEWS"


def _log(*args, **kwargs):
    C.log(*args, **kwargs)


# ----------------------------------------------------------------------
# Historique des vidéos
# ----------------------------------------------------------------------
def _video_history_path():
    nom = ("historique_videos.json" if not C.meta_suffix
           else f"historique_videos{C.meta_suffix.lower()}.json")
    return Path(C.base_dir) / nom


def _charger_hist_videos():
    path = _video_history_path()
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"Historique vidéos illisible ({e}) : repart de zéro.", level="WARN")
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("used_post_ids"), list):
        data["used_post_ids"] = []
    if not isinstance(data.get("videos"), list):
        data["videos"] = []
    return data


def _sauver_hist_videos(data):
    _video_history_path().write_text(json.dumps(data, ensure_ascii=False, indent=4),
                                     encoding="utf-8")


# ----------------------------------------------------------------------
# Sélection + téléchargement des images
# ----------------------------------------------------------------------
def _choisir_posts_video(historique, deja_utilises):
    """Les 3 à 5 posts les plus récents (avec image hébergée) jamais utilisés dans une vidéo."""
    libres = [(pid, v) for pid, v in historique.items()
              if isinstance(v, dict) and v.get("image_url") and v.get("publie_le")
              and pid not in deja_utilises]
    libres.sort(key=lambda x: x[1]["publie_le"], reverse=True)
    if len(libres) < 3:
        return None, len(libres)
    n = random.randint(3, min(5, len(libres)))
    return libres[:n], len(libres)


def _telecharger_images(posts):
    """Télécharge dans output/video/img1.png, img2.png, ... -> [(post_id, Path, titre)]."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for ancien in VIDEO_DIR.glob("*"):
        if ancien.is_file():
            ancien.unlink()
    out = []
    for pid, v in posts:
        idx = len(out) + 1
        try:
            r = requests.get(v["image_url"], timeout=60)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            chemin = VIDEO_DIR / f"img{idx}.png"
            img.save(chemin)
            out.append((pid, chemin, v.get("titre", "")))
            _log(f"Image {idx} téléchargée : {v['image_url']}")
        except Exception as e:
            _log(f"Image ignorée ({type(e).__name__} {str(e)[:100]}) : {v.get('image_url')}",
                 level="WARN")
    return out


def _composer_cadre(src, dst, size):
    """Image au centre + fond flou assombri (plein écran 9:16, sans bandes noires)."""
    w, h = size
    img = Image.open(src).convert("RGB")
    fond = ImageOps.fit(img, (w, h), Image.LANCZOS, centering=(0.5, 0.5)).filter(
        ImageFilter.GaussianBlur(35))
    fond = ImageEnhance.Brightness(fond).enhance(0.45)
    avant = ImageOps.contain(img, (w, h), Image.LANCZOS)
    fond.paste(avant, ((w - avant.width) // 2, (h - avant.height) // 2))
    fond.save(dst)


def _choisir_musique():
    fichiers = sorted(MUSIC_DIR.glob("news_musique*.mp3"))
    if not fichiers:
        _log(f"Aucune musique trouvée dans {MUSIC_DIR} (news_musique*.mp3) : vidéo sans musique.",
             level="WARN")
        return None
    choix = random.choice(fichiers)
    _log(f"Musique choisie : {choix.name} (sur {len(fichiers)})")
    return choix


# ----------------------------------------------------------------------
# Montage ffmpeg
# ----------------------------------------------------------------------
def _rendre_video(items, size, duration, music, plan, sortie):
    """Ken Burns par image + transitions xfade + musique (coupée si longue, bouclée si courte)."""
    w, h = size
    n = len(items)
    fps, T = VIDEO_FPS, VIDEO_TRANSITION_S
    L = (duration + (n - 1) * T) / n          # durée de chaque plan (chevauchement T)
    N = max(2, int(round(L * fps)))           # images par plan

    frames = []
    for i, (_, src, _) in enumerate(items, 1):
        dst = VIDEO_DIR / f"frame_{w}x{h}_{i}.png"
        _composer_cadre(src, dst, size)
        frames.append(dst)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for f in frames:
        cmd += ["-i", str(f)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]      # boucle si trop courte
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]

    amt = VIDEO_ZOOM_AMOUNT
    big_w, big_h = int(w * 1.5) // 2 * 2, int(h * 1.5) // 2 * 2   # sur-échantillonnage anti-saccades
    fc = []
    for i in range(n):
        if plan["zooms"][i] == "in":
            z = f"1+{amt / N:.7f}*on"
        else:
            z = f"{1 + amt:.3f}-{amt / N:.7f}*on"
        fc.append(
            f"[{i}:v]scale={big_w}:{big_h}:flags=lanczos,"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={N}:s={w}x{h}:fps={fps},setsar=1,format=yuv420p[v{i}]")
    cur = "v0"
    for k in range(1, n):
        fc.append(f"[{cur}][v{k}]xfade=transition={plan['transitions'][k - 1]}:"
                  f"duration={T}:offset={k * (L - T):.3f}[x{k}]")
        cur = f"x{k}"
    fc.append(f"[{cur}]fade=t=in:st=0:d=0.4,fade=t=out:st={duration - 0.6:.2f}:d=0.6[vout]")
    fc.append(f"[{n}:a]afade=t=in:st=0:d=0.5,"
              f"afade=t=out:st={max(0, duration - 1.5):.2f}:d=1.5,volume=0.9[aout]")

    cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[aout]",
            "-t", f"{duration:.2f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-maxrate", "8M", "-bufsize", "16M",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", str(sortie)]

    t = time.time()
    _log(f"ffmpeg : montage {w}x{h} | {n} images | {duration:.1f}s | "
         f"transitions={plan['transitions']}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    if res.returncode != 0 or not Path(sortie).is_file():
        _log("ffmpeg ÉCHEC : " + (res.stderr or "")[-700:], level="ERROR")
        return False
    _log(f"Vidéo prête : {Path(sortie).name} ({Path(sortie).stat().st_size // 1024} Ko) "
         f"en {time.time() - t:.0f}s", level="OK")
    return True


# ----------------------------------------------------------------------
# Légende
# ----------------------------------------------------------------------
def _legende_video(titres):
    base_fixe = "أهم أخبار السوق اليوم 📊"
    pied = f"\n\n{C.brand_handle}\n{VIDEO_HASHTAGS}"
    liste = "\n".join(f"- {t}" for t in titres if t)
    msgs = [
        {"role": "system", "content":
            "You write short Instagram/Facebook Reels captions in Modern Standard Arabic for a "
            "financial-markets news page. Translate the given English headlines faithfully into "
            "Arabic as a short bullet list (one line each, starting with '•'), preceded by one short "
            "catchy Arabic intro line. Do NOT add facts, numbers, hashtags, the page name or links. "
            "Max 700 characters. Return ONLY the caption text."},
        {"role": "user", "content": liste},
    ]
    try:
        out = C.llm_chat(msgs, temperature=0.2, max_tokens=900)
    except Exception as e:
        _log(f"Légende LLM : {type(e).__name__} {str(e)[:120]}", level="WARN")
        out = None
    texte = (out or "").strip().strip("`")
    if len(texte) < 20:
        _log("Légende LLM indisponible : légende fixe utilisée.", level="WARN")
        texte = base_fixe
    return (texte[:1700].rstrip() + pied)[:C.ig_caption_limit]


# ----------------------------------------------------------------------
# Hébergement + publication
# ----------------------------------------------------------------------
def _heberger_video(chemin, cloud):
    cloudinary.config(cloud_name=cloud[0], api_key=cloud[1], api_secret=cloud[2], secure=True)
    _log("Cloudinary : envoi de la vidéo…")
    up = cloudinary.uploader.upload_large(str(chemin), resource_type="video",
                                          folder="reels_money_impact", chunk_size=6_000_000)
    url = up.get("secure_url")
    if not (url and url.startswith("https://")):
        raise RuntimeError(f"Cloudinary : URL vidéo invalide : {up}")
    _log(f"Cloudinary : vidéo hébergée -> {url}", level="OK")
    return url


def _reel_instagram(video_url, caption, ig_id, token):
    try:
        _log("Reel Instagram 1/3 : création du conteneur…")
        r = requests.post(f"{C.graph_url}/{ig_id}/media",
                          data={"media_type": "REELS", "video_url": video_url,
                                "caption": caption[:C.ig_caption_limit], "share_to_feed": "true",
                                "access_token": token}, timeout=90)
        data = C.json_or_text(r)
        if not (r.ok and "error" not in data and data.get("id")):
            _log("Reel Instagram : ÉCHEC création -> " + C.explain_graph_error(data), level="ERROR")
            return None
        creation_id = data["id"]

        _log("Reel Instagram 2/3 : traitement de la vidéo par Instagram…")
        statut = None
        for i in range(45):                                   # jusqu'à ~6 min
            sr = requests.get(f"{C.graph_url}/{creation_id}",
                              params={"fields": "status_code,status", "access_token": token},
                              timeout=30).json()
            statut = sr.get("status_code")
            _log(f"      contrôle {i + 1}/45 : {statut}")
            if statut in ("FINISHED", "ERROR", "EXPIRED"):
                if statut != "FINISHED":
                    _log(f"Reel Instagram : détail -> {sr.get('status')}", level="ERROR")
                break
            time.sleep(8)
        if statut != "FINISHED":
            _log(f"Reel Instagram : média non prêt (statut : {statut})", level="ERROR")
            return None

        _log("Reel Instagram 3/3 : publication…")
        pr = requests.post(f"{C.graph_url}/{ig_id}/media_publish",
                           data={"creation_id": creation_id, "access_token": token}, timeout=90)
        pdata = C.json_or_text(pr)
        if pr.ok and "error" not in pdata and pdata.get("id"):
            _log(f"Reel Instagram : PUBLIÉ (id={pdata['id']})", level="OK")
            return pdata["id"]
        _log("Reel Instagram : ÉCHEC publication -> " + C.explain_graph_error(pdata), level="ERROR")
    except Exception as e:
        _log(f"Reel Instagram : exception {type(e).__name__} : {str(e)[:300]}", level="ERROR")
    return None


def _reel_facebook(chemin, caption, page_id, token):
    """Page Reels API : start -> upload binaire -> finish (PUBLISHED)."""
    try:
        _log("Reel Facebook 1/3 : initialisation…")
        r = requests.post(f"{C.graph_url}/{page_id}/video_reels",
                          data={"upload_phase": "start", "access_token": token}, timeout=60)
        d = C.json_or_text(r)
        video_id, upload_url = d.get("video_id"), d.get("upload_url")
        if not (r.ok and "error" not in d and video_id and upload_url):
            _log("Reel Facebook : ÉCHEC start -> " + C.explain_graph_error(d), level="ERROR")
            return None

        _log(f"Reel Facebook 2/3 : envoi du fichier ({chemin.stat().st_size // 1024} Ko)…")
        with chemin.open("rb") as f:
            up = requests.post(upload_url, data=f, timeout=900,
                               headers={"Authorization": f"OAuth {token}", "offset": "0",
                                        "file_size": str(chemin.stat().st_size)})
        ud = C.json_or_text(up)
        if not (up.ok and ud.get("success")):
            _log(f"Reel Facebook : ÉCHEC upload (HTTP {up.status_code}) -> {str(ud)[:300]}",
                 level="ERROR")
            return None

        _log("Reel Facebook 3/3 : publication…")
        fin = requests.post(f"{C.graph_url}/{page_id}/video_reels",
                            data={"upload_phase": "finish", "video_id": video_id,
                                  "video_state": "PUBLISHED", "description": caption[:2200],
                                  "access_token": token}, timeout=90)
        fd = C.json_or_text(fin)
        if not (fin.ok and "error" not in fd):
            _log("Reel Facebook : ÉCHEC finish -> " + C.explain_graph_error(fd), level="ERROR")
            return None
        for i in range(20):                                   # simple suivi du traitement
            st = requests.get(f"{C.graph_url}/{video_id}",
                              params={"fields": "status", "access_token": token},
                              timeout=30).json().get("status", {})
            vs = st.get("video_status")
            _log(f"      statut vidéo {i + 1}/20 : {vs}")
            if vs in ("ready", "error", "expired"):
                break
            time.sleep(6)
        _log(f"Reel Facebook : PUBLIÉ (video_id={video_id})", level="OK")
        return video_id
    except Exception as e:
        _log(f"Reel Facebook : exception {type(e).__name__} : {str(e)[:300]}", level="ERROR")
    return None


# ----------------------------------------------------------------------
# Point d'entrée
# ----------------------------------------------------------------------
def publier_video_reel(historique, ctx):
    """Dernière étape : vidéo Reel (3-5 dernières images jamais utilisées) -> Facebook + Instagram.
    Ne lève jamais d'exception et ne change pas le code de sortie du script."""
    global C, VIDEO_DIR, MUSIC_DIR
    import os
    C = ctx
    VIDEO_DIR = Path(C.output_dir) / "video"
    MUSIC_DIR = Path(os.getenv("MUSIC_DIR", str(Path(C.base_dir) / "music")))

    if os.getenv("VIDEO_ENABLED", "1") != "1":
        _log("Vidéo Reel désactivée (VIDEO_ENABLED=0).")
        return
    try:
        _log("━━ Vidéo Reel : diaporama des dernières images ━━", level="STEP")
        if not shutil.which("ffmpeg"):
            _log("ffmpeg introuvable (sudo apt-get install -y ffmpeg) : vidéo ignorée.",
                 level="ERROR")
            return
        hv = _charger_hist_videos()
        deja = set(hv["used_post_ids"])

        # 1) sélection : 3 à 5 posts récents dont l'image n'a JAMAIS servi dans une vidéo
        posts, nb_libres = _choisir_posts_video(historique, deja)
        if not posts:
            _log(f"Seulement {nb_libres} image(s) neuve(s) disponible(s) (minimum 3) : "
                 "pas de vidéo pour ce run.", level="WARN")
            return

        # 2) accès Meta / Cloudinary
        token, page_id, ig_id = (C.menv("PAGE_ACCESS_TOKEN"), C.menv("PAGE_ID"),
                                 C.menv("INSTAGRAM_BUSINESS_ID"))
        cloud = (os.getenv("CLOUDINARY_CLOUD_NAME", ""), os.getenv("CLOUDINARY_API_KEY", ""),
                 os.getenv("CLOUDINARY_API_SECRET", ""))
        if not token or not (page_id or ig_id):
            _log("Token ou identifiants Meta absents : vidéo ignorée.", level="ERROR")
            return
        if ig_id and not all(cloud):
            _log("Cloudinary incomplet : Instagram sera ignoré pour la vidéo.", level="WARN")
            ig_id = ""

        # 3) téléchargement : output/video/img1.png, img2.png, ...
        items = _telecharger_images(posts)
        if len(items) < 3:
            _log(f"Seulement {len(items)} image(s) téléchargée(s) : vidéo annulée.", level="WARN")
            return
        n = len(items)

        # 4) paramètres aléatoires (communs aux deux formats)
        duree = round(random.uniform(*VIDEO_DURATIONS[n]), 1)
        musique = _choisir_musique()
        plan = {"transitions": [random.choice(VIDEO_TRANSITIONS) for _ in range(n - 1)],
                "zooms": [random.choice(("in", "out")) for _ in range(n)]}
        _log(f"Vidéo : {n} images | {duree}s | zooms={plan['zooms']}")

        # 5) rendu (un fichier par format distinct ; ici 9:16 pour les deux)
        rendus = {}
        for plateforme, size in VIDEO_FORMATS.items():
            if size not in rendus:
                sortie = VIDEO_DIR / f"reel_{size[0]}x{size[1]}.mp4"
                if not _rendre_video(items, size, duree, musique, plan, sortie):
                    _log("Montage échoué : vidéo annulée (images non marquées).", level="ERROR")
                    return
                rendus[size] = sortie

        caption = _legende_video([t for _, _, t in items])
        _log(f"Légende vidéo : {len(caption)} caractères")

        # 6) publication
        fb_id = ig_post_id = None
        if page_id:
            fb_id = _reel_facebook(rendus[VIDEO_FORMATS["facebook"]], caption, page_id, token)
        if ig_id:
            try:
                url = _heberger_video(rendus[VIDEO_FORMATS["instagram"]], cloud)
                ig_post_id = _reel_instagram(url, caption, ig_id, token)
            except Exception as e:
                _log(f"Instagram vidéo : {type(e).__name__} {str(e)[:300]}", level="ERROR")

        # 7) suivi : une réussite suffit pour marquer les images comme utilisées
        if fb_id or ig_post_id:
            hv["used_post_ids"].extend(pid for pid, _, _ in items)
            hv["videos"].append({
                "date": datetime.now(timezone.utc).isoformat(),
                "post_ids": [pid for pid, _, _ in items],
                "facebook_video_id": fb_id, "instagram_media_id": ig_post_id,
                "duree_s": duree, "musique": musique.name if musique else None,
                "transitions": plan["transitions"],
            })
            _sauver_hist_videos(hv)
            _log(f"VIDÉO PUBLIÉE (Facebook : {'oui' if fb_id else 'non'} | "
                 f"Instagram : {'oui' if ig_post_id else 'non'}) -> historique vidéos mis à jour",
                 level="OK")
            C.step_summary(f"**Vidéo Reel** : {n} images, {duree}s — "
                           f"FB={'✅' if fb_id else '❌'} IG={'✅' if ig_post_id else '❌'}")
        else:
            _log("Vidéo NON publiée sur aucune plateforme : images conservées pour le prochain run.",
                 level="ERROR")
    except Exception as e:
        _log(f"Vidéo Reel : erreur inattendue {type(e).__name__} : {str(e)[:300]}", level="ERROR")
        _log(traceback.format_exc(limit=4))
