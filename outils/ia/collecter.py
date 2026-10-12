"""Robot : rassemble des photos libres de droits (CC0 et CC-BY) pour entraîner
l'IA de reconnaissance, via GBIF (observations naturalistes publiques).

Usage : python collecter.py champignons --dossier donnees --especes 150 --par-espece 250

- Liste des espèces : les plus observées en France (toutes licences confondues,
  c'est juste pour savoir ce que les gens rencontrent).
- Photos : seulement CC0 ou CC-BY, prises en Europe (licences compatibles avec
  un usage commercial). La liste des auteurs est gardée dans credits.csv.
- Une classe « autre » (plantes, feuillages, sol) apprend à l'IA à dire
  « ce n'est pas un champignon ».
"""

import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import glob
import hashlib
import random
import re
import sys
import time
import urllib.parse

import requests
from PIL import Image

API = "https://api.gbif.org/v1"
S = requests.Session()
S.headers["User-Agent"] = "AppliNature-robot/1.0 (projet personnel)"
COTE = 192  # côté des images gardées (l'entraînement en découpe des 160×160)
LICENCES_OK = ("publicdomain/zero", "/by/4.0", "/by/3.0", "/by/2.0", "CC0_1_0", "CC_BY_4_0", "publicdomain/mark")

# Mêmes genres d'arbres que l'appli (src/lib/gbif.ts).
ARBRES = [
    "Quercus", "Castanea", "Fagus", "Pinus", "Betula", "Abies", "Picea", "Larix", "Cedrus", "Acer", "Fraxinus", "Populus",
    "Salix", "Alnus", "Carpinus", "Corylus", "Ulmus", "Tilia", "Platanus", "Juglans", "Robinia", "Celtis", "Sorbus",
    "Arbutus", "Olea", "Juniperus", "Cupressus", "Taxus", "Ilex", "Prunus", "Malus", "Pyrus", "Morus", "Ailanthus",
    "Cercis", "Laurus", "Phillyrea", "Pistacia", "Sambucus", "Crataegus",
]
HERBES = ["Poaceae", "Cyperaceae", "Juncaceae"]

# Pour chaque groupe : ce qu'on inclut (nom, rang GBIF), ce qu'on exclut (genres, familles),
# et ce qui sert d'exemples « autre chose ».
GROUPES = {
    "champignons": {
        "inclure": [("Agaricomycetes", "CLASS"), ("Pezizomycetes", "CLASS")],
        "autre": [("Plantae", "KINGDOM"), ("Bryophyta", "PHYLUM")],
    },
    "arbres": {
        "inclure": [(g, "GENUS") for g in ARBRES],
        "autre": [("Fungi", "KINGDOM"), ("Magnoliopsida", "CLASS"), ("Bryophyta", "PHYLUM")],
        "autre_sans_genres": ARBRES,
    },
    "fleurs": {
        "inclure": [("Magnoliopsida", "CLASS"), ("Liliopsida", "CLASS")],
        "exclure_genres": ARBRES,
        "exclure_familles": HERBES,
        "autre": [("Fungi", "KINGDOM"), ("Pinopsida", "CLASS"), ("Bryophyta", "PHYLUM"), ("Polypodiopsida", "CLASS")],
    },
    "plantes": {
        "inclure": [("Polypodiopsida", "CLASS"), ("Lycopodiopsida", "CLASS"), ("Bryophyta", "PHYLUM")] + [(f, "FAMILY") for f in HERBES],
        "autre": [("Fungi", "KINGDOM"), ("Magnoliopsida", "CLASS")],
        "autre_sans_familles": HERBES,
    },
}


def get(url, params=None, essais=8):
    for i in range(essais):
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 + 3 * i)
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(2 + 3 * i)
    raise RuntimeError(f"GBIF ne répond pas : {url} {params}")


def cle_taxon(nom, rang):
    """Clé GBIF d'un groupe. Certains noms existent dans plusieurs règnes (ex. « Pinus ») : on précise le règne."""
    essais = [{}] if rang == "KINGDOM" else [{"kingdom": "Plantae"}, {"kingdom": "Fungi"}, {}]
    for extra in essais:
        r = get(f"{API}/species/match", {"name": nom, "rank": rang, "strict": "true", "verbose": "true", **extra})
        if "usageKey" in r and r.get("rank") == rang and r.get("matchType") == "EXACT":
            return r["usageKey"]
        # Nom ambigu (ex. « Cedrus ») : on prend le groupe accepté de ce nom parmi les autres possibilités.
        for alt in r.get("alternatives", []):
            if alt.get("rank") == rang and alt.get("status") == "ACCEPTED" and alt.get("canonicalName", nom) == nom and (not extra or alt.get("kingdom") == extra["kingdom"]):
                return alt["usageKey"]
    raise RuntimeError(f"Groupe introuvable dans GBIF : {nom} ({rang})")


def nom_francais(cle):
    r = get(f"{API}/species/{cle}/vernacularNames", {"limit": 300})
    noms = [v for v in r.get("results", []) if v.get("language") in ("fra", "fr")]
    if not noms:
        return None
    noms.sort(key=lambda v: (0 if "TAXREF" in (v.get("source") or "") else 1, len(v["vernacularName"])))
    # TAXREF donne parfois plusieurs noms à la suite (« Amanite tue-mouches, Fausse oronge ») : on garde le premier.
    n = noms[0]["vernacularName"].split(",")[0].strip()
    return n[:1].upper() + n[1:]


def licence_ok(media, occ):
    lic = media.get("license") or occ.get("license") or ""
    return any(x in lic for x in LICENCES_OK)


def url_image(identifiant):
    # iNaturalist : la version « medium » (500 px) suffit et pèse 10 fois moins.
    if "inaturalist" in identifiant and "/original." in identifiant:
        return identifiant.replace("/original.", "/medium.")
    return identifiant


def photos_espece(cle, n, exclure=()):
    """Liste (url, auteur, licence, occurrence) de photos libres, en Europe."""
    out = []
    vus = set()
    # D'abord les photos prises en Europe, puis (s'il en manque) celles du reste du monde.
    for continent, licence in (("EUROPE", "CC0_1_0"), ("EUROPE", "CC_BY_4_0"), (None, "CC0_1_0"), (None, "CC_BY_4_0")):
        offset = 0
        while len(out) < n * 1.4 and offset < 1500:
            params = {"taxonKey": cle, "mediaType": "StillImage", "license": licence, "limit": 300, "offset": offset}
            if continent:
                params["continent"] = continent
            r = get(f"{API}/occurrence/search", params)
            for occ in r.get("results", []):
                if occ.get("taxonKey") in exclure:
                    continue
                for m in (occ.get("media") or [])[:2]:
                    ident = m.get("identifier")
                    if not ident or m.get("type") not in (None, "StillImage") or ident in vus or not licence_ok(m, occ):
                        continue
                    vus.add(ident)
                    out.append(
                        {
                            "url": url_image(ident),
                            "auteur": m.get("rightsHolder") or m.get("creator") or occ.get("recordedBy") or "",
                            "licence": m.get("license") or occ.get("license") or "",
                            "occurrence": occ["key"],
                        }
                    )
            if r.get("endOfRecords"):
                break
            offset += 300
    random.Random(cle).shuffle(out)
    return out


def nom_fichier(p):
    """Nom stable d'une photo (ne dépend pas de l'ordre de la liste) : occurrence + empreinte de l'adresse."""
    return f"{p['occurrence']}-{hashlib.md5(p['url'].encode()).hexdigest()[:8]}.jpg"


def telecharger(p, chemin):
    if os.path.exists(chemin):
        return True
    # Ancien nommage (« occurrence-numéro.jpg ») : on reprend la photo déjà téléchargée.
    dossier = os.path.dirname(chemin)
    for ancien in glob.glob(os.path.join(dossier, f"{p['occurrence']}-[0-9]*.jpg")):
        if re.fullmatch(r"\d+-\d+\.jpg", os.path.basename(ancien)):
            try:
                os.rename(ancien, chemin)
                return True
            except OSError:  # déjà repris par un autre téléchargement
                break
    for i in range(3):
        try:
            r = S.get(p["url"], timeout=30)
            if r.status_code != 200 or len(r.content) > 6_000_000:
                return False
            im = Image.open(io.BytesIO(r.content))
            im = im.convert("RGB")
            l, h = im.size
            if min(l, h) < 120:
                return False
            # carré central (on garde 90 % du petit côté), puis 192×192
            c = int(min(l, h) * 0.9)
            im = im.crop(((l - c) // 2, (h - c) // 2, (l - c) // 2 + c, (h - c) // 2 + c)).resize((COTE, COTE), Image.BICUBIC)
            im.save(chemin, "JPEG", quality=90)
            return True
        except Exception:
            time.sleep(1 + i)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groupe", choices=list(GROUPES))
    ap.add_argument("--dossier", default="donnees")
    ap.add_argument("--especes", type=int, default=150)
    ap.add_argument("--par-espece", type=int, default=250)
    ap.add_argument("--min-photos", type=int, default=60)
    ap.add_argument("--limite-minutes", type=float, default=150, help="au-delà, on s'arrête avec les espèces déjà prêtes")
    a = ap.parse_args()
    g = GROUPES[a.groupe]
    racine = os.path.join(a.dossier, a.groupe)
    os.makedirs(racine, exist_ok=True)

    inclus = [cle_taxon(n, r) for n, r in g["inclure"]]
    params = [("country", "FR"), ("mediaType", "StillImage"), ("facet", "speciesKey"), ("facetLimit", str(a.especes * 3)), ("limit", "0")]
    params += [("taxonKey", str(k)) for k in inclus]
    r = get(f"{API}/occurrence/search?" + urllib.parse.urlencode(params))
    candidates = [(int(c["name"]), c["count"]) for c in r["facets"][0]["counts"]]
    print(f"{len(candidates)} espèces candidates en France", flush=True)

    especes = []
    credits = []
    pool = cf.ThreadPoolExecutor(32)  # téléchargements
    debut = time.time()

    compte = {"reprises": 0, "nouvelles": 0}

    def traiter(cle, nb_fr):
        try:
            return preparer(cle, nb_fr)
        except Exception as e:  # une espèce qui coince ne doit pas tout arrêter
            print(f"  ! espèce {cle} : {e}", flush=True)
            return None

    def preparer(cle, nb_fr):
        """Prépare une espèce : ses photos libres, téléchargées. None si pas assez."""
        dossier = os.path.join(racine, str(cle))
        memo = os.path.join(dossier, "fiche.json")
        # Déjà préparée lors d'un passage précédent (assez de photos, ou GBIF n'en a pas plus) : on reprend
        # sans réinterroger GBIF. Sans ça, chaque passage refaisait tout et la collecte n'avançait plus.
        if os.path.exists(memo):
            try:
                m = json.load(open(memo, encoding="utf-8"))
                assez = m["demande"] >= a.par_espece or m["trouvees"] < m["demande"]
                if assez and m.get("fiche") is None:
                    compte["reprises"] += 1
                    return None
                if assez and len(glob.glob(os.path.join(dossier, "*.jpg"))) >= m["fiche"]["photos"]:
                    m["fiche"]["obsFrance"] = nb_fr
                    compte["reprises"] += 1
                    return m["fiche"], m["lignes"]
            except Exception:
                pass
        compte["nouvelles"] += 1
        if (time.time() - debut) / 60 > a.limite_minutes:
            return None
        info = get(f"{API}/species/{cle}")
        if info.get("rank") != "SPECIES" or info.get("taxonomicStatus") not in ("ACCEPTED", None):
            return None
        if info.get("genus") in g.get("exclure_genres", []) or info.get("family") in g.get("exclure_familles", []):
            return None
        photos = photos_espece(cle, a.par_espece)
        # Une même photo peut apparaître deux fois (même occurrence, même adresse) : une seule fois.
        vus = set()
        photos = [p for p in photos if not (nom_fichier(p) in vus or vus.add(nom_fichier(p)))]
        os.makedirs(dossier, exist_ok=True)

        def retenir(fiche, lignes=None):
            with open(memo, "w", encoding="utf-8") as f:
                json.dump({"demande": a.par_espece, "trouvees": len(photos), "fiche": fiche, "lignes": lignes}, f, ensure_ascii=False)

        if len(photos) < a.min_photos:
            print(f"  - {info.get('canonicalName')} : seulement {len(photos)} photos libres, ignorée", flush=True)
            retenir(None)
            return None
        futurs = {pool.submit(telecharger, p, os.path.join(dossier, nom_fichier(p))): (i, p) for i, p in enumerate(photos)}
        ok = []
        for f in cf.as_completed(futurs):
            i, p = futurs[f]
            if f.result():
                ok.append((i, p))
        ok.sort(key=lambda x: x[0])
        # On garde au plus « par-espece » photos.
        for i, p in ok[a.par_espece :]:
            try:
                os.remove(os.path.join(dossier, nom_fichier(p)))
            except OSError:
                pass
        ok = ok[: a.par_espece]
        if len(ok) < a.min_photos:
            print(f"  - {info.get('canonicalName')} : {len(ok)} photos téléchargées, ignorée", flush=True)
            return None
        nom = nom_francais(cle)
        fiche = {"key": cle, "latin": info.get("canonicalName"), "nom": nom, "genre": info.get("genus"), "famille": info.get("family"), "photos": len(ok), "obsFrance": nb_fr}
        lignes = [[a.groupe, info.get("canonicalName"), p["auteur"], p["licence"], f"https://www.gbif.org/occurrence/{p['occurrence']}"] for i, p in ok]
        print(f"  ✓ {info.get('canonicalName')} ({nom or '—'}) : {len(ok)} photos [{(time.time() - debut) / 60:.0f} min]", flush=True)
        retenir(fiche, lignes)
        return fiche, lignes

    # Plusieurs espèces à la fois (l'attente des serveurs domine), dans l'ordre de popularité.
    especes_pool = cf.ThreadPoolExecutor(5)
    for i in range(0, len(candidates), 5):
        if len(especes) >= a.especes or (time.time() - debut) / 60 > a.limite_minutes:
            break
        lot = candidates[i : i + 5]
        for r in especes_pool.map(lambda c: traiter(*c), lot):
            if r and len(especes) < a.especes:
                especes.append(r[0])
                credits.extend(r[1])
    # Fini si toutes les espèces voulues sont prêtes, ou si toute la liste a été parcourue dans le temps.
    complet = len(especes) >= a.especes or (time.time() - debut) / 60 <= a.limite_minutes
    print(f"Espèces reprises telles quelles : {compte['reprises']}, à (re)préparer : {compte['nouvelles']}", flush=True)
    print(f"{len(especes)} espèces prêtes en {(time.time() - debut) / 60:.0f} min{'' if complet else ' (temps écoulé : collecte à reprendre)'}", flush=True)

    if g.get("autre"):
        # « Autre » : d'autres êtres vivants, pour apprendre « ce n'est pas un champignon / un arbre… ».
        dossier = os.path.join(racine, "autre")
        os.makedirs(dossier, exist_ok=True)
        photos = []
        for nom_t, rang_t in g["autre"]:
            k = cle_taxon(nom_t, rang_t)
            for offset in range(0, 1800, 300):
                r = get(f"{API}/occurrence/search", {"taxonKey": k, "mediaType": "StillImage", "license": "CC0_1_0", "country": "FR", "limit": 300, "offset": offset})
                for occ in r.get("results", []):
                    if occ.get("genus") in g.get("autre_sans_genres", []) or occ.get("family") in g.get("autre_sans_familles", []):
                        continue
                    for m in (occ.get("media") or [])[:1]:
                        if m.get("identifier") and licence_ok(m, occ):
                            photos.append({"url": url_image(m["identifier"]), "auteur": m.get("rightsHolder") or m.get("creator") or "", "licence": m.get("license") or "", "occurrence": occ["key"]})
        random.Random(1).shuffle(photos)
        photos = photos[: a.par_espece * 4]
        res = list(pool.map(lambda ip: telecharger(ip[1], os.path.join(dossier, f"{ip[1]['occurrence']}-{ip[0]}.jpg")), enumerate(photos)))
        for (i, p), r_ in zip(enumerate(photos), res):
            if r_:
                credits.append([a.groupe, "autre", p["auteur"], p["licence"], f"https://www.gbif.org/occurrence/{p['occurrence']}"])
        print(f"  ✓ autre : {sum(res)} photos [{(time.time() - debut) / 60:.0f} min]", flush=True)

    with open(os.path.join(racine, "especes.json"), "w", encoding="utf-8") as f:
        json.dump(especes, f, ensure_ascii=False, indent=1)
    with open(os.path.join(racine, "credits.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["groupe", "espece", "auteur", "licence", "source"])
        w.writerows(credits)
    with open(os.path.join(racine, "etat.json"), "w") as f:
        json.dump({"complet": complet, "especes": len(especes), "par_espece": a.par_espece}, f)
    print(f"Terminé : {len(especes)} espèces, {len(credits)} photos.")
    if len(especes) < 10:
        sys.exit("Trop peu d'espèces : quelque chose ne va pas.")


if __name__ == "__main__":
    main()
