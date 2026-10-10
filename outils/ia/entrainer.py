"""Robot : entraîne l'IA de reconnaissance (MobileNetV2) sur les photos
rassemblées par collecter.py, puis l'exporte au format AN01 lu par l'appli
(src/lib/ia/moteur.ts).

Usage : python entrainer.py champignons --dossier donnees --sortie ../../assets/ia
Options de test : --sans-poids (pas de poids ImageNet), --epoques 0 (export seul).
"""

import argparse
import base64
import json
import os
import random
import struct
import sys
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402

TAILLE = 160
ALPHA = float(os.environ.get("IA_ALPHA", "0.5"))  # largeur du réseau : 0,5 (rapide) ou 0,75 (plus juste, ~2 fois plus lent)
LISSAGE = 0.1  # « personne n'est sûr à 100 % » : l'IA apprend à ne pas être trop sûre d'elle
POIDS = os.environ.get("IA_POIDS", "f16")  # « f16 » (précis) ou « i8 » (2 fois plus léger)


# ---------- Données ----------

def lister(racine):
    especes = json.load(open(os.path.join(racine, "especes.json"), encoding="utf-8"))
    classes = [{"key": e["key"], "latin": e["latin"], "nom": e.get("nom"), "genre": e.get("genre"), "famille": e.get("famille")} for e in especes]
    dossiers = [str(e["key"]) for e in especes]
    if os.path.isdir(os.path.join(racine, "autre")):
        classes.append({"key": -1, "latin": "autre", "nom": "Autre chose"})
        dossiers.append("autre")
    fichiers = []
    for i, d in enumerate(dossiers):
        for f in sorted(os.listdir(os.path.join(racine, d))):
            if f.endswith(".jpg"):
                fichiers.append((os.path.join(racine, d, f), i, f.split("-")[0]))
    # Validation : 10 % des observations (jamais la même observation des deux côtés).
    obs = sorted({(c, o) for _, c, o in fichiers})
    random.Random(42).shuffle(obs)
    val = set(obs[: len(obs) // 10])
    entr = [(f, c) for f, c, o in fichiers if (c, o) not in val]
    valid = [(f, c) for f, c, o in fichiers if (c, o) in val]
    return classes, entr, valid


def lire(chemin):
    return tf.io.decode_jpeg(tf.io.read_file(chemin), channels=3)


def augmenter(img):
    # Recadrage aléatoire (zoom 40-100 % : l'utilisateur serre souvent sur une partie), miroir, lumière et couleurs : la photo
    # du téléphone ne sera jamais cadrée ni éclairée comme celle d'entraînement.
    t = tf.shape(img)[0]
    f = tf.random.uniform([], 0.4, 1.0)
    c = tf.cast(tf.cast(t, tf.float32) * f, tf.int32)
    img = tf.image.random_crop(img, [c, c, 3])
    img = tf.image.resize(img, [TAILLE, TAILLE])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 40.0)
    img = tf.image.random_contrast(img, 0.75, 1.25)
    img = tf.image.random_saturation(img, 0.75, 1.3)
    img = tf.image.random_hue(img, 0.03)
    return tf.clip_by_value(img, 0.0, 255.0)


def centre(img):
    t = tf.shape(img)[0]
    c = tf.cast(tf.cast(t, tf.float32) * 0.85, tf.int32)
    img = tf.image.resize_with_crop_or_pad(img, c, c)
    return tf.image.resize(tf.cast(img, tf.float32), [TAILLE, TAILLE])


def jeu(liste, entrainement, n, lot=64):
    ds = tf.data.Dataset.from_tensor_slices(([f for f, _ in liste], [c for _, c in liste]))
    if entrainement:
        ds = ds.shuffle(len(liste), seed=1)
    ds = ds.map(lambda f, c: (augmenter(tf.cast(lire(f), tf.float32)) if entrainement else centre(lire(f)), c), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, c: (x / 127.5 - 1.0, tf.one_hot(c, n)))
    return ds.batch(lot).prefetch(tf.data.AUTOTUNE)


# ---------- Modèle ----------

def construire(n, poids):
    base = keras.applications.MobileNetV2(input_shape=(TAILLE, TAILLE, 3), alpha=ALPHA, include_top=False, weights=poids)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = keras.layers.Dropout(0.3, name="oubli")(x)
    out = keras.layers.Dense(n, name="classes")(x)
    return keras.Model(base.input, out), base


# ---------- Export AN01 ----------

def entrees(cfg_couche):
    """Noms des couches qui alimentent une couche (formats Keras 2 et 3)."""
    noms = []

    def fouiller(o):
        if isinstance(o, dict):
            if "keras_history" in o:
                noms.append(o["keras_history"][0])
                return
            for v in o.values():
                fouiller(v)
        elif isinstance(o, (list, tuple)):
            if len(o) >= 3 and isinstance(o[0], str) and isinstance(o[1], int) and isinstance(o[2], int):
                noms.append(o[0])
                return
            for v in o:
                fouiller(v)

    fouiller(cfg_couche.get("inbound_nodes", []))
    return noms


def pads_same(n, k, s):
    sortie = -(-n // s)
    total = max((sortie - 1) * s + k - n, 0)
    return total // 2, total - total // 2


def exporter(modele, classes, chemin, infos):
    cfg = {c["name"]: c for c in modele.get_config()["layers"]}
    ops = []  # dicts, avec poids float32 « W » (rangement maison) et biais « B »
    produit = {}  # nom de couche -> (indice d'op ou -1, padding en attente)

    for couche in modele.layers:
        nom = couche.name
        src = entrees(cfg[nom]) if nom in cfg else []
        typ = couche.__class__.__name__
        if typ == "InputLayer":
            produit[nom] = (-1, (0, 0, 0, 0))
            continue
        a, pad = produit[src[0]]
        h_in = couche.input.shape[1] if hasattr(couche.input, "shape") else None
        if typ == "ZeroPadding2D":
            (pt, pb), (pl, pr) = couche.padding
            produit[nom] = (a, (pad[0] + pt, pad[1] + pl, pad[2] + pb, pad[3] + pr))
            continue
        if pad != (0, 0, 0, 0) and typ not in ("Conv2D", "DepthwiseConv2D"):
            raise RuntimeError(f"Remplissage inattendu avant {nom}")
        if typ in ("Conv2D", "DepthwiseConv2D"):
            k = couche.kernel_size[0]
            s = couche.strides[0]
            p = list(pad)
            if couche.padding == "same":
                t_, b_ = pads_same(h_in, k, s)
                l_, r_ = pads_same(couche.input.shape[2], k, s)
                p = [p[0] + t_, p[1] + l_, p[2] + b_, p[3] + r_]
            noyau = couche.get_weights()[0]
            biais = couche.get_weights()[1] if couche.use_bias else None
            if typ == "Conv2D":
                cin, cout = noyau.shape[2], noyau.shape[3]
                if k == 1 and s == 1 and p == [0, 0, 0, 0]:
                    W = noyau[0, 0].T.copy()  # [cout][cin]
                    op = {"t": "pw", "in": a, "cin": int(cin), "cout": int(cout), "act": 0}
                else:
                    W = noyau.transpose(3, 0, 1, 2).reshape(cout, -1).copy()  # [cout][k][k][cin]
                    op = {"t": "conv", "in": a, "k": k, "s": s, "p": p, "cin": int(cin), "cout": int(cout), "act": 0}
                n_out = cout
            else:
                c = noyau.shape[2]
                W = noyau[:, :, :, 0].reshape(k * k, c).copy()  # [k][k][c]
                op = {"t": "dw", "in": a, "k": k, "s": s, "p": p, "c": int(c), "act": 0}
                n_out = c
            op["W"] = W.astype(np.float32)
            op["B"] = (biais if biais is not None else np.zeros(n_out)).astype(np.float32)
            ops.append(op)
            produit[nom] = (len(ops) - 1, (0, 0, 0, 0))
        elif typ == "BatchNormalization":
            gamma, beta, moy, var = couche.get_weights()
            e = gamma / np.sqrt(var + couche.epsilon)
            op = ops[a]
            if op["t"] == "dw":
                op["W"] = op["W"] * e[None, :]
            else:
                op["W"] = op["W"] * e[:, None]
            op["B"] = (op["B"] - moy) * e + beta
            produit[nom] = (a, (0, 0, 0, 0))
        elif typ == "ReLU":
            mv = couche.max_value
            ops[a]["act"] = float(mv) if mv is not None else 1e30
            produit[nom] = (a, (0, 0, 0, 0))
        elif typ == "Add":
            b, _ = produit[src[1]]
            ops.append({"t": "add", "in": a, "in2": b})
            produit[nom] = (len(ops) - 1, (0, 0, 0, 0))
        elif typ == "GlobalAveragePooling2D":
            ops.append({"t": "gap", "in": a})
            produit[nom] = (len(ops) - 1, (0, 0, 0, 0))
        elif typ == "Dropout":
            produit[nom] = (a, (0, 0, 0, 0))
        elif typ == "Dense":
            noyau, biais = couche.get_weights()
            ops.append({"t": "dense", "in": a, "cin": int(noyau.shape[0]), "cout": int(noyau.shape[1]), "W": noyau.T.astype(np.float32).copy(), "B": biais.astype(np.float32)})
            produit[nom] = (len(ops) - 1, (0, 0, 0, 0))
        else:
            raise RuntimeError(f"Couche non gérée : {nom} ({typ})")

    # La dernière op doit être la sortie du modèle.
    sortie = produit[modele.layers[-1].name][0]
    if sortie != len(ops) - 1:
        raise RuntimeError("La sortie du modèle n'est pas la dernière opération")

    # Poids en float16, ou en int8 avec une échelle par canal de sortie.
    donnees = bytearray()

    def aligner():
        while len(donnees) % 4:
            donnees.append(0)

    entete_ops = []
    for op in ops:
        o = {k: v for k, v in op.items() if k not in ("W", "B")}
        if "W" in op and POIDS == "f16":
            o["w"] = len(donnees)
            donnees += op["W"].astype("<f2").tobytes()
            aligner()
            o["sc"] = 0
            o["b"] = len(donnees)
            donnees += op["B"].astype("<f4").tobytes()
        elif "W" in op:
            W = op["W"]
            if op["t"] == "dw":
                m = np.abs(W).max(axis=0)
                sc = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
                q = np.clip(np.round(W / sc[None, :]), -127, 127).astype(np.int8)
            else:
                m = np.abs(W).max(axis=1)
                sc = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
                q = np.clip(np.round(W / sc[:, None]), -127, 127).astype(np.int8)
            o["w"] = len(donnees)
            donnees += q.tobytes()
            aligner()
            o["sc"] = len(donnees)
            donnees += sc.astype("<f4").tobytes()
            o["b"] = len(donnees)
            donnees += op["B"].astype("<f4").tobytes()
        entete_ops.append(o)

    entete = {"version": 1, "poids": POIDS, "taille": TAILLE, "ops": entete_ops, "classes": classes, "infos": infos}
    js = json.dumps(entete, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    with open(chemin, "wb") as f:
        f.write(b"AN01")
        f.write(struct.pack("<I", len(js)))
        f.write(js)
        f.write(b"\0" * ((4 - (8 + len(js)) % 4) % 4))
        f.write(donnees)
    return os.path.getsize(chemin)


def reference(modele, liste, n, chemin):
    """Quelques images de validation et la réponse exacte de Keras, pour vérifier le moteur de l'appli."""
    out = []
    for f, c in liste[:n]:
        img = centre(lire(f)).numpy()
        px = np.round(img).astype(np.uint8)  # l'appli reçoit des octets
        x = px.astype(np.float32)[None] / 127.5 - 1.0
        logits = modele(x, training=False).numpy()[0]
        rgba = np.concatenate([px, np.full((TAILLE, TAILLE, 1), 255, np.uint8)], axis=2)
        out.append({"classe": int(c), "rgba": base64.b64encode(rgba.tobytes()).decode(), "logits": [float(v) for v in logits]})
    json.dump({"taille": TAILLE, "images": out}, open(chemin, "w"))


class Chrono(keras.callbacks.Callback):
    """Arrête l'entraînement quand le temps alloué est dépassé (le robot a 6 h en tout)."""

    def __init__(self, t0, minutes):
        super().__init__()
        self.t0, self.minutes = t0, minutes

    def on_epoch_end(self, epoch, logs=None):
        if (time.time() - self.t0) / 60 > self.minutes:
            print("Temps d'entraînement écoulé : on garde le meilleur modèle.", flush=True)
            self.model.stop_training = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groupe")
    ap.add_argument("--dossier", default="donnees")
    ap.add_argument("--sortie", default="assets/ia")
    ap.add_argument("--reference", default="tests/ia-reference.json")
    ap.add_argument("--epoques", type=int, default=4)
    ap.add_argument("--epoques-fines", type=int, default=24)
    ap.add_argument("--sans-poids", action="store_true")
    ap.add_argument("--limite-minutes", type=float, default=150, help="on arrête d'affiner au-delà (le meilleur modèle est gardé)")
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(1)

    racine = os.path.join(a.dossier, a.groupe)
    classes, entr, valid = lister(racine)
    print(f"{len(classes)} classes, {len(entr)} photos d'entraînement, {len(valid)} de validation", flush=True)
    modele, base = construire(len(classes), None if a.sans_poids else "imagenet")
    perte = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=LISSAGE)
    metriques = [keras.metrics.CategoricalAccuracy(name="top1"), keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")]
    d_entr, d_val = jeu(entr, True, len(classes)), jeu(valid, False, len(classes))
    t0 = time.time()

    if a.epoques > 0:
        # 1) On n'apprend que la dernière couche (le reste garde ce qu'il sait déjà voir).
        base.trainable = False
        modele.compile(optimizer=keras.optimizers.Adam(1e-3), loss=perte, metrics=metriques)
        modele.fit(d_entr, validation_data=d_val, epochs=a.epoques, verbose=2)
    if a.epoques_fines > 0:
        # 2) On affine la fin du réseau, doucement.
        base.trainable = True
        # On fige le début du réseau (formes simples, déjà bien apprises) : IA_FIGER = part figée.
        for c in base.layers[: int(len(base.layers) * float(os.environ.get('IA_FIGER', '0.4')))]:
            c.trainable = False
        for c in base.layers:
            if isinstance(c, keras.layers.BatchNormalization):
                c.trainable = False
        # Pas d'apprentissage qui décroît doucement (cosinus) : grands pas au début, petits à la fin.
        pas = keras.optimizers.schedules.CosineDecay(2e-4, decay_steps=a.epoques_fines * len(d_entr), alpha=0.05)
        modele.compile(optimizer=keras.optimizers.Adam(pas), loss=perte, metrics=metriques)
        modele.fit(
            d_entr,
            validation_data=d_val,
            epochs=a.epoques_fines,
            verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_top1", patience=5, mode="max", restore_best_weights=True), Chrono(t0, a.limite_minutes)],
        )

    res = modele.evaluate(d_val, verbose=0, return_dict=True) if valid else {}
    infos = {
        "groupe": a.groupe,
        "date": time.strftime("%Y-%m-%d"),
        "alpha": ALPHA,
        "photos": len(entr) + len(valid),
        "top1": round(float(res.get("top1", 0)), 4),
        "top3": round(float(res.get("top3", 0)), 4),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print("Résultat :", infos, flush=True)
    os.makedirs(a.sortie, exist_ok=True)
    chemin = os.path.join(a.sortie, f"{a.groupe}.an1")
    taille = exporter(modele, classes, chemin, infos)
    print(f"Modèle exporté : {chemin} ({taille / 1e6:.2f} Mo)", flush=True)
    reference(modele, valid or entr, 4, a.reference)


if __name__ == "__main__":
    sys.exit(main())
