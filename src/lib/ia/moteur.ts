// Notre petit moteur d'IA : il fait tourner, directement dans le téléphone et
// sans aucun service extérieur, un réseau de neurones de type « MobileNet »
// entraîné par notre robot. Écrit en pur JavaScript pour marcher dans Expo Go.
//
// Format du fichier modèle (produit par outils/ia/exporter.py) :
//   « AN01 » | longueur de l'en-tête (uint32, petit-boutiste) | en-tête JSON (ASCII)
//   | bourrage jusqu'à un multiple de 4 | données (poids int8, échelles et biais float32)
// Les images sont rangées « ligne, colonne, canal » (HWC).

export type Op =
  | { t: 'conv'; in: number; k: number; s: number; p: [number, number, number, number]; cin: number; cout: number; act: number; w: number; sc: number; b: number }
  | { t: 'dw'; in: number; k: number; s: number; p: [number, number, number, number]; c: number; act: number; w: number; sc: number; b: number }
  | { t: 'pw'; in: number; cin: number; cout: number; act: number; w: number; sc: number; b: number }
  | { t: 'add'; in: number; in2: number }
  | { t: 'gap'; in: number }
  | { t: 'dense'; in: number; cin: number; cout: number; w: number; sc: number; b: number };

export type EnTete = {
  version: 1;
  /** Rangement des poids : int8 + échelle par canal, ou float16 (par défaut : i8). */
  poids?: 'i8' | 'f16';
  /** Côté de l'image d'entrée, en pixels. */
  taille: number;
  ops: Op[];
  /** Une entrée par sortie du réseau. */
  classes: { key: number; latin: string; nom?: string; genre?: string; famille?: string }[];
  /** Informations libres (date d'entraînement, précision mesurée…). */
  infos?: Record<string, unknown>;
};

type Tenseur = { d: Float32Array; h: number; l: number; c: number };
type OpPrete = Op & { W: Float32Array; B: Float32Array };

export class Modele {
  readonly entete: EnTete;
  private ops: (Op & { W?: Float32Array; B?: Float32Array })[];
  private derniereUtilisation: number[];

  constructor(octets: Uint8Array) {
    const vue = new DataView(octets.buffer, octets.byteOffset, octets.byteLength);
    const magie = String.fromCharCode(octets[0], octets[1], octets[2], octets[3]);
    if (magie !== 'AN01') throw new Error('Fichier modèle inconnu');
    const n = vue.getUint32(4, true);
    let json = '';
    for (let i = 0; i < n; i += 4096) json += String.fromCharCode(...octets.subarray(8 + i, 8 + Math.min(n, i + 4096)));
    this.entete = JSON.parse(json) as EnTete;
    const debut = 8 + n + ((4 - ((8 + n) % 4)) % 4);
    const donnees = octets.subarray(debut);
    const f32 = (off: number, len: number) => {
      // Copie alignée (les octets du fichier ne sont pas forcément alignés en mémoire).
      const out = new Float32Array(len);
      const v = new DataView(donnees.buffer, donnees.byteOffset + off, len * 4);
      for (let i = 0; i < len; i++) out[i] = v.getFloat32(i * 4, true);
      return out;
    };
    const i8 = (off: number, len: number) => new Int8Array(donnees.buffer, donnees.byteOffset + off, len);
    const f16 = (off: number, len: number) => {
      const out = new Float32Array(len);
      const v = new DataView(donnees.buffer, donnees.byteOffset + off, len * 2);
      for (let i = 0; i < len; i++) out[i] = demi(v.getUint16(i * 2, true));
      return out;
    };

    this.ops = this.entete.ops.map((op) => {
      if (op.t === 'add' || op.t === 'gap') return op;
      const nOut = op.t === 'dw' ? op.c : op.cout;
      const nPar = op.t === 'conv' ? op.k * op.k * op.cin : op.t === 'dw' ? op.k * op.k : op.cin;
      if (this.entete.poids === 'f16') return { ...op, W: f16(op.w, nOut * nPar), B: f32(op.b, nOut) };
      const q = i8(op.w, nOut * nPar);
      const sc = f32(op.sc, nOut);
      const W = new Float32Array(nOut * nPar);
      // Rangement : conv [cout][k][k][cin], dw [k][k][c], pw/dense [cout][cin].
      if (op.t === 'dw') for (let j = 0; j < nPar; j++) for (let c = 0; c < nOut; c++) W[j * nOut + c] = q[j * nOut + c] * sc[c];
      else for (let o = 0; o < nOut; o++) for (let j = 0; j < nPar; j++) W[o * nPar + j] = q[o * nPar + j] * sc[o];
      return { ...op, W, B: f32(op.b, nOut) };
    });

    // Pour libérer la mémoire des résultats intermédiaires dès qu'ils ne servent plus.
    this.derniereUtilisation = this.ops.map(() => -1);
    this.ops.forEach((op, i) => {
      if (op.in >= 0) this.derniereUtilisation[op.in] = i;
      if (op.t === 'add' && op.in2 >= 0) this.derniereUtilisation[op.in2] = i;
    });
  }

  /** Image RGBA (octets 0-255) déjà à la bonne taille → probabilités de chaque classe. */
  predire(rgba: Uint8Array | Uint8ClampedArray, largeur: number, hauteur: number): Float32Array {
    const T = this.entete.taille;
    if (largeur !== T || hauteur !== T) throw new Error(`Image attendue en ${T}×${T}`);
    const x = new Float32Array(T * T * 3);
    for (let p = 0; p < T * T; p++) {
      x[p * 3] = rgba[p * 4] / 127.5 - 1;
      x[p * 3 + 1] = rgba[p * 4 + 1] / 127.5 - 1;
      x[p * 3 + 2] = rgba[p * 4 + 2] / 127.5 - 1;
    }
    const logits = this.executer({ d: x, h: T, l: T, c: 3 }).d;
    return softmax(logits);
  }

  /** Fait passer un tenseur d'entrée dans tout le réseau (utile pour les tests). */
  executer(entree: Tenseur): Tenseur {
    const res: (Tenseur | null)[] = this.ops.map(() => null);
    const lire = (i: number) => (i < 0 ? entree : res[i]!);
    for (let i = 0; i < this.ops.length; i++) {
      const op = this.ops[i];
      const a = lire(op.in);
      let r: Tenseur;
      switch (op.t) {
        case 'conv':
          r = conv(a, op as OpPrete & { t: 'conv' });
          break;
        case 'dw':
          r = depthwise(a, op as OpPrete & { t: 'dw' });
          break;
        case 'pw':
          r = pointwise(a, op as OpPrete & { t: 'pw' });
          break;
        case 'dense':
          r = pointwise(a, { ...(op as OpPrete & { t: 'dense' }), t: 'pw', act: 0 } as OpPrete & { t: 'pw' });
          break;
        case 'add': {
          const b = lire(op.in2);
          const d = new Float32Array(a.d.length);
          for (let k = 0; k < d.length; k++) d[k] = a.d[k] + b.d[k];
          r = { ...a, d };
          break;
        }
        case 'gap': {
          const d = new Float32Array(a.c);
          const n = a.h * a.l;
          for (let p = 0; p < n; p++) for (let c = 0; c < a.c; c++) d[c] += a.d[p * a.c + c];
          for (let c = 0; c < a.c; c++) d[c] /= n;
          r = { d, h: 1, l: 1, c: a.c };
          break;
        }
      }
      res[i] = r;
      // On oublie ce qui ne servira plus.
      for (let j = 0; j < i; j++) if (res[j] && this.derniereUtilisation[j] <= i && j !== this.ops.length - 1) res[j] = null;
    }
    return res[this.ops.length - 1]!;
  }
}

/** Nombre à virgule sur 16 bits → nombre ordinaire. */
function demi(h: number): number {
  const signe = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return signe * m * 5.960464477539063e-8; // 2^-24
  if (e === 31) return m ? NaN : signe * Infinity;
  return signe * (1 + m / 1024) * Math.pow(2, e - 15);
}

function sortie(n: number, k: number, s: number, p1: number, p2: number) {
  return Math.floor((n + p1 + p2 - k) / s) + 1;
}

function activer(v: number, act: number) {
  return act > 0 ? (v < 0 ? 0 : v > act ? act : v) : v;
}

function conv(a: Tenseur, op: OpPrete & { t: 'conv' }): Tenseur {
  const { k, s, cin, cout, act, W, B } = op;
  const [pt, pl, pb, pr] = op.p;
  const h = sortie(a.h, k, s, pt, pb);
  const l = sortie(a.l, k, s, pl, pr);
  const d = new Float32Array(h * l * cout);
  const kk = k * k * cin;
  const patch = new Float32Array(kk);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < l; x++) {
      let j = 0;
      for (let dy = 0; dy < k; dy++) {
        const iy = y * s + dy - pt;
        for (let dx = 0; dx < k; dx++) {
          const ix = x * s + dx - pl;
          const dedans = iy >= 0 && iy < a.h && ix >= 0 && ix < a.l;
          const base = (iy * a.l + ix) * cin;
          for (let c = 0; c < cin; c++) patch[j++] = dedans ? a.d[base + c] : 0;
        }
      }
      const o0 = (y * l + x) * cout;
      for (let o = 0; o < cout; o++) {
        let v = B[o];
        const w0 = o * kk;
        for (let q = 0; q < kk; q++) v += patch[q] * W[w0 + q];
        d[o0 + o] = activer(v, act);
      }
    }
  return { d, h, l, c: cout };
}

function depthwise(a: Tenseur, op: OpPrete & { t: 'dw' }): Tenseur {
  const { k, s, c: C, act, W, B } = op;
  const [pt, pl, pb, pr] = op.p;
  const h = sortie(a.h, k, s, pt, pb);
  const l = sortie(a.l, k, s, pl, pr);
  const d = new Float32Array(h * l * C);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < l; x++) {
      const o0 = (y * l + x) * C;
      for (let c = 0; c < C; c++) d[o0 + c] = B[c];
      for (let dy = 0; dy < k; dy++) {
        const iy = y * s + dy - pt;
        if (iy < 0 || iy >= a.h) continue;
        for (let dx = 0; dx < k; dx++) {
          const ix = x * s + dx - pl;
          if (ix < 0 || ix >= a.l) continue;
          const i0 = (iy * a.l + ix) * C;
          const w0 = (dy * k + dx) * C;
          for (let c = 0; c < C; c++) d[o0 + c] += a.d[i0 + c] * W[w0 + c];
        }
      }
      if (act > 0) for (let c = 0; c < C; c++) d[o0 + c] = activer(d[o0 + c], act);
    }
  return { d, h, l, c: C };
}

function pointwise(a: Tenseur, op: OpPrete & { t: 'pw' }): Tenseur {
  const { cin, cout, act, W, B } = op;
  const n = a.h * a.l;
  const d = new Float32Array(n * cout);
  const x = a.d;
  // 4 sorties à la fois : chaque valeur d'entrée lue sert 4 fois (bien plus rapide sans JIT).
  const o4 = cout - (cout % 4);
  for (let p = 0; p < n; p++) {
    const i0 = p * cin;
    const o0 = p * cout;
    let o = 0;
    for (; o < o4; o += 4) {
      let v0 = B[o];
      let v1 = B[o + 1];
      let v2 = B[o + 2];
      let v3 = B[o + 3];
      let w0 = o * cin;
      let w1 = w0 + cin;
      let w2 = w1 + cin;
      let w3 = w2 + cin;
      for (let q = i0, fin = i0 + cin; q < fin; q++) {
        const xv = x[q];
        v0 += xv * W[w0++];
        v1 += xv * W[w1++];
        v2 += xv * W[w2++];
        v3 += xv * W[w3++];
      }
      d[o0 + o] = activer(v0, act);
      d[o0 + o + 1] = activer(v1, act);
      d[o0 + o + 2] = activer(v2, act);
      d[o0 + o + 3] = activer(v3, act);
    }
    for (; o < cout; o++) {
      let v = B[o];
      const w0 = o * cin;
      for (let q = 0; q < cin; q++) v += x[i0 + q] * W[w0 + q];
      d[o0 + o] = activer(v, act);
    }
  }
  return { d, h: a.h, l: a.l, c: cout };
}

export function softmax(z: Float32Array): Float32Array {
  let m = -Infinity;
  for (const v of z) if (v > m) m = v;
  const out = new Float32Array(z.length);
  let s = 0;
  for (let i = 0; i < z.length; i++) s += out[i] = Math.exp(z[i] - m);
  for (let i = 0; i < z.length; i++) out[i] /= s;
  return out;
}
