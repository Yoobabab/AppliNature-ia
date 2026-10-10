// Vérifie que le moteur de l'appli (src/lib/ia/moteur.ts, copie exacte) donne
// la même réponse que l'entraînement Keras sur les images de référence.
// node --experimental-strip-types outils/ia/verifier.ts modele.an1 reference.json
import { readFileSync } from 'node:fs';
import { Modele } from '../../src/lib/ia/moteur.ts';

const [fichier, refFichier] = process.argv.slice(2);
const m = new Modele(new Uint8Array(readFileSync(fichier)));
const ref = JSON.parse(readFileSync(refFichier, 'utf8')) as { taille: number; images: { rgba: string; logits: number[] }[] };
let pire = 0;
for (const im of ref.images) {
  const rgba = new Uint8Array(Buffer.from(im.rgba, 'base64'));
  const T = ref.taille;
  const x = new Float32Array(T * T * 3);
  for (let p = 0; p < T * T; p++) for (let c = 0; c < 3; c++) x[p * 3 + c] = rgba[p * 4 + c] / 127.5 - 1;
  const l = Array.from(m.executer({ d: x, h: T, l: T, c: 3 }).d);
  pire = Math.max(pire, ...im.logits.map((v, i) => Math.abs(v - l[i])));
  const tri = [...im.logits].sort((a, b) => b - a);
  if (tri[0] - tri[1] > 0.5 && l.indexOf(Math.max(...l)) !== im.logits.indexOf(tri[0])) throw new Error('le moteur ne donne pas la même 1re idée que Keras');
}
console.log(`Vérification : écart max avec Keras ${pire.toFixed(4)}`);
if (pire >= 0.25) throw new Error(`écart trop grand avec Keras : ${pire}`);
