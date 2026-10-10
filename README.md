# AppliNature — robot de l'IA

Entraîne les petites IA de reconnaissance (champignons, arbres, fleurs, plantes…) embarquées dans l'appli AppliNature.

- `outils/ia/collecter.py` : rassemble des photos sous licence libre (CC0, CC-BY) via GBIF ; les auteurs sont listés dans `credits-{groupe}.csv` de chaque Release.
- `outils/ia/entrainer.py` : entraîne un MobileNetV2 (Keras) et l'exporte au format AN01 lu par `src/lib/ia/moteur.ts`.
- Onglet « Actions » > « Entraîner l'IA » > « Run workflow ». Le modèle est publié dans les Releases.
