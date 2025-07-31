
# 🧠 Generative AI Content Pipeline with Ethical Filtering

## 📌 Objectif
Ce projet implémente une pipeline complète et automatisée qui :
1. Génère du texte à partir de prompts (`distilGPT2`)
2. Vérifie la qualité via un résumé (`facebook/bart-base`)
3. Applique un filtre éthique (détection de toxicité avec `toxic-bert`)
4. Évalue le texte via des métriques comme la perplexité

---

## 🧱 Structure des modules

```
app/
├── main.py         # Script principal : génération, résumé, filtrage, évaluation
├── main_old.py     # Ancienne version non utilisée
requirements.txt    # Dépendances Python
README.md           # Ce fichier
```

---

## 🚀 Pipeline de traitement

```text
prompt (str)
   ↓
generate_text()         → Texte généré (str)
   ↓
summarize_text()        → Résumé (str)
   ↓
check_similarity()      → Vérifie fidélité au prompt
   ↓
automatic_filter()      → Détecte contenu inapproprié
   ↓
compute_perplexity()    → Évalue cohérence linguistique
```

---

## 🧪 Modèles utilisés

| Tâche               | Modèle                         |
|---------------------|--------------------------------|
| Génération          | distilgpt2                     |
| Résumé              | facebook/bart-base             |
| Filtrage éthique    | unitary/toxic-bert             |

---

## 📥 Entrées / Sorties

| Fonction             | Entrée(s)     | Sortie(s)                       |
|----------------------|---------------|---------------------------------|
| `generate_text()`    | `prompt:str`  | `str` (texte généré)            |
| `summarize_text()`   | `text:str`    | `str` (résumé)                  |
| `automatic_filter()` | `text:str`    | `(bool, List[str])`             |
| `check_similarity()` | `prompt,str`  | `(bool, float)` (ROUGE-L score) |
| `compute_perplexity()` | `List[str]` | `float`                         |

---

## ⚙️ Lancer le projet

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

---

## 📌 Notes CPU

- Utilisation de modèles distillés pour performance légère.
- Aucune dépendance GPU requise.
- Idéal pour exécutions locales ou sur serveurs modestes.

---

## 🧑‍⚖️ Filtrage éthique

- Utilisation de `toxic-bert` pour classer chaque texte généré.
- Tout texte marqué comme **toxique** (score > 0.6) est rejeté ou signalé.

---

## 📈 Évaluation

- **Perplexité** (cohérence linguistique)
- **ROUGE-L** (fidélité du résumé par rapport au prompt)

---
