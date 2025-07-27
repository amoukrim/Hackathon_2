
# 🧠 TextGen Pipeline

Un projet complet de génération de texte éthique avec Streamlit, utilisant des modèles de NLP, du contrôle qualité et des outils de visualisation.

---

## 📁 Structure du projet

```
textgen_pipeline/
├── app/
│   └── main.py             # App Streamlit principale
├── data/                   # (ex. prompts ou corpus personnalisés)
├── tests/                  # (tests unitaires futurs)
├── requirements.txt        # Dépendances Python
└── README.md               # Ce fichier
```

---

## ⚙️ Installation

1. Cloner ou extraire le projet :
```bash
cd textgen_pipeline
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

---

## 🚀 Lancer l'application

```bash
streamlit run app/main.py
```

Cela ouvre automatiquement l'application dans votre navigateur (`http://localhost:8501`).

---

## 🎛️ Fonctionnalités incluses

- Génération de texte avec `distilGPT2`
- Résumé automatique (BART)
- Contrôle qualité :
  - Similarité avec le prompt
  - Résumé pertinent
  - Grammaire correcte
  - Absence de biais / haine / exagération
- Dashboard en temps réel :
  - Graphiques de performance
  - Histogramme des causes de rejet
- Fine-tuning personnalisé (optionnel)

---

## 💡 Technologies choisies et justifications

| Outil / Librairie               | Rôle                                         | Justification principale                        |
|----------------------------------|-----------------------------------------------|-------------------------------------------------|
| **Streamlit**                   | Interface web légère et interactive           | Simple, rapide pour prototypage et démo         |
| **Transformers**                | Génération et résumé (GPT2, BART)             | Modèles pré-entraînés robustes et standardisés  |
| **Sentence-Transformers**       | Calcul de similarité                         | Utile pour mesurer la cohérence texte/prompt    |
| **TextBlob**                    | Analyse de sentiments                         | Très léger pour mesurer la polarité             |
| **language-tool-python**        | Vérification grammaticale                     | Détection fiable des erreurs simples            |
| **Torch / PyTorch**             | Backend pour modèles et fine-tuning           | Compatible avec Transformers                    |
| **Scikit-learn**                | Accuracy / évaluation                        | Pour les métriques de classification            |
| **Plotly**                      | Visualisation dynamique dans Streamlit        | Graphiques clairs, interactifs                  |
| **HuggingFace Datasets**        | Accès aux jeux de données standardisés        | Pour fine-tuning ou testing                     |

---

## 📈 Prochaines extensions possibles

- Intégration API REST
- Mode batch / testing automatique
- Export des logs au format CSV / HTML
- Déploiement sur Hugging Face Spaces

---

## 🧪 Tests adverses (en option)

Ajoutez vos prompts « dangereux » dans `data/prompts.txt`, et faites une boucle pour évaluer la robustesse du système.

---

