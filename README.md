
# 🧠 Générateur de Texte + Filtrage Éthique

Une application Streamlit de génération de texte avec résumé, filtres éthiques automatiques, visualisation en temps réel, et fine-tuning personnalisé.

---

## 🔧 Fonctionnalités

1. **Génération de texte** (`distilgpt2`)
2. **Résumé automatique** (`distilbart-cnn-12-6`)
3. **Filtrage éthique automatique** (modèle `unitary/toxic-bert`)
4. **Calcul de perplexité**
5. **Visualisation de similarité prompt/résumé avec Plotly**
6. **Dashboard interactif**
7. **Fine-tuning sur IMDb (1%) avec `distilbert`** : métriques `accuracy` et `loss`

---

## ⚙️ Technologies utilisées

| Technologie         | Justification                                                                 |
|---------------------|--------------------------------------------------------------------------------|
| **Streamlit**       | Interface simple, rapide pour déploiement web                                 |
| **Transformers HF** | Accès aux modèles préentraînés d'État de l'art pour NLP                       |
| **Datasets HF**     | Accès instantané à des jeux de données publics (ex: IMDb)                     |
| **Plotly**          | Visualisations dynamiques dans la sidebar                                     |
| **PyTorch**         | Support léger du fine-tuning avec intégration via `Trainer`                   |

---

## 🚀 Lancer le projet

```bash
# 1. Cloner le repo
git clone <repo-url>
cd hackathon_2

# 2. Créer un environnement
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\activate sous Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app/main.py
```

---

## 📂 Structure du projet

```
hackathon_2/
│
├── app/
│   └── main.py              # Application principale Streamlit
│
├── README.md                # Explication du projet
└── requirements.txt         # Dépendances Python
```

---

## 📈 Améliorations futures

- Support multilingue
- Filtres éthiques plus fins (humour offensif, fake news, etc.)
- Téléchargement rapport PDF / CSV
