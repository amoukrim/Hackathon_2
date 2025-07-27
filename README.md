# 🧠 Générateur de Texte + Filtrage Éthique

Cette application Streamlit génère des textes, les résume automatiquement, applique un filtrage éthique et permet un fine-tuning sur IMDb.

## 🚀 Fonctionnalités
- Génération de texte avec `distilgpt2`
- Résumé automatique avec `distilbart-cnn-12-6`
- Filtrage automatique avec `unitary/toxic-bert`
- Mesure de similarité avec ROUGE
- Calcul de perplexité
- Dashboard en temps réel
- 🔧 Fine-tuning sur IMDb (1%) avec suivi accuracy / loss

## 🧰 Technologies
- **Streamlit** : Interface web rapide
- **HuggingFace Transformers / Datasets** : Modèles et données modernes
- **Plotly** : Dashboard interactif
- **PyTorch** : Fine-tuning léger
- ✅ Compatible CPU

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run main.py
```

## 📁 Structure
- `main.py` : App principale
- `requirements.txt` : Dépendances
- `README.md` : Documentation
