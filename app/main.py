import streamlit as st
import plotly.graph_objects as go
import torch
from sklearn.metrics import accuracy_score

# Placeholders pour fonctions manquantes

def generate_text(prompt):
    return f"{prompt}... this is a sample generated text."

def summarize_text(text):
    return text[:75] + "..."

def check_similarity(prompt, summary):
    return True, 0.85

def ethical_filter(text):
    return True

def is_hateful(text):
    return False

def contains_bias(text):
    return False

def grammar_ok(text):
    return True

def no_exaggeration(text):
    return True

# === Interface Streamlit ===
st.title("🧠 Générateur de Texte + Filtrage Éthique")

# Initialiser compteur par cause et statistiques globales
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "passed": 0,
        "rejected": 0,
        "similarities": [],
        "causes": {
            "mots-clés": 0,
            "haine": 0,
            "biais": 0,
            "grammaire": 0,
            "exagération": 0
        }
    }

prompt = st.text_input("Entrez un prompt :", value="This movie was")

if st.button("Générer"):
    with st.spinner("Génération en cours..."):
        generated = generate_text(prompt)
        summary = summarize_text(generated)
        is_relevant, sim_score = check_similarity(prompt, summary)

        failed = False
        if not ethical_filter(generated):
            st.session_state.stats["causes"]["mots-clés"] += 1
            failed = True
        if is_hateful(generated):
            st.session_state.stats["causes"]["haine"] += 1
            failed = True
        if contains_bias(generated):
            st.session_state.stats["causes"]["biais"] += 1
            failed = True
        if not grammar_ok(generated):
            st.session_state.stats["causes"]["grammaire"] += 1
            failed = True
        if not no_exaggeration(generated):
            st.session_state.stats["causes"]["exagération"] += 1
            failed = True

        passes_filters = not failed
        st.session_state.stats["total"] += 1
        st.session_state.stats["passed"] += int(passes_filters)
        st.session_state.stats["rejected"] += int(not passes_filters)
        st.session_state.stats["similarities"].append(sim_score)

    st.subheader("Texte généré")
    st.write(generated)

    st.subheader("Résumé")
    st.write(summary)

    st.subheader("Qualité & Filtres")
    st.write(f"🔎 Similarité prompt/résumé : {sim_score:.2f} ({'OK' if is_relevant else 'BAS'})")
    st.write(f"🛡️ Filtrage éthique : {'✅ Accepté' if passes_filters else '❌ Rejeté'}")

# === Dashboard temps réel ===
st.sidebar.header("📊 Statistiques en temps réel")
st.sidebar.write(f"Total générés : {st.session_state.stats['total']}")
st.sidebar.write(f"✅ Acceptés : {st.session_state.stats['passed']}")
st.sidebar.write(f"❌ Rejetés : {st.session_state.stats['rejected']}")

if st.session_state.stats['similarities']:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.stats['similarities'], mode='lines+markers', name='Similarité'))
    fig.update_layout(title='Évolution de la similarité (prompt vs résumé)', yaxis=dict(range=[0, 1]))
    st.sidebar.plotly_chart(fig, use_container_width=True)

    # Histogramme causes de rejet
    causes = st.session_state.stats["causes"]
    if sum(causes.values()) > 0:
        fig2 = go.Figure([go.Bar(x=list(causes.keys()), y=list(causes.values()))])
        fig2.update_layout(title="📉 Causes des rejets", xaxis_title="Filtre", yaxis_title="Nombre")
        st.sidebar.plotly_chart(fig2, use_container_width=True)

# === Étape 7 : Fine-tuning personnalisé ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}
