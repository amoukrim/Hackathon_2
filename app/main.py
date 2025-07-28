import streamlit as st
import plotly.graph_objects as go
import torch
import math
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline, GPT2LMHeadModel
from datasets import load_dataset
import evaluate

# === Générateur de texte réel ===
@st.cache_resource(show_spinner=False)
def get_generator():
    return pipeline("text-generation", model="distilgpt2")

def generate_text(prompt):
    generator = get_generator()
    output = generator(prompt, max_length=50, do_sample=True, top_k=30)[0]["generated_text"]
    return output

@st.cache_resource(show_spinner=False)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    summarizer = get_summarizer()
    summary = summarizer(text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
    return summary

@st.cache_resource(show_spinner=False)
def get_toxic_classifier():
    return pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

def automatic_filter(text):
    classifier = get_toxic_classifier()
    result = classifier(text)[0]
    toxic_labels = [label['label'] for label in result if label['score'] > 0.6 and label['label'].lower() != 'non-toxic']
    return len(toxic_labels) == 0, toxic_labels

def check_similarity(prompt, summary):
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=[summary], references=[prompt])
    rouge_score = scores["rougeL"].mid.fmeasure
    return rouge_score > 0.3, rouge_score

def compute_perplexity(texts):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()
    total_log_likelihood = 0
    total_length = 0
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            log_likelihood = outputs.loss.item() * input_ids.size(1)
        total_log_likelihood += log_likelihood
        total_length += input_ids.size(1)
    perplexity = math.exp(total_log_likelihood / total_length)
    return perplexity

# === Interface Streamlit ===
st.title("🧠 Générateur de Texte + Filtrage Éthique")

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "passed": 0,
        "rejected": 0,
        "similarities": [],
        "causes": {}
    }

prompt = st.text_input("Entrez un prompt :", value="This movie was")

if st.button("Générer"):
    with st.spinner("Génération en cours..."):
        generated = generate_text(prompt)
        summary = summarize_text(generated)
        is_relevant, sim_score = check_similarity(prompt, summary)

        passes_filters, reasons = automatic_filter(generated)

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
    if not passes_filters:
        st.write("**Raisons du rejet :**")
        for r in reasons:
            st.write(f"- {r}")

    with st.spinner("Calcul de perplexité..."):
        try:
            ppl = compute_perplexity([generated])
            st.write(f"🦩 Perplexité du texte généré : {ppl:.2f}")
        except Exception as e:
            st.warning(f"Erreur perplexité : {e}")

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

# === Étape 7 : Fine-tuning personnalisé ===
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = torch.argmax(torch.tensor(logits), dim=1)
#     acc = accuracy_score(labels, predictions)
#     return {"accuracy": acc}

# @st.cache_resource
# def fine_tune_on_imdb():
#     imdb = load_dataset("imdb", split="train[:1%]").train_test_split(test_size=0.2)
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#     model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

#     def preprocess(example):
#         return tokenizer(example["text"], padding="max_length", truncation=True)

#     tokenized = imdb.map(preprocess, batched=True)

#     args = TrainingArguments(
#         output_dir="results",
#         evaluation_strategy="epoch",
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=1,
#         logging_dir="logs",
#         disable_tqdm=True
#     )

#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=tokenized["train"],
#         eval_dataset=tokenized["test"],
#         compute_metrics=compute_metrics
#     )

#     trainer.train()
#     metrics = trainer.evaluate()
#     return metrics

# with st.expander("🔧 Fine-tuning personnalisé (IMDb)"):
#     if st.button("Lancer le fine-tuning"):
#         with st.spinner("Entraînement en cours (1% IMDb)..."):
#             results = fine_tune_on_imdb()
#         st.success("Fine-tuning terminé ✅")
#         st.write(f"🎯 Accuracy: {results['eval_accuracy']:.2f}")
#         st.write(f"📉 Loss: {results['eval_loss']:.4f}")
