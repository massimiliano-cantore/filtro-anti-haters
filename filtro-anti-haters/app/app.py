"""Demo Gradio del filtro anti-haters (DistilBERT) per Hugging Face Spaces."""
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from filtro_anti_haters.predict import LABELS, ToxicityClassifier
except ImportError:  # su Spaces predict.py è copiato accanto ad app.py
    from predict import LABELS, ToxicityClassifier

MODEL_ID = os.getenv("MODEL_ID", "UTENTE_HF/filtro-anti-haters-distilbert")
clf = ToxicityClassifier(MODEL_ID)

LABELS_IT = {
    "toxic": "Tossico",
    "severe_toxic": "Gravemente tossico",
    "obscene": "Osceno",
    "threat": "Minaccia",
    "insult": "Insulto",
    "identity_hate": "Odio identitario",
}

EXAMPLES = [
    "Thanks for fixing the references, the article reads much better now.",
    "I don't find this offensive at all.",
    "You are an idiot and nobody wants your edits here.",
    "If you revert my edit again I will find you.",
    "This is not stupid, it is actually a good idea.",
]


def classify(text: str):
    text = (text or "").strip()
    if not text:
        return {}, "Scrivi un commento (in inglese) da analizzare."
    pred = clf.predict(text)[0]
    scores = {LABELS_IT[k]: v for k, v in pred.probabilities.items()}
    if pred.is_toxic:
        verdict = "🚫 **Commento da moderare** — " + ", ".join(LABELS_IT[l] for l in pred.labels)
    else:
        verdict = "✅ **Commento accettabile**"
    return scores, verdict


with gr.Blocks(title="Filtro anti-haters") as demo:
    gr.Markdown(
        "# Filtro anti-haters\n"
        "Classificazione multi-label di commenti tossici con **DistilBERT** fine-tuned sul dataset "
        "Jigsaw/Wikipedia (160k commenti, in inglese). Per ogni categoria il modello restituisce una "
        "probabilità; la decisione usa soglie ottimizzate per classe sul validation set."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Commento", lines=4, placeholder="Scrivi un commento in inglese…")
            btn = gr.Button("Analizza", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            verdict = gr.Markdown()
            scores = gr.Label(label="Probabilità per categoria", num_top_classes=len(LABELS))
    btn.click(classify, inp, [scores, verdict])
    inp.submit(classify, inp, [scores, verdict])
    gr.Markdown(
        "Test set: ROC AUC 0.992 · F1 micro 0.790 · F1 macro 0.683. "
        "Il modello può sbagliare: è una demo, non uno strumento di moderazione in produzione."
    )

if __name__ == "__main__":
    demo.launch()
