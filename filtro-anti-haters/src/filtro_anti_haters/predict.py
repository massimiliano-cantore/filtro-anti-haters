"""Inferenza con il modello DistilBERT addestrato (notebook 03).

Uso:
    from filtro_anti_haters.predict import ToxicityClassifier
    clf = ToxicityClassifier("percorso/o/utente-hf/filtro-anti-haters-distilbert")
    clf.predict(["You are an idiot", "Thanks for the edit!"])
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


@dataclass
class Prediction:
    text: str
    probabilities: dict[str, float]
    labels: list[str]

    @property
    def is_toxic(self) -> bool:
        return bool(self.labels)


class ToxicityClassifier:
    def __init__(self, model_path: str, max_len: int = 128, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.max_len = max_len
        self.thresholds = self._load_thresholds(model_path)

    @staticmethod
    def _load_thresholds(model_path: str) -> dict[str, float]:
        """Soglie per classe scelte sul validation set; 0.5 se il file non è disponibile."""
        local = os.path.join(model_path, "thresholds.json")
        try:
            if os.path.exists(local):
                path = local
            else:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(model_path, "thresholds.json")
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {label: 0.5 for label in LABELS}

    @torch.no_grad()
    def predict(self, texts: str | list[str]) -> list[Prediction]:
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_len, padding=True,
                             return_tensors="pt").to(self.device)
        proba = torch.sigmoid(self.model(**enc).logits.float()).cpu().numpy()
        out = []
        for text, row in zip(texts, proba):
            probs = {label: float(p) for label, p in zip(LABELS, row)}
            labels = [label for label in LABELS if probs[label] >= self.thresholds.get(label, 0.5)]
            out.append(Prediction(text, probs, labels))
        return out
