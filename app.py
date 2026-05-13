# app.py
import json, torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from train_phobert_evidence import EvidenceAwareFactChecker, PhoBERTEvidenceTokenizer
from cascade_router import UncertaintyRouter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# --- Cargar modelos al arrancar ---
device = "cuda" if torch.cuda.is_available() else "cpu"

tok_light = PhoBERTEvidenceTokenizer("vinai/phobert-base")
light = EvidenceAwareFactChecker()
ckpt = torch.load("phobert_evidence_checkpoints/best_model.pt", map_location=device)
light.load_state_dict(ckpt["model_state_dict"])

tok_heavy = AutoTokenizer.from_pretrained("./mdeberta_factcheck")
heavy = AutoModelForSequenceClassification.from_pretrained("./mdeberta_factcheck")

router = UncertaintyRouter(
    light_model=light,
    heavy_model=heavy,
    tokenizer_light=tok_light,
    tokenizer_heavy=tok_heavy,
    thresholds_path="phobert_evidence_checkpoints/router_thresholds.json",
    device=device,
)

# --- Schema del input ---
class PredictRequest(BaseModel):
    claim: str
    contexts: List[str] = []

# --- Endpoint ---
@app.post("/predict")
def predict(req: PredictRequest):
    result = router.predict(req.claim, req.contexts)
    # Output exacto que pide el torneo
    return {"predicted_label": result["predicted_label"]}

@app.get("/health")
def health():
    return {"status": "ok"}