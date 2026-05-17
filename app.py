# app.py
# FastAPI para torneo de fact-checking vietnamita.
# Arquitectura:
#   Modelo ligero: PhoBERT-base
#   Modelo pesado: PhoBERT-base-v2
#   Salida final: {"predicted_label": "SUPPORTED" | "REFUTED"}

import os
import logging
from typing import List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from train_phobert_evidence import EvidenceAwareFactChecker, PhoBERTEvidenceTokenizer
from cascade_router import UncertaintyRouter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Vietnamese Fact Checking API")

# En Azure normalmente correrá en CPU, pero se mantiene CUDA si existe.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carpetas esperadas dentro del proyecto / contenedor Docker.
LIGHT_DIR = "phobert_evidence_checkpoints"
HEAVY_DIR = "phobert_v2_evidence_checkpoints"

LIGHT_MODEL_NAME = "vinai/phobert-base"
HEAVY_MODEL_NAME = "vinai/phobert-base-v2"

router = None


class PredictRequest(BaseModel):
    claim: str
    contexts: List[str] = []


def _build_evidence_model(model_name: str) -> EvidenceAwareFactChecker:
    """
    Crea EvidenceAwareFactChecker usando la firma correcta de tu clase.
    """
    return EvidenceAwareFactChecker(
        phobert_name=model_name,
        num_evidence_attn_heads=4,
        num_labels=3
    )


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Acepta checkpoints guardados como:
    - {"model_state_dict": ...}
    - state_dict directo
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def load_phobert_checkpoint(model_name: str, checkpoint_dir: str):
    """
    Carga tokenizer + modelo evidence-aware desde una carpeta con best_model.pt.
    """
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")

    tokenizer = PhoBERTEvidenceTokenizer(model_name)
    model = _build_evidence_model(model_name)

    # Necesario porque tu tokenizer personalizado agrega tokens especiales.
    # Evita errores CUDA/embedding out of bounds.
    if hasattr(model, "encoder") and hasattr(model.encoder, "resize_token_embeddings"):
        model.encoder.resize_token_embeddings(len(tokenizer.tokenizer))
        logger.info(
            "Resize embeddings aplicado para %s: %d tokens",
            model_name,
            len(tokenizer.tokenizer),
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Modelo cargado: %s desde %s", model_name, checkpoint_path)

    return model, tokenizer


@app.on_event("startup")
def startup_event():
    """
    Carga los dos modelos una vez al iniciar la API.
    """
    global router

    try:
        light_model, light_tokenizer = load_phobert_checkpoint(
            model_name=LIGHT_MODEL_NAME,
            checkpoint_dir=LIGHT_DIR,
        )

        heavy_model, heavy_tokenizer = load_phobert_checkpoint(
            model_name=HEAVY_MODEL_NAME,
            checkpoint_dir=HEAVY_DIR,
        )

        router = UncertaintyRouter(
            light_model=light_model,
            heavy_model=heavy_model,
            tokenizer_light=light_tokenizer,
            tokenizer_heavy=heavy_tokenizer,
            light_thresholds_path=os.path.join(LIGHT_DIR, "router_thresholds.json"),
            heavy_thresholds_path=os.path.join(HEAVY_DIR, "router_thresholds.json"),
            # Umbrales manuales razonables si el JSON calibrado es demasiado estricto.
            confidence_threshold=0.65,
            entropy_threshold=0.70,
            certainty_threshold=0.50,
            margin_threshold=0.15,
            device=device,
        )

        logger.info("Router PhoBERT cascade cargado correctamente en %s", device)

    except Exception as e:
        logger.exception("Error cargando modelos/router: %s", e)
        router = None


@app.get("/")
def root():
    return {
        "service": "Vietnamese Fact Checking API",
        "status": "ok",
        "device": device,
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if router is not None else "error",
        "router_loaded": router is not None,
        "device": device,
    }


@app.get("/stats")
def stats():
    if router is None:
        raise HTTPException(status_code=503, detail="Router no cargado")
    return router.get_stats()


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint final del torneo.

    IMPORTANTE:
    La salida se mantiene exactamente en el formato requerido:
        {"predicted_label": "SUPPORTED" | "REFUTED"}

    Nunca devuelve NEI aunque internamente los modelos lo usen.
    """
    if router is None:
        raise HTTPException(status_code=503, detail="Router no cargado")

    result = router.predict(req.claim, req.contexts)

    label = result["predicted_label"]

    # Safety check para asegurar cumplimiento del formato del torneo.
    if label not in {"SUPPORTED", "REFUTED"}:
        label = "SUPPORTED"

    return {"predicted_label": label}
