#!/usr/bin/env python3
# cascade_router.py
# Router de cascada: Ligero (PhoBERT) → Pesado (mDeBERTa-v3)
# Correcciones aplicadas:
#   1. Umbrales se cargan desde router_thresholds.json (calibrado en val set)
#      en lugar de estar hardcodeados.
#   2. Monitoreo de was_forced con alerta configurable.
#   3. Ventana deslizante para detección de drift en tasa de forced.

import json
import logging
import numpy as np
from collections import deque
from typing import List, Dict, Tuple

import torch

from train_phobert_evidence import EvidenceAwareFactChecker, PhoBERTEvidenceTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class UncertaintyRouter:
    """
    Cascada con routing por incertidumbre.
    El modelo ligero resuelve lo trivial; lo dudoso va al pesado.

    Los umbrales se cargan desde `thresholds_path` (generado por calibrate_router
    en train_phobert_evidence.py) en lugar de valores hardcodeados.
    """

    LABEL_MAP = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}

    def __init__(
        self,
        light_model: EvidenceAwareFactChecker,
        heavy_model: torch.nn.Module,
        tokenizer_light: PhoBERTEvidenceTokenizer,
        tokenizer_heavy,
        thresholds_path: str = "phobert_evidence_checkpoints/router_thresholds.json",
        # Fallback si no existe el archivo de calibración
        confidence_threshold: float = 0.90,
        entropy_threshold: float = 0.45,
        certainty_threshold: float = 0.75,
        device: str = "cpu",
        # FIX #4: monitoreo de forced en ventana deslizante
        forced_alert_threshold: float = 0.25,
        monitoring_window: int = 200,
    ):
        self.light = light_model.to(device).eval()
        self.heavy = heavy_model.to(device).eval()
        self.tok_l = tokenizer_light
        self.tok_h = tokenizer_heavy
        self.device = device
        self.forced_alert_threshold = forced_alert_threshold

        # FIX #1: cargar umbrales calibrados si existen
        try:
            with open(thresholds_path) as f:
                t = json.load(f)
            self.conf_thresh     = float(t["conf"])
            self.entropy_thresh  = float(t["entropy"])
            self.certainty_thresh = float(t["cert"])
            logger.info(
                "Umbrales cargados desde %s → conf=%.2f, entropy=%.2f, cert=%.2f",
                thresholds_path, self.conf_thresh, self.entropy_thresh, self.certainty_thresh,
            )
        except (FileNotFoundError, KeyError):
            logger.warning(
                "No se encontró %s. Usando umbrales por defecto (conf=%.2f, entropy=%.2f, cert=%.2f). "
                "Ejecuta train_phobert_evidence.py con --quantize para generar el archivo.",
                thresholds_path, confidence_threshold, entropy_threshold, certainty_threshold,
            )
            self.conf_thresh      = confidence_threshold
            self.entropy_thresh   = entropy_threshold
            self.certainty_thresh = certainty_threshold

        # FIX #4: ventana deslizante para monitoreo de tasa de forced
        self._forced_window: deque = deque(maxlen=monitoring_window)
        self.stats = {"light": 0, "heavy": 0, "forced": 0, "total": 0}

    # ------------------------------------------------------------------
    # Métricas de incertidumbre
    # ------------------------------------------------------------------

    @staticmethod
    def _uncertainty_metrics(logits: torch.Tensor) -> Tuple[float, float, int]:
        probs = torch.softmax(logits, dim=-1)
        max_prob, pred_idx = torch.max(probs, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1) / np.log(logits.size(-1))
        return max_prob.item(), entropy.item(), pred_idx.item()

    # ------------------------------------------------------------------
    # Predicción principal
    # ------------------------------------------------------------------

    def predict(self, claim: str, contexts: List[str]) -> Dict:
        self.stats["total"] += 1

        # ---- ETAPA 1: LIGERO ----
        enc = self.tok_l.encode(claim, contexts)
        input_ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits_l, certainty, attn_probs, _ = self.light(
                input_ids, mask, enc["evidence_spans"]
            )

        conf, ent, pred = self._uncertainty_metrics(logits_l)
        light_label = self.LABEL_MAP[pred]

        # ---- DECISIÓN DEL ROUTER ----
        route_to_heavy = (
            conf < self.conf_thresh
            or ent > self.entropy_thresh
            or light_label == "NEI"
            or certainty.item() < self.certainty_thresh
        )

        if not route_to_heavy:
            self.stats["light"] += 1
            self._record_forced(False)
            return {
                "predicted_label": light_label,
                "confidence": round(conf, 4),
                "was_forced": False,
                "tier": "light",
                "routing_reason": "high_confidence",
                "certainty_not_nei": round(certainty.item(), 4),
                "evidence_attention": attn_probs[0, : len(contexts), 0].cpu().tolist(),
            }

        # ---- ETAPA 2: PESADO ----
        self.stats["heavy"] += 1

        evidence_text = " </s> ".join(contexts)
        text_heavy = f"{claim} </s> {evidence_text}"

        enc_h = self.tok_h(
            text_heavy,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        enc_h = {k: v.to(self.device) for k, v in enc_h.items()}

        with torch.no_grad():
            logits_h = self.heavy(**enc_h).logits

        conf_h, _, pred_h = self._uncertainty_metrics(logits_h)
        heavy_label = self.LABEL_MAP[pred_h]

        # Forzar binario si el pesado también duda (NEI)
        was_forced = False
        if heavy_label == "NEI":
            probs_h = torch.softmax(logits_h, dim=-1)[0]
            binary = probs_h[:2] / (probs_h[:2].sum() + 1e-10)
            forced_idx = torch.argmax(binary).item()
            heavy_label = self.LABEL_MAP[forced_idx]
            conf_h = binary[forced_idx].item()
            was_forced = True
            self.stats["forced"] += 1

        self._record_forced(was_forced)

        return {
            "predicted_label": heavy_label,
            "confidence": round(conf_h, 4),
            "was_forced": was_forced,
            "tier": "heavy",
            "routing_reason": (
                f"light_uncertain(conf={conf:.2f}, ent={ent:.2f}, label={light_label})"
            ),
            "light_fallback_label": light_label,
        }

    # ------------------------------------------------------------------
    # FIX #4: Monitoreo de forced en ventana deslizante
    # ------------------------------------------------------------------

    def _record_forced(self, was_forced: bool):
        self._forced_window.append(int(was_forced))
        if len(self._forced_window) == self._forced_window.maxlen:
            rate = sum(self._forced_window) / len(self._forced_window)
            if rate > self.forced_alert_threshold:
                logger.warning(
                    "ALERTA DE DRIFT: tasa de was_forced=%.1f%% en las últimas %d solicitudes "
                    "(umbral=%.1f%%). Revisar distribución de datos en producción.",
                    rate * 100,
                    self._forced_window.maxlen,
                    self.forced_alert_threshold * 100,
                )

    def get_stats(self) -> Dict:
        total = max(self.stats["total"], 1)
        return {
            **self.stats,
            "light_rate": round(self.stats["light"] / total, 3),
            "heavy_rate": round(self.stats["heavy"] / total, 3),
            "forced_rate": round(self.stats["forced"] / total, 3),
        }