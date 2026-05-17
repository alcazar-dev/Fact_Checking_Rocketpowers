#!/usr/bin/env python3
# cascade_router.py
# Router de cascada: Ligero (PhoBERT-base) → Pesado (PhoBERT-base-v2)
#
# Cambio principal:
#   - Se elimina la dependencia de mDeBERTa.
#   - Ambos niveles usan EvidenceAwareFactChecker + PhoBERTEvidenceTokenizer.
#   - La salida final se fuerza a formato binario: SUPPORTED | REFUTED.
#
# Uso esperado:
#   light_model  = PhoBERT-base fine-tuned
#   heavy_model  = PhoBERT-base-v2 fine-tuned
#   tokenizer_light = PhoBERTEvidenceTokenizer("vinai/phobert-base")
#   tokenizer_heavy = PhoBERTEvidenceTokenizer("vinai/phobert-base-v2")

import json
import logging
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

import torch

from train_phobert_evidence import EvidenceAwareFactChecker, PhoBERTEvidenceTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class PhoBERTCascadeRouter:
    """
    Router de cascada para fact-checking vietnamita.

    Arquitectura:
        1. Modelo ligero: PhoBERT-base
        2. Si el modelo ligero duda: PhoBERT-base-v2
        3. Salida final SIEMPRE binaria:
           {"predicted_label": "SUPPORTED" | "REFUTED"}

    Nota:
        Internamente los modelos pueden producir 3 clases:
        0 = SUPPORTED
        1 = REFUTED
        2 = NEI

        Pero el endpoint final del torneo solo acepta SUPPORTED o REFUTED.
        Por eso, cuando el resultado interno es NEI, se fuerza la salida
        comparando las probabilidades de SUPPORTED y REFUTED.
    """

    LABEL_MAP = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}

    def __init__(
        self,
        light_model: EvidenceAwareFactChecker,
        heavy_model: EvidenceAwareFactChecker,
        tokenizer_light: PhoBERTEvidenceTokenizer,
        tokenizer_heavy: PhoBERTEvidenceTokenizer,
        light_thresholds_path: Optional[str] = "phobert_evidence_checkpoints/router_thresholds.json",
        heavy_thresholds_path: Optional[str] = "phobert_v2_evidence_checkpoints/router_thresholds.json",
        confidence_threshold: float = 0.65,
        entropy_threshold: float = 0.70,
        certainty_threshold: float = 0.50,
        margin_threshold: float = 0.15,
        device: str = "cpu",
        forced_alert_threshold: float = 0.25,
        monitoring_window: int = 200,
    ):
        self.device = device

        self.light = light_model.to(device).eval()
        self.heavy = heavy_model.to(device).eval()

        self.tok_l = tokenizer_light
        self.tok_h = tokenizer_heavy

        self.forced_alert_threshold = forced_alert_threshold
        self.margin_threshold = margin_threshold

        # Cargar umbrales del modelo ligero si existen.
        # Si los umbrales calibrados quedaron demasiado estrictos, puedes ajustar
        # manualmente confidence_threshold, entropy_threshold y certainty_threshold.
        self.conf_thresh = confidence_threshold
        self.entropy_thresh = entropy_threshold
        self.certainty_thresh = certainty_threshold

        if light_thresholds_path:
            self._load_thresholds_if_available(light_thresholds_path)

        # El heavy_thresholds_path queda guardado solo para trazabilidad/logs.
        # No se usa para decidir si responder, porque el heavy es la última etapa.
        self.heavy_thresholds_path = heavy_thresholds_path

        self._forced_window: deque = deque(maxlen=monitoring_window)
        self.stats = {
            "light": 0,
            "heavy": 0,
            "forced": 0,
            "total": 0,
        }

    # ------------------------------------------------------------------
    # Carga de umbrales
    # ------------------------------------------------------------------

    def _load_thresholds_if_available(self, thresholds_path: str) -> None:
        try:
            with open(thresholds_path, "r", encoding="utf-8") as f:
                t = json.load(f)

            self.conf_thresh = float(t.get("conf", self.conf_thresh))
            self.entropy_thresh = float(t.get("entropy", self.entropy_thresh))
            self.certainty_thresh = float(t.get("cert", self.certainty_thresh))

            logger.info(
                "Umbrales del modelo ligero cargados desde %s → conf=%.2f, entropy=%.2f, cert=%.2f, margin=%.2f",
                thresholds_path,
                self.conf_thresh,
                self.entropy_thresh,
                self.certainty_thresh,
                self.margin_threshold,
            )

        except FileNotFoundError:
            logger.warning(
                "No se encontró %s. Usando umbrales manuales → conf=%.2f, entropy=%.2f, cert=%.2f, margin=%.2f",
                thresholds_path,
                self.conf_thresh,
                self.entropy_thresh,
                self.certainty_thresh,
                self.margin_threshold,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(
                "No se pudieron leer correctamente los umbrales de %s (%s). Usando umbrales manuales.",
                thresholds_path,
                e,
            )

    # ------------------------------------------------------------------
    # Métricas y utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _uncertainty_metrics(logits: torch.Tensor) -> Tuple[torch.Tensor, float, float, int]:
        """
        Regresa:
            probs: tensor de probabilidades [num_labels]
            max_prob: confianza máxima
            entropy: entropía normalizada
            pred_idx: índice predicho
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs = torch.softmax(logits, dim=-1)[0]
        max_prob, pred_idx = torch.max(probs, dim=-1)

        entropy = -(probs * torch.log(probs + 1e-10)).sum() / np.log(probs.numel())

        return probs, max_prob.item(), entropy.item(), pred_idx.item()

    @staticmethod
    def _ensure_batch_tensor(x: torch.Tensor, device: str) -> torch.Tensor:
        """
        El tokenizer personalizado suele devolver tensores 1D: [seq_len].
        El modelo espera batch: [batch_size, seq_len].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x.to(device)

    @staticmethod
    def _ensure_batched_spans(spans):
        """
        El forward de EvidenceAwareFactChecker suele esperar lista por batch:
            [[(start, end), ...]]
        Si llega [(start, end), ...], lo convertimos a batch de 1.
        """
        if spans is None:
            return [[]]

        if isinstance(spans, list):
            if len(spans) == 0:
                return [[]]

            # Caso: [(43, 200), (201, 255)]
            if isinstance(spans[0], tuple):
                return [spans]

            # Caso: [[(43, 200), (201, 255)]]
            if isinstance(spans[0], list):
                return spans

        return [spans]

    @staticmethod
    def _force_binary_from_probs(probs: torch.Tensor) -> Tuple[str, float]:
        """
        Convierte probabilidades de 3 clases a salida binaria.
        Solo compara SUPPORTED vs REFUTED.
        """
        supported_prob = probs[0].item()
        refuted_prob = probs[1].item()

        denom = supported_prob + refuted_prob + 1e-10
        supported_bin = supported_prob / denom
        refuted_bin = refuted_prob / denom

        if supported_bin >= refuted_bin:
            return "SUPPORTED", supported_bin
        return "REFUTED", refuted_bin

    # ------------------------------------------------------------------
    # Predicción con un modelo PhoBERT evidence-aware
    # ------------------------------------------------------------------

    def _predict_with_phobert(
        self,
        model: EvidenceAwareFactChecker,
        tokenizer: PhoBERTEvidenceTokenizer,
        claim: str,
        contexts: List[str],
    ) -> Dict:
        enc = tokenizer.encode(claim, contexts)

        input_ids = self._ensure_batch_tensor(enc["input_ids"], self.device)
        attention_mask = self._ensure_batch_tensor(enc["attention_mask"], self.device)
        evidence_spans = self._ensure_batched_spans(enc.get("evidence_spans", []))

        with torch.no_grad():
            logits, certainty, attn_probs, _ = model(
                input_ids,
                attention_mask,
                evidence_spans,
            )

        probs, conf, entropy, pred_idx = self._uncertainty_metrics(logits)
        raw_label = self.LABEL_MAP[pred_idx]
        margin = abs(probs[0].item() - probs[1].item())

        return {
            "probs": probs,
            "pred_idx": pred_idx,
            "raw_label": raw_label,
            "confidence": conf,
            "entropy": entropy,
            "margin": margin,
            "certainty": float(certainty.item()) if torch.is_tensor(certainty) else float(certainty),
            "attn_probs": attn_probs,
        }

    # ------------------------------------------------------------------
    # Predicción principal
    # ------------------------------------------------------------------

    def predict(self, claim: str, contexts: List[str]) -> Dict:
        self.stats["total"] += 1

        # ---- ETAPA 1: MODELO LIGERO ----
        light_result = self._predict_with_phobert(
            self.light,
            self.tok_l,
            claim,
            contexts,
        )

        light_label = light_result["raw_label"]
        light_conf = light_result["confidence"]
        light_entropy = light_result["entropy"]
        light_certainty = light_result["certainty"]
        light_margin = light_result["margin"]

        route_to_heavy = (
            light_label == "NEI"
            or light_conf < self.conf_thresh
            or light_entropy > self.entropy_thresh
            or light_certainty < self.certainty_thresh
            or light_margin < self.margin_threshold
        )

        # Si el modelo ligero no duda y ya predijo SUPPORTED/REFUTED, respondemos.
        if not route_to_heavy:
            self.stats["light"] += 1
            self._record_forced(False)

            final_label, final_conf = self._force_binary_from_probs(light_result["probs"])

            return {
                "predicted_label": final_label,
                "confidence": round(final_conf, 4),
                "tier": "light",
                "was_routed": False,
                "was_forced": False,
                "light_raw_label": light_label,
                "routing_reason": "light_high_confidence",
            }

        # ---- ETAPA 2: MODELO PESADO PhoBERT-v2 ----
        self.stats["heavy"] += 1

        heavy_result = self._predict_with_phobert(
            self.heavy,
            self.tok_h,
            claim,
            contexts,
        )

        heavy_raw_label = heavy_result["raw_label"]
        heavy_probs = heavy_result["probs"]

        final_label, final_conf = self._force_binary_from_probs(heavy_probs)
        was_forced = heavy_raw_label == "NEI"

        if was_forced:
            self.stats["forced"] += 1

        self._record_forced(was_forced)

        return {
            "predicted_label": final_label,
            "confidence": round(final_conf, 4),
            "tier": "heavy",
            "was_routed": True,
            "was_forced": was_forced,
            "light_raw_label": light_label,
            "heavy_raw_label": heavy_raw_label,
            "routing_reason": (
                f"light_uncertain("
                f"conf={light_conf:.2f}, "
                f"entropy={light_entropy:.2f}, "
                f"certainty={light_certainty:.2f}, "
                f"margin={light_margin:.2f}, "
                f"label={light_label})"
            ),
        }

    # ------------------------------------------------------------------
    # Monitoreo
    # ------------------------------------------------------------------

    def _record_forced(self, was_forced: bool) -> None:
        self._forced_window.append(int(was_forced))

        if len(self._forced_window) == self._forced_window.maxlen:
            rate = sum(self._forced_window) / len(self._forced_window)
            if rate > self.forced_alert_threshold:
                logger.warning(
                    "ALERTA DE DRIFT: tasa de forced=%.1f%% en las últimas %d solicitudes "
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


# Alias opcional para no romper imports antiguos:
# Si tu app.py todavía importa `UncertaintyRouter`, seguirá funcionando.
UncertaintyRouter = PhoBERTCascadeRouter
