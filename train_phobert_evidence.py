#!/usr/bin/env python3
# train_phobert_evidence.py
# Entrenamiento completo: PhoBERT + Atención por Evidencia + Loss de Certeza
# Correcciones aplicadas:
#   1. attn_weights tiene shape (B, E, E); se hace mean(dim=-1) sobre keys,
#      no sobre heads. Se usa average_attn_weights=True (default) para que
#      PyTorch ya promedia las heads → shape (B, E, E). La reducción final
#      sobre la dimensión de keys (dim=2) es la correcta.
#   2. Spans vacíos/truncados se loguean en lugar de descartarse silenciosamente.
#   3. Router calibrado empíricamente desde val set en lugar de umbrales fijos.
#   4. Monitoreo de was_forced y alertas por umbral configurable.
#   5. Épocas reducidas a 7 (con patience=3 sigue haciendo early stopping).
#   6. Cuantización INT8 del ligero antes de guardar.

import os
import json
import logging
import argparse
import warnings
import numpy as np
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. MODELO: PhoBERT + Self-Attention sobre evidencias
# ============================================================

class EvidenceAwareFactChecker(nn.Module):
    def __init__(
        self,
        phobert_name: str = "vinai/phobert-base",
        num_evidence_attn_heads: int = 4,
        dropout: float = 0.15,
        num_labels: int = 3
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(phobert_name)
        self.hidden_size = self.encoder.config.hidden_size

        # average_attn_weights=True (default PyTorch ≥1.9) → shape (B, E, E)
        self.evidence_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_evidence_attn_heads,
            batch_first=True,
            dropout=dropout / 2,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 1.5),
            nn.Linear(self.hidden_size // 2, num_labels),
        )

        # 1.0 = seguro de NO ser NEI | 0.0 = probable NEI
        self.certainty_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        evidence_spans: List[List[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, H)
        cls_vec = hidden_states[:, 0, :]           # (B, H)

        # --- Extraer vectores de evidencia con logging de spans truncados ---
        batch_evidence_vecs = []
        max_evidence = 0
        for b in range(hidden_states.size(0)):
            spans = evidence_spans[b]
            vecs = []
            for (start, end) in spans:
                if start >= end:
                    logger.debug("Span vacío descartado (start=%d, end=%d) en muestra %d", start, end, b)
                    continue
                if end > hidden_states.size(1):
                    # FIX #2: loguear truncamiento en lugar de ignorar silenciosamente
                    logger.debug(
                        "Span truncado por max_length: end=%d > seq_len=%d. Ajustando.",
                        end, hidden_states.size(1)
                    )
                    end = hidden_states.size(1)
                if start >= end:
                    continue
                vecs.append(hidden_states[b, start:end, :].mean(dim=0))

            if len(vecs) == 0:
                logger.debug("Sin evidencias válidas en muestra %d; usando vector cero.", b)
                vecs = [torch.zeros(self.hidden_size, device=hidden_states.device)]

            batch_evidence_vecs.append(torch.stack(vecs))
            max_evidence = max(max_evidence, len(vecs))

        padded_evidence = []
        evidence_masks = []
        for vecs in batch_evidence_vecs:
            num_e = vecs.size(0)
            if num_e < max_evidence:
                pad = torch.zeros(max_evidence - num_e, self.hidden_size, device=vecs.device)
                vecs = torch.cat([vecs, pad], dim=0)
                mask = [False] * num_e + [True] * (max_evidence - num_e)
            else:
                mask = [False] * num_e
            padded_evidence.append(vecs)
            evidence_masks.append(mask)

        evidence_tensor = torch.stack(padded_evidence)              # (B, E, H)
        key_padding_mask = torch.tensor(evidence_masks, device=hidden_states.device)

        # FIX #1: average_attn_weights=True (default) → attn_weights shape (B, E, E)
        # La dimensión -1 es la de las KEYS (sobre qué evidencias atiende cada query).
        # mean(dim=-1) promedia la atención recibida por cada evidencia → (B, E).
        attended_evidence, attn_weights = self.evidence_attention(
            query=evidence_tensor,
            key=evidence_tensor,
            value=evidence_tensor,
            key_padding_mask=key_padding_mask,
            average_attn_weights=True,   # Explícito para claridad
        )
        # attn_weights: (B, E, E)  →  mean sobre dim=2 (keys) → (B, E)
        attn_scores = attn_weights.mean(dim=2)
        attn_scores = attn_scores.masked_fill(key_padding_mask, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (B, E, 1)
        evidence_pooled = (attended_evidence * attn_probs).sum(dim=1)  # (B, H)

        combined = torch.cat([cls_vec, evidence_pooled], dim=-1)  # (B, 2H)
        logits = self.classifier(combined)                         # (B, 3)
        certainty = self.certainty_head(combined)                  # (B, 1)

        return logits, certainty, attn_probs, evidence_pooled


# ============================================================
# 2. TOKENIZER con marcadores <unused0> por evidencia
# ============================================================

class PhoBERTEvidenceTokenizer:
    def __init__(self, model_name: str = "vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.evidence_token = "<unused0>"
        if self.evidence_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.evidence_token])

    def encode(self, claim: str, contexts: List[str], max_length: int = 512) -> Dict:
        claim_tokens = self.tokenizer.encode(claim, add_special_tokens=False)
        all_tokens = [self.tokenizer.cls_token_id] + claim_tokens + [self.tokenizer.sep_token_id]
        spans = []

        for ctx in contexts:
            marker = self.tokenizer.encode(self.evidence_token, add_special_tokens=False)
            ctx_tokens = self.tokenizer.encode(ctx, add_special_tokens=False)
            start = len(all_tokens) + len(marker)
            end = start + len(ctx_tokens)
            spans.append((start, end))
            all_tokens.extend(marker + ctx_tokens)

        if len(all_tokens) > max_length - 1:
            all_tokens = all_tokens[: max_length - 1]
            # FIX #2: filtrar spans y recalcular end; descartar spans que quedaron vacíos
            adjusted = []
            for s, e in spans:
                if s >= max_length - 1:
                    break  # Este y los siguientes cayeron fuera
                e_clipped = min(e, max_length - 1)
                if e_clipped > s:
                    adjusted.append((s, e_clipped))
                else:
                    logger.debug("Span (%d, %d) colapsó tras truncamiento; descartado.", s, e)
            spans = adjusted
            all_tokens += [self.tokenizer.sep_token_id]
        else:
            all_tokens += [self.tokenizer.sep_token_id]

        seq_len = len(all_tokens)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)
        all_tokens += [self.tokenizer.pad_token_id] * (max_length - seq_len)

        return {
            "input_ids": torch.tensor([all_tokens]),
            "attention_mask": torch.tensor([attention_mask]),
            "evidence_spans": [spans],
        }


# ============================================================
# 3. DATASET & COLLATE
# ============================================================

class VietnameseFactDataset(Dataset):
    # Rutas de datos confirmadas del proyecto
    DEFAULT_TRAIN = "HIGH_CONFIDENCE_train_dataset.json"
    DEFAULT_VAL   = "HIGH_CONFIDENCE_validation_dataset.json"
    DEFAULT_TEST  = "HIGH_CONFIDENCE_test_dataset.json"

    def __init__(self, json_path: str, tokenizer: PhoBERTEvidenceTokenizer):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.label_map = {"SUPPORTED": 0, "REFUTED": 1, "NEI": 2}
        logger.info("Dataset cargado: %s (%d registros)", json_path, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item["claim"]
        contexts = item.get("contexts", [])
        label = self.label_map.get(item["label"], 2)

        enc = self.tokenizer.encode(claim, contexts)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "evidence_spans": enc["evidence_spans"][0],
            "label": torch.tensor(label, dtype=torch.long),
            "num_contexts": len(contexts),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "evidence_spans": [b["evidence_spans"] for b in batch],
        "labels": torch.stack([b["label"] for b in batch]),
        "num_contexts": [b["num_contexts"] for b in batch],
    }


# ============================================================
# 4. MÉTRICAS
# ============================================================

def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


# ============================================================
# 5. EVALUACIÓN
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, lambda_cert=0.3, desc="Valid"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    forced_count = 0
    ce_loss_fn = nn.CrossEntropyLoss()
    bce_loss_fn = nn.BCELoss()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, certainty, _, _ = model(input_ids, attention_mask, batch["evidence_spans"])

        loss_cls = ce_loss_fn(logits, labels)
        cert_target = (labels != 2).float().unsqueeze(1)
        loss_cert = bce_loss_fn(certainty, cert_target)
        total_loss += (loss_cls + lambda_cert * loss_cert).item()

        preds = torch.argmax(logits, dim=-1)
        forced_count += (preds == 2).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)
    metrics["forced_rate"] = forced_count / max(len(all_preds), 1)

    logger.info("\n--- %s ---", desc)
    logger.info(
        "Loss: %.4f | Acc: %.4f | F1-macro: %.4f | NEI_pred_rate: %.2f%%",
        metrics["loss"], metrics["accuracy"], metrics["f1_macro"],
        metrics["forced_rate"] * 100,
    )
    print(classification_report(
        all_labels, all_preds,
        target_names=["SUPPORTED", "REFUTED", "NEI"],
        digits=4, zero_division=0,
    ))
    return metrics


# ============================================================
# 6. CALIBRACIÓN DE UMBRALES DEL ROUTER
#    Busca (conf_thresh, entropy_thresh, certainty_thresh) que maximicen
#    cobertura del modelo ligero manteniendo F1 ≥ target_f1 en val set.
# ============================================================

@torch.no_grad()
def calibrate_router(model, dataloader, device, target_f1: float = 0.85):
    """
    Retorna dict con los mejores umbrales calibrados sobre el val set.
    """
    model.eval()
    all_confs, all_entropies, all_certs, all_labels, all_logits = [], [], [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, certainty, _, _ = model(input_ids, attention_mask, batch["evidence_spans"])
        probs = torch.softmax(logits, dim=-1)
        max_conf, _ = probs.max(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1) / np.log(logits.size(-1))

        all_confs.extend(max_conf.cpu().numpy())
        all_entropies.extend(entropy.cpu().numpy())
        all_certs.extend(certainty.squeeze(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(probs.cpu().numpy())

    all_confs = np.array(all_confs)
    all_entropies = np.array(all_entropies)
    all_certs = np.array(all_certs)
    all_labels = np.array(all_labels)
    all_logits = np.concatenate(all_logits, axis=0)

    preds = all_logits.argmax(axis=1)
    best = {"coverage": 0.0, "conf": 0.80, "entropy": 0.50, "cert": 0.65}

    # Grid search liviano sobre los tres umbrales
    for conf_t in np.arange(0.75, 0.97, 0.05):
        for ent_t in np.arange(0.30, 0.60, 0.05):
            for cert_t in np.arange(0.55, 0.85, 0.05):
                mask = (
                    (all_confs >= conf_t)
                    & (all_entropies <= ent_t)
                    & (all_certs >= cert_t)
                    & (preds != 2)          # No rutear NEI al ligero
                )
                if mask.sum() < 10:
                    continue
                f1 = f1_score(all_labels[mask], preds[mask], average="macro", zero_division=0)
                coverage = mask.mean()
                if f1 >= target_f1 and coverage > best["coverage"]:
                    best = {"coverage": coverage, "conf": conf_t, "entropy": ent_t, "cert": cert_t}

    logger.info(
        "Umbrales calibrados → conf=%.2f, entropy=%.2f, cert=%.2f | cobertura_ligero=%.1f%%",
        best["conf"], best["entropy"], best["cert"], best["coverage"] * 100,
    )
    return best


# ============================================================
# 7. ENTRENAMIENTO
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = PhoBERTEvidenceTokenizer(args.model_name)
    model = EvidenceAwareFactChecker(
        phobert_name=args.model_name,
        num_evidence_attn_heads=args.attn_heads,
        dropout=args.dropout,
        num_labels=3,
    )
    model.to(device)

    if args.freeze_layers > 0:
        for p in model.encoder.embeddings.parameters():
            p.requires_grad = False
        for layer in model.encoder.encoder.layer[: args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info("Congeladas primeras %d capas del encoder.", args.freeze_layers)

    train_ds = VietnameseFactDataset(args.train_json, tokenizer)
    val_ds = VietnameseFactDataset(args.val_json, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    bce_loss_fn = nn.BCELoss()

    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None
    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()

            def forward_pass():
                logits, certainty, _, _ = model(input_ids, attention_mask, batch["evidence_spans"])
                loss_cls = ce_loss_fn(logits, labels)
                cert_target = (labels != 2).float().unsqueeze(1)
                loss_cert = bce_loss_fn(certainty, cert_target)
                return loss_cls + args.lambda_cert * loss_cert

            if scaler:
                with torch.cuda.amp.autocast():
                    loss = forward_pass()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward_pass()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

            if (i + 1) % args.log_every == 0:
                logger.info(
                    "Epoch %d | Batch %d/%d | Loss: %.4f",
                    epoch + 1, i + 1, len(train_loader), epoch_loss / (i + 1),
                )

        val_metrics = evaluate(
            model, val_loader, device,
            lambda_cert=args.lambda_cert,
            desc=f"Epoch {epoch + 1} Validation",
        )

        # FIX #4: Alerta si la tasa de NEI predichos supera umbral operacional
        if val_metrics["forced_rate"] > args.forced_alert_threshold:
            logger.warning(
                "ALERTA: tasa de predicciones NEI = %.1f%% (umbral=%.1f%%). "
                "Revisar datos o umbrales del router.",
                val_metrics["forced_rate"] * 100,
                args.forced_alert_threshold * 100,
            )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "f1_macro": best_f1,
                },
                ckpt_path,
            )
            logger.info("Checkpoint guardado (F1-macro: %.4f)", best_f1)
        else:
            patience_counter += 1
            logger.info("Sin mejora. Patience: %d/%d", patience_counter, args.patience)
            if patience_counter >= args.patience:
                logger.info("Early stopping activado.")
                break

    # --- Calibrar umbrales del router sobre val set ---
    logger.info("Calibrando umbrales del router sobre el val set...")
    thresholds = calibrate_router(model, val_loader, device, target_f1=args.router_target_f1)
    thresholds_path = os.path.join(args.output_dir, "router_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Umbrales guardados en %s", thresholds_path)

    # FIX #6: Cuantización INT8 del modelo ligero para inferencia en CPU
    if args.quantize:
        logger.info("Aplicando cuantización dinámica INT8...")
        model_cpu = model.cpu().eval()
        quantized = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8,
        )
        q_path = os.path.join(args.output_dir, "best_model_int8.pt")
        torch.save(quantized, q_path)
        logger.info("Modelo cuantizado guardado en %s", q_path)

    logger.info("Entrenamiento finalizado. Mejor F1-macro: %.4f", best_f1)


# ============================================================
# 8. INFERENCIA
# ============================================================

def predict_json(model, tokenizer, claim, contexts, device, thresholds=None, force_binary=True):
    model.eval()
    enc = tokenizer.encode(claim, contexts)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits, certainty, attn_probs, _ = model(input_ids, attention_mask, enc["evidence_spans"])

    probs = torch.softmax(logits, dim=-1)[0]
    id2label = {0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}
    pred_id = torch.argmax(probs).item()
    label = id2label[pred_id]
    confidence = probs[pred_id].item()

    was_forced = False
    if force_binary and label == "NEI":
        binary = probs[:2] / (probs[:2].sum() + 1e-10)
        pred_id = torch.argmax(binary).item()
        label = id2label[pred_id]
        confidence = binary[pred_id].item()
        was_forced = True

    num_ctx = len(contexts)
    evidence_scores = attn_probs[0, :num_ctx, 0].cpu().tolist()
    ranked = sorted(enumerate(evidence_scores), key=lambda x: x[1], reverse=True)

    return {
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "was_forced": was_forced,
        "certainty_not_nei": round(certainty.item(), 4),
        "top_evidence_indices": [i for i, _ in ranked[:3]],
        "evidence_attention": [
            {"evidence_idx": i, "score": round(s, 4)} for i, s in ranked
        ],
    }


# ============================================================
# 9. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Rutas de datos ajustadas al proyecto
    parser.add_argument("--train_json", default="HIGH_CONFIDENCE_train_dataset.json")
    parser.add_argument("--val_json",   default="HIGH_CONFIDENCE_validation_dataset.json")
    parser.add_argument("--test_json",  default="HIGH_CONFIDENCE_test_dataset.json")
    parser.add_argument("--model_name", default="vinai/phobert-base")
    parser.add_argument("--output_dir", default="./phobert_evidence_checkpoints")
    # FIX #5: épocas reducidas a 7 (el early stopping con patience=3 ya controla sobreajuste)
    parser.add_argument("--epochs",     type=int,   default=7)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout",    type=float, default=0.15)
    parser.add_argument("--attn_heads", type=int,   default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lambda_cert", type=float, default=0.3)
    parser.add_argument("--freeze_layers", type=int, default=3)
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every",  type=int,   default=50)
    parser.add_argument("--amp",        action="store_true")
    # FIX #4: umbral de alerta para tasa de NEI predichos
    parser.add_argument("--forced_alert_threshold", type=float, default=0.25,
                        help="Emitir WARNING si la tasa de predicciones NEI supera este valor")
    # FIX #6: cuantización INT8 al finalizar
    parser.add_argument("--quantize",   action="store_true",
                        help="Guardar modelo cuantizado INT8 para inferencia CPU")
    # Umbral de F1 objetivo para calibración del router
    parser.add_argument("--router_target_f1", type=float, default=0.85,
                        help="F1-macro mínimo exigido al router para maximizar cobertura del ligero")
    args = parser.parse_args()

    train(args)