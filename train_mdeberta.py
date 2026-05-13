#!/usr/bin/env python3
# train_mdeberta.py
# Entrenamiento del modelo pesado: mDeBERTa-v3-base-mnli-xnli
# Correcciones aplicadas:
#   1. Rutas de datos ajustadas al proyecto confirmado.
#   2. Épocas reducidas a 5 (era 5, se mantiene; patience=3 controla).
#   3. Alerta de was_forced en evaluación.
#   4. Exportación ONNX al final (--export_onnx).
#   5. AMP (mixed precision) opcional para GPU.

import os
import json
import logging
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 1. DATASET
# ============================================================

class DebertaFactDataset(Dataset):
    # Rutas confirmadas del proyecto
    DEFAULT_TRAIN = "HIGH_CONFIDENCE_train_dataset.json"
    DEFAULT_VAL   = "HIGH_CONFIDENCE_validation_dataset.json"
    DEFAULT_TEST  = "HIGH_CONFIDENCE_test_dataset.json"

    def __init__(self, json_path: str, tokenizer, max_length: int = 512):
        with open(json_path, "r", encoding="utf-8") as f:
            self.raw = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"SUPPORTED": 0, "REFUTED": 1, "NEI": 2}
        logger.info("Dataset cargado: %s (%d registros)", json_path, len(self.raw))

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        item = self.raw[idx]
        claim = item["claim"]
        contexts = item.get("contexts", [])
        label_str = item["label"]

        # mDeBERTa-v3 usa </s> como separador
        evidence_block = " </s> ".join(contexts)
        text = f"{claim} </s> {evidence_block}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        label = self.label_map.get(label_str, 2)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros_like(encoding["input_ids"])
            ).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# 2. MÉTRICAS
# ============================================================

def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


# ============================================================
# 3. EVALUACIÓN CON MONITOREO DE NEI
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, desc="Valid", forced_alert_threshold=0.25):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        total_loss += criterion(outputs.logits, labels).item()
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader)

    nei_pred_rate = sum(p == 2 for p in all_preds) / max(len(all_preds), 1)
    metrics["nei_pred_rate"] = nei_pred_rate

    logger.info("\n--- %s ---", desc)
    logger.info(
        "Loss: %.4f | Acc: %.4f | F1-macro: %.4f | NEI_pred_rate: %.1f%%",
        metrics["loss"], metrics["accuracy"], metrics["f1_macro"], nei_pred_rate * 100,
    )
    print(classification_report(
        all_labels, all_preds,
        target_names=["SUPPORTED", "REFUTED", "NEI"],
        digits=4, zero_division=0,
    ))

    # FIX #4: alerta de forced rate elevado
    if nei_pred_rate > forced_alert_threshold:
        logger.warning(
            "ALERTA: el modelo pesado predice NEI en %.1f%% de los casos (umbral=%.1f%%). "
            "Esto incrementará la tasa de was_forced en producción.",
            nei_pred_rate * 100, forced_alert_threshold * 100,
        )

    return metrics


# ============================================================
# 4. ENTRENAMIENTO
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    if args.freeze_layers > 0:
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
        for layer in model.deberta.encoder.layer[: args.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        logger.info("Congeladas las primeras %d capas del encoder.", args.freeze_layers)

    train_ds = DebertaFactDataset(args.train_json, tokenizer, args.max_length)
    val_ds   = DebertaFactDataset(args.val_json,   tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

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

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    ).logits
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).logits
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info("Epoch %d/%d | Train Loss: %.4f", epoch + 1, args.epochs, avg_loss)

        metrics = evaluate(
            model, val_loader, device,
            desc=f"Epoch {epoch + 1} Validation",
            forced_alert_threshold=args.forced_alert_threshold,
        )

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            tokenizer.save_pretrained(args.output_dir)
            logger.info("Mejor modelo guardado (F1-macro: %.4f)", best_f1)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping activado.")
                break

    logger.info("Entrenamiento finalizado. Mejor F1-macro: %.4f", best_f1)

    # FIX #4: exportar a ONNX para inferencia GPU en la nube
    if args.export_onnx:
        logger.info("Exportando modelo a ONNX...")
        model.eval()
        dummy_ids   = torch.zeros(1, args.max_length, dtype=torch.long, device=device)
        dummy_mask  = torch.ones(1, args.max_length, dtype=torch.long, device=device)
        dummy_types = torch.zeros(1, args.max_length, dtype=torch.long, device=device)
        onnx_path = os.path.join(args.output_dir, "model.onnx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask, dummy_types),
                onnx_path,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids":      {0: "batch"},
                    "attention_mask": {0: "batch"},
                    "token_type_ids": {0: "batch"},
                    "logits":         {0: "batch"},
                },
                opset_version=14,
            )
        logger.info("Modelo ONNX guardado en %s", onnx_path)


# ============================================================
# 5. INFERENCIA
# ============================================================

def predict_json(model, tokenizer, claim: str, contexts: list, device, force_binary: bool = True):
    model.eval()
    evidence_block = " </s> ".join(contexts)
    text = f"{claim} </s> {evidence_block}"

    enc = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]

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

    return {
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "was_forced": was_forced,
        "probabilities": {
            "SUPPORTED": round(probs[0].item(), 4),
            "REFUTED":   round(probs[1].item(), 4),
            "NEI":       round(probs[2].item(), 4),
        },
    }


# ============================================================
# 6. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar mDeBERTa-v3 para fact-checking vietnamita")
    # Rutas ajustadas al proyecto
    parser.add_argument("--train_json",  default="HIGH_CONFIDENCE_train_dataset.json")
    parser.add_argument("--val_json",    default="HIGH_CONFIDENCE_validation_dataset.json")
    parser.add_argument("--test_json",   default="HIGH_CONFIDENCE_test_dataset.json")
    parser.add_argument("--model_name",  default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--output_dir",  default="./mdeberta_factcheck")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--freeze_layers",   type=int,   default=6)
    parser.add_argument("--patience",    type=int,   default=3)
    parser.add_argument("--max_grad_norm",   type=float, default=1.0)
    parser.add_argument("--amp",         action="store_true", help="Mixed precision en GPU")
    parser.add_argument("--export_onnx", action="store_true", help="Exportar a ONNX al finalizar")
    parser.add_argument("--forced_alert_threshold", type=float, default=0.25,
                        help="Alerta si la tasa de NEI predichos supera este valor")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)