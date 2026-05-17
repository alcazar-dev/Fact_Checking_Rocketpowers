# Fact-Check Vietnamita - Sistema de Verificación de Afirmaciones

Sistema de fact-checking para idioma vietnamita con arquitectura en cascada:
**PhoBERT ligero** (ruta rápida) → **mDeBERTa-v3 pesado** (casos difíciles).
Incluye atención por evidencia, calibración automática de umbrales y monitoreo de drift en producción.

---

## Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Archivos del Proyecto](#archivos-del-proyecto)
3. [Requisitos](#requisitos)
4. [Estructura de Datos](#estructura-de-datos)
5. [Entrenamiento](#entrenamiento)
   - [5.1 Modelo Ligero (PhoBERT)](#51-modelo-ligero-phobert)
   - [5.2 Modelo Pesado (mDeBERTa-v3)](#52-modelo-pesado-mdeberta-v3)
6. [Inferencia y Cascada](#inferencia-y-cascada)
7. [Calibración del Router](#calibración-del-router)
8. [Monitoreo en Producción](#monitoreo-en-producción)
9. [Formato de Salida](#formato-de-salida)
10. [Docker y Despliegue](#docker-y-despliegue)

---

## Arquitectura General

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   API Request   │────▶│  PhoBERT Ligero │────▶│ mDeBERTa Pesado │
│  claim+contexts │     │  + Atención     │     │  (si duda)      │
│                 │     │  por Evidencia  │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐             │
         │              │  Router decide: │             │
         │              │  ¿Confianza     │             │
         │              │  suficiente?    │             │
         │              └─────────────────┘             │
         │                       │                       │
         ▼                       ▼                       ▼
   ┌──────────┐           ┌──────────┐           ┌──────────┐
   │ SUPPORTED│           │ SUPPORTED│           │ SUPPORTED│
   │ REFUTED  │           │ REFUTED  │           │ REFUTED  │
   │ (~50ms)  │           │ (~50ms)  │           │ (~200ms) │
   └──────────┘           └──────────┘           └──────────┘
        Rápida              Rápida                 Lenta
      (conf alta)        (umbral calibrado)     (escalar)
```

**Flujo de una solicitud:**
1. El `claim` y sus `contexts` se tokenizan con marcadores especiales `<unused0>` antes de cada evidencia.
2. El **modelo ligero** (PhoBERT-base + atención entre evidencias) evalúa la solicitud.
3. El **router** analiza la confianza, entropía y certeza del ligero.
4. Si los umbrales se cumplen → responde el ligero (~50ms en CPU).
5. Si no → escala al **modelo pesado** (mDeBERTa-v3, ~200ms en GPU T4).
6. Si el modelo pesado predice `NEI`, se fuerza a `SUPPORTED` o `REFUTED` con normalización de probabilidades.

---

## Archivos del Proyecto

| Archivo | Propósito | Cuándo ejecutar |
|---------|-----------|-----------------|
| `train_phobert_evidence.py` | Entrena el modelo ligero con atención por evidencia y loss de certeza | Primero (base del sistema) |
| `train_mdeberta.py` | Entrena el modelo pesado pre-entrenado en NLI multilingüe | Segundo (fallback del sistema) |
| `cascade_router.py` | Orquesta la cascada: carga umbrales calibrados, decide ligero vs pesado | En producción (importar como módulo) |
| `HIGH_CONFIDENCE_*.json` | Datos limpios con consenso de anotadores | Entrenamiento principal |
| `OG_*.json` | Datos originales con más ruido, incluye clase `NEI` | Complemento opcional |
| `output_dataset.json` | Metadata de votación y consenso | Auditoría y análisis de errores |

---

## Requisitos

```bash
# Core
pip install torch transformers scikit-learn

# Opcional: ONNX para inferencia acelerada en GPU
pip install onnx onnxruntime-gpu

# Opcional: Mixed Precision (AMP) — requiere GPU NVIDIA
# torch.cuda.amp ya viene incluido en PyTorch ≥1.6
```

**Hardware recomendado:**
- **Entrenamiento ligero**: CPU o GPU con 4GB VRAM (GTX 1050 Ti suficiente)
- **Entrenamiento pesado**: GPU con 8GB VRAM (RTX 2070 / T4)
- **Producción**: CPU para ligero + GPU opcional para pesado, o solo CPU con modelo cuantizado

---

## Estructura de Datos

### Formato de entrada (JSON)

```json
{
  "claim": "Vào chiều ngày 9/9, Tổng Bí thư Tô Lâm đã chủ trì họp triển khai...",
  "contexts": [
    "Tổng Bí thư Tô Lâm nhấn mạnh, ngành giáo dục phải trả lời...",
    "Ngành giáo dục cần tập trung vào chất lượng đào tạo..."
  ],
  "label": "SUPPORTED"
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `claim` | `str` | Afirmación a verificar (vietnamita) |
| `contexts` | `list[str]` | Evidencias recuperadas de la web |
| `label` | `str` | Veredicto: `SUPPORTED` \| `REFUTED` \| `NEI` |

### Familias de datos

| Familia | Archivos | Registros | Uso |
|---------|----------|-----------|-----|
| **HIGH_CONFIDENCE** | `train` (3,106), `val` (666), `test` (665) | ~4,437 | **Entrenamiento principal**. Datos con consenso de anotadores. Más limpios, menos ruido. |
| **OG (Original)** | `train` (5,320), `val` (1,140), `test` (1,140) | ~7,600 | Datos crudos. Incluyen `NEI` y más variabilidad. Útil para robustecer el modelo pesado. |
| **output_dataset** | `output_dataset.json` | 7,600 | **No para entrenar**. Metadata de votación (`voter_details`, `consensus_score`, `seed_context`). Usar para debug y auditoría. |

---

## Entrenamiento

### 5.1 Modelo Ligero (PhoBERT)

**Qué hace:**
- Usa `vinai/phobert-base` (135M parámetros, vietnamita nativo)
- Inserta token `<unused0>` antes de cada contexto para delimitar evidencias
- Aplica `nn.MultiheadAttention` entre los vectores de evidencia extraídos
- Predice: (1) clase del veredicto, (2) certeza de que NO es `NEI`

**Ejecutar:**

```bash
python train_phobert_evidence.py     --train_json HIGH_CONFIDENCE_train_dataset.json     --val_json HIGH_CONFIDENCE_validation_dataset.json     --batch_size 16     --epochs 7     --freeze_layers 3     --quantize     --output_dir ./phobert_evidence_checkpoints
```

**Parámetros clave:**

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `--freeze_layers` | 3 | Congela primeras N capas del encoder. Con 7K datos, evita sobreajuste. |
| `--label_smoothing` | 0.1 | Suaviza targets. Evita confianza excesiva con pocos ejemplos. |
| `--lambda_cert` | 0.3 | Peso de la loss auxiliar de certeza. Enseña al modelo a detectar `NEI`. |
| `--quantize` | — | Guarda versión INT8 para CPU (`best_model_int8.pt`). |
| `--router_target_f1` | 0.85 | F1 mínimo exigido al calibrar umbrales del router. |

**Salidas generadas:**

```
phobert_evidence_checkpoints/
├── best_model.pt              # Checkpoint PyTorch completo
├── best_model_int8.pt         # Versión cuantizada (si --quantize)
└── router_thresholds.json     # Umbrales calibrados automáticamente
```

### 5.2 Modelo Pesado (mDeBERTa-v3)

**Qué hace:**
- Usa `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (278M parámetros)
- Ya pre-entrenado en NLI (Natural Language Inference) multilingüe, incluye vietnamita
- Fine-tuning mínimo: reemplaza la cabeza clasificadora (`ignore_mismatched_sizes=True`)
- Formato de entrada: `claim </s> context1 </s> context2 </s> ...`

**Ejecutar:**

```bash
python train_mdeberta.py     --train_json HIGH_CONFIDENCE_train_dataset.json     --val_json HIGH_CONFIDENCE_validation_dataset.json     --batch_size 8     --epochs 5     --freeze_layers 6     --export_onnx     --output_dir ./mdeberta_factcheck
```

**Parámetros clave:**

| Parámetro | Default | Efecto |
|-----------|---------|--------|
| `--freeze_layers` | 6 | Congela embeddings + 6 capas del encoder. Solo entrena la cabeza y capas superiores. |
| `--export_onnx` | — | Exporta `model.onnx` para inferencia acelerada con ONNX Runtime. |
| `--amp` | — | Habilita Automatic Mixed Precision (FP16). Ahorra VRAM y acelera en GPU. |

**Salidas generadas:**

```
mdeberta_factcheck/
├── best_model.pt              # Pesos PyTorch
├── tokenizer_config.json      # Tokenizer guardado
├── model.onnx                 # Modelo ONNX (si --export_onnx)
└── ...
```

---

## Inferencia y Cascada

### Uso básico del router

```python
import torch
from train_phobert_evidence import EvidenceAwareFactChecker, PhoBERTEvidenceTokenizer
from cascade_router import UncertaintyRouter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cargar ligero ---
tok_light = PhoBERTEvidenceTokenizer()
light = EvidenceAwareFactChecker()
ckpt = torch.load("phobert_evidence_checkpoints/best_model.pt", map_location=device)
light.load_state_dict(ckpt["model_state_dict"])

# --- Cargar pesado ---
tok_heavy = AutoTokenizer.from_pretrained("./mdeberta_factcheck")
heavy = AutoModelForSequenceClassification.from_pretrained(
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    num_labels=3,
    ignore_mismatched_sizes=True
)
heavy.load_state_dict(torch.load("mdeberta_factcheck/best_model.pt", map_location=device))

# --- Crear router ---
router = UncertaintyRouter(
    light_model=light,
    heavy_model=heavy,
    tokenizer_light=tok_light,
    tokenizer_heavy=tok_heavy,
    thresholds_path="phobert_evidence_checkpoints/router_thresholds.json",
    device=device,
)

# --- Predecir ---
result = router.predict(
    claim="Vào chiều ngày 9/9, Tổng Bí thư Tô Lâm đã chủ trì họp triển khai...",
    contexts=[
        "Tổng Bí thư Tô Lâm nhấn mạnh, ngành giáo dục phải trả lời...",
        "Ngành giáo dục cần tập trung vào chất lượng đào tạo..."
    ]
)

print(result)
```

### Estadísticas del router

```python
# Ver métricas de routing
stats = router.get_stats()
print(stats)
# {
#   "light": 145, "heavy": 55, "forced": 12, "total": 200,
#   "light_rate": 0.725, "heavy_rate": 0.275, "forced_rate": 0.060
# }
```

---

## Calibración del Router

El router **no usa umbrales fijos**. Al finalizar el entrenamiento del ligero, se ejecuta `calibrate_router()` que hace grid search sobre el validation set:

```python
# Pseudocódigo de la calibración
for conf_t in [0.75, 0.80, ..., 0.95]:
    for ent_t in [0.30, 0.35, ..., 0.55]:
        for cert_t in [0.55, 0.60, ..., 0.80]:
            # Aceptar solo si F1-macro ≥ 0.85 en val set
            # Maximizar cobertura del ligero (tasa de requests que resuelve)
```

Los umbrales óptimos se guardan en `router_thresholds.json`:

```json
{
  "coverage": 0.72,
  "conf": 0.85,
  "entropy": 0.45,
  "cert": 0.75
}
```

**Si el archivo no existe**, el router usa valores por defecto y emite un `WARNING`.

---

## Monitoreo en Producción

### Alertas configurables

| Situación | Umbral | Acción |
|-----------|--------|--------|
| Tasa de `NEI` predichos en validación | `--forced_alert_threshold` (default: 25%) | `WARNING` en logs |
| Tasa de `was_forced` en producción (ventana deslizante) | `forced_alert_threshold` en router (default: 25%) | `WARNING` en logs |

### Ventana deslizante de drift

El router mantiene una ventana de las últimas 200 solicitudes. Si la tasa de `was_forced` supera el umbral, emite:

```
WARNING | ALERTA DE DRIFT: tasa de was_forced=32.0% en las últimas 200 solicitudes
          (umbral=25.0%). Revisar distribución de datos en producción.
```

Esto detecta:
- **Data drift**: Los nuevos claims son más difíciles que los de entrenamiento
- **Model degradation**: El ligero está perdiendo capacidad (posible reentrenamiento necesario)

---

## Formato de Salida

### Output obligatorio (tu especificación)

```json
{
  "predicted_label": "SUPPORTED",
  "confidence": 0.9476,
  "was_forced": false,
  "tier": "light",
  "routing_reason": "high_confidence",
  "certainty_not_nei": 0.8912,
  "top_evidence_indices": [2, 0, 1],
  "evidence_attention": [
    {"evidence_idx": 2, "score": 0.4123},
    {"evidence_idx": 0, "score": 0.3891},
    {"evidence_idx": 1, "score": 0.1986}
  ]
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `predicted_label` | `str` | **Siempre** `SUPPORTED` o `REFUTED`. Nunca `NEI` en producción. |
| `confidence` | `float` | Probabilidad de la clase elegida (0.0–1.0) |
| `was_forced` | `bool` | `true` si el modelo originalmente predijo `NEI` y se normalizó a binario |
| `tier` | `str` | `"light"` o `"heavy"` — qué modelo tomó la decisión |
| `routing_reason` | `str` | Por qué se eligió ese tier (o por qué se escaló) |
| `certainty_not_nei` | `float` | Confianza interna de que NO es `NEI` (solo en tier ligero) |
| `top_evidence_indices` | `list[int]` | Índices de los 3 contexts más influyentes |
| `evidence_attention` | `list[dict]` | Score de atención para **cada** contexto, ordenado descendente |

### Caso: forzado desde `NEI`

```json
{
  "predicted_label": "REFUTED",
  "confidence": 0.6234,
  "was_forced": true,
  "tier": "heavy",
  "routing_reason": "light_uncertain(conf=0.72, ent=0.51, label=NEI)",
  "light_fallback_label": "NEI"
}
```

Cuando `was_forced=true`, el sistema está operando en modo degradado: el modelo no encontró evidencia clara y tuvo que "adivinar" entre las dos opciones permitidas.

---

## Docker y Despliegue

### Dockerfile recomendado

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu     && pip install --no-cache-dir transformers scikit-learn fastapi uvicorn

# Copiar modelos entrenados
COPY phobert_evidence_checkpoints/ ./phobert_evidence_checkpoints/
COPY mdeberta_factcheck/ ./mdeberta_factcheck/

# Copiar código
COPY train_phobert_evidence.py cascade_router.py main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
numpy>=1.24.0
```

### Endpoints FastAPI mínimos

```python
# main.py (esqueleto)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class FactCheckInput(BaseModel):
    claim: str
    contexts: List[str]

class FactCheckOutput(BaseModel):
    predicted_label: str
    confidence: float
    was_forced: bool
    tier: str

# Cargar router una vez al iniciar (singleton)
# router = UncertaintyRouter(...)

@app.post("/predict", response_model=FactCheckOutput)
async def predict(input_data: FactCheckInput):
    result = router.predict(input_data.claim, input_data.contexts)
    return FactCheckOutput(
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        was_forced=result["was_forced"],
        tier=result["tier"]
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "light_rate": router.get_stats()["light_rate"],
        "heavy_rate": router.get_stats()["heavy_rate"],
    }
```


---

*Sistema desarrollado para fact-checking en idioma vietnamita.*
*Arquitectura: PhoBERT Evidence-Attention + mDeBERTa-v3 Cascada con Calibración Automática.*
