# Endpoint desplegado en Azure

Proyecto desarrollado para el módulo de Cloud Computing.

**Equipo 1:**

Erick Isaac Lascano Otañez - A00836571

Pedro Soto Juárez - A00837560

Alexei Carrillo Acosta - A01285424

Mateo Zepeda Pérez - A01722398

Luis Fernando Alcazar Díaz - A00836287


**URL para entregar al torneo/profesor:**

```text
https://fact-checking-api.wittypond-c03af643.eastus.azurecontainerapps.io/predict
```

**Health check:**

```text
https://fact-checking-api.wittypond-c03af643.eastus.azurecontainerapps.io/health
```

---

# Fact-Check Vietnamita - Sistema de Verificación de Afirmaciones

Sistema de fact-checking para idioma vietnamita desplegado como API con **FastAPI**, **Docker** y **Azure Container Apps**.

La arquitectura final usa una cascada de dos modelos vietnamitas:

```text
PhoBERT-base  →  si duda  →  PhoBERT-base-v2
modelo ligero                modelo pesado/fallback
```

El endpoint final cumple el formato requerido por el torneo:

```json
{
  "predicted_label": "SUPPORTED"
}
```

o:

```json
{
  "predicted_label": "REFUTED"
}
```

Aunque los modelos internamente fueron entrenados con tres clases (`SUPPORTED`, `REFUTED`, `NEI`), la API **nunca devuelve `NEI`**. Si internamente aparece `NEI`, el sistema fuerza una decisión binaria comparando las probabilidades de `SUPPORTED` y `REFUTED`.

---

## Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Resultados de Entrenamiento](#resultados-de-entrenamiento)
3. [Archivos del Proyecto](#archivos-del-proyecto)
4. [Estructura Esperada](#estructura-esperada)
5. [Requisitos](#requisitos)
6. [Formato de Entrada y Salida](#formato-de-entrada-y-salida)
7. [Entrenamiento](#entrenamiento)
8. [Inferencia y Cascada](#inferencia-y-cascada)
9. [Prueba Local con Uvicorn](#prueba-local-con-uvicorn)
10. [Docker](#docker)
11. [Despliegue en Azure](#despliegue-en-azure)
12. [Notas Técnicas](#notas-técnicas)

---

## Arquitectura General

```text
┌────────────────────┐
│ API Request         │
│ claim + contexts    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ PhoBERT-base        │
│ modelo ligero       │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Router de duda      │
│ confianza/entropía  │
│ margen/NEI          │
└──────┬─────────────┘
       │
       ├── Alta confianza ─────▶ respuesta binaria
       │                         SUPPORTED / REFUTED
       │
       └── Baja confianza ─────▶ PhoBERT-base-v2
                                  modelo pesado/fallback
                                  ↓
                                  respuesta binaria
                                  SUPPORTED / REFUTED
```

### Flujo de una solicitud

1. El usuario manda un `claim` y una lista de `contexts`.
2. El modelo ligero `vinai/phobert-base` evalúa primero la solicitud.
3. El router calcula métricas de incertidumbre: confianza, entropía, margen entre `SUPPORTED` y `REFUTED`, y predicción interna `NEI`.
4. Si el modelo ligero está seguro, responde directamente.
5. Si el modelo ligero duda, se escala a `vinai/phobert-base-v2`.
6. La API devuelve únicamente:

```json
{
  "predicted_label": "SUPPORTED" | "REFUTED"
}
```

---

## Resultados de Entrenamiento

### PhoBERT-base

Resultado validación:

```text
Accuracy: 0.6577
F1-macro: 0.6005
SUPPORTED f1: 0.6756
REFUTED f1: 0.3801
NEI f1: 0.7458
```

### PhoBERT-base-v2

Mejor resultado validación:

```text
Best F1-macro: 0.6382
```

Resultado de una de las últimas épocas:

```text
Accuracy: 0.6622
F1-macro: 0.6273
SUPPORTED f1: 0.6747
REFUTED f1: 0.4674
NEI f1: 0.7396
```

### Decisión final

Se eligió `vinai/phobert-base-v2` como modelo pesado/fallback porque mejoró el F1-macro general y especialmente la clase `REFUTED`, que era la clase más débil.

`mDeBERTa` fue descartado porque durante el fine-tuning presentó inestabilidad numérica recurrente (`Loss: NaN`) y colapso de predicción hacia una sola clase.

---

## Archivos del Proyecto

| Archivo / Carpeta | Propósito |
|---|---|
| `app.py` | API FastAPI. Carga modelos y expone `/health` y `/predict`. |
| `cascade_router.py` | Router de cascada PhoBERT-base → PhoBERT-base-v2. |
| `train_phobert_evidence.py` | Define la arquitectura `EvidenceAwareFactChecker`, tokenizer e inferencia. |
| `requirements.txt` | Dependencias Python. No debe incluir `torch` si Docker instala PyTorch CPU por separado. |
| `dockerfile` | Imagen Docker para la API. |
| `.dockerignore` | Evita subir datasets, entorno virtual y archivos innecesarios al build. |
| `phobert_evidence_checkpoints/` | Checkpoint del modelo ligero PhoBERT-base. |
| `phobert_v2_evidence_checkpoints/` | Checkpoint del modelo pesado PhoBERT-base-v2. |

---

## Estructura Esperada

Antes de ejecutar localmente, dockerizar o subir a Azure, el proyecto debe verse así:

```text
Fact_Checking_Rocketpowers/
├── app.py
├── cascade_router.py
├── train_phobert_evidence.py
├── requirements.txt
├── dockerfile
├── .dockerignore
│
├── phobert_evidence_checkpoints/
│   ├── best_model.pt
│   └── router_thresholds.json
│
└── phobert_v2_evidence_checkpoints/
    ├── best_model.pt
    └── router_thresholds.json
```

> Importante: los archivos `.pt` no deben extraerse aunque Windows/WinRAR los detecte como comprimidos. PyTorch guarda internamente algunos `.pt` con estructura tipo ZIP, pero el código necesita cargar el archivo completo con `torch.load()`.

---

## Requisitos

### requirements.txt recomendado

```txt
fastapi
uvicorn[standard]
transformers
scikit-learn
numpy
pydantic
sentencepiece
protobuf
```

En Docker, `torch` se instala aparte en versión CPU para evitar instalar librerías CUDA/NVIDIA innecesarias.

---

## Formato de Entrada y Salida

### Entrada esperada

```json
{
  "claim": "So với cùng kỳ năm 2024, số vụ tai nạn giao thông tăng.",
  "contexts": [
    "Tình hình tai nạn giao thông giảm cả 3 tiêu chí so với cùng kỳ năm 2024."
  ]
}
```

### Salida obligatoria del torneo

```json
{
  "predicted_label": "REFUTED"
}
```

El endpoint nunca debe devolver `NEI`, probabilidades, explicaciones ni campos extra si la plataforma del torneo exige estrictamente el JSON anterior.

---

## Entrenamiento

### Modelo ligero: PhoBERT-base

```bash
python train_phobert_evidence.py \
  --model_name vinai/phobert-base \
  --train_json HIGH_CONFIDENCE_train_dataset.json \
  --val_json HIGH_CONFIDENCE_validation_dataset.json \
  --batch_size 1 \
  --epochs 5 \
  --freeze_layers 3 \
  --lr 5e-6 \
  --label_smoothing 0 \
  --max_grad_norm 0.5 \
  --output_dir ./phobert_evidence_checkpoints
```

### Modelo pesado/fallback: PhoBERT-base-v2

```bash
python train_phobert_evidence.py \
  --model_name vinai/phobert-base-v2 \
  --train_json HIGH_CONFIDENCE_train_dataset.json \
  --val_json HIGH_CONFIDENCE_validation_dataset.json \
  --batch_size 1 \
  --epochs 5 \
  --freeze_layers 3 \
  --lr 5e-6 \
  --label_smoothing 0 \
  --max_grad_norm 0.5 \
  --output_dir ./phobert_v2_evidence_checkpoints
```

### Ajustes necesarios en `train_phobert_evidence.py`

Para evitar errores CUDA de índices fuera de rango, se usaron dos ajustes importantes:

```python
# Longitud máxima reducida
max_length: int = 256
```

```python
# Redimensionar embeddings al agregar tokens especiales
model.encoder.resize_token_embeddings(len(tokenizer.tokenizer))
```

---

## Inferencia y Cascada

La cascada funciona así:

```text
1. PhoBERT-base predice primero.
2. Si tiene alta confianza, se usa su salida.
3. Si predice NEI, tiene baja confianza o bajo margen entre SUPPORTED/REFUTED, se consulta PhoBERT-base-v2.
4. La salida final se fuerza a SUPPORTED o REFUTED.
```

Criterios típicos de duda:

```python
doubt = (
    pred_id == 2 or          # NEI interno
    confidence < 0.65 or     # baja confianza
    margin < 0.15            # SUPPORTED y REFUTED están muy cerca
)
```

---

## Prueba Local con Uvicorn

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Levantar API:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Probar health:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Resultado esperado:

```text
status router_loaded device
------ ------------- ------
ok              True cpu
```

Probar predict:

```powershell
$body = @{
  claim = "So với cùng kỳ năm 2024, số vụ tai nạn giao thông tăng."
  contexts = @(
    "Tình hình tai nạn giao thông giảm cả 3 tiêu chí so với cùng kỳ năm 2024."
  )
} | ConvertTo-Json -Depth 5 -Compress

$utf8Body = [System.Text.Encoding]::UTF8.GetBytes($body)

Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $utf8Body
```

Salida esperada:

```text
predicted_label
---------------
REFUTED
```

---

## Docker

### Dockerfile recomendado

```dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch CPU >= 2.6, sin CUDA/NVIDIA
RUN pip install --no-cache-dir torch==2.6.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY cascade_router.py .
COPY train_phobert_evidence.py .

COPY phobert_evidence_checkpoints/ ./phobert_evidence_checkpoints/
COPY phobert_v2_evidence_checkpoints/ ./phobert_v2_evidence_checkpoints/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### .dockerignore recomendado

```dockerignore
env/
venv/
.venv/
__pycache__/
*.pyc

.git/
.gitignore

*.zip
*.rar
*.7z

*.ipynb
*.csv
*.xlsx
*.jsonl

OG_train_dataset.json
OG_validation_dataset.json
OG_test_dataset.json
HIGH_CONFIDENCE_train_dataset.json
HIGH_CONFIDENCE_validation_dataset.json
HIGH_CONFIDENCE_test_dataset.json
HIGH_CONFIDENCE_converted_dataset.json
output_dataset.json

mdeberta_factcheck/
mdeberta_factcheck_bad_nan/
mdeberta_factcheck_base/
mdeberta_factcheck_base_bad_nan/

sample_data/
```

No ignorar:

```text
phobert_evidence_checkpoints/
phobert_v2_evidence_checkpoints/
```

### Construir imagen

```powershell
docker build --no-cache -t fact-checking-api -f dockerfile .
```

### Ejecutar contenedor

```powershell
docker run -p 8001:8000 fact-checking-api
```

### Probar contenedor

```powershell
Invoke-RestMethod http://localhost:8001/health
```

```powershell
$body = @{
  claim = "So với cùng kỳ năm 2024, số vụ tai nạn giao thông tăng."
  contexts = @(
    "Tình hình tai nạn giao thông giảm cả 3 tiêu chí so với cùng kỳ năm 2024."
  )
} | ConvertTo-Json -Depth 5 -Compress

$utf8Body = [System.Text.Encoding]::UTF8.GetBytes($body)

Invoke-RestMethod `
  -Uri "http://localhost:8001/predict" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $utf8Body
```

---

## Despliegue en Azure

### Variables

```powershell
$RESOURCE_GROUP="rg-fact-checking"
$LOCATION="eastus"
$ACR_NAME="acrocketpowers01"
$APP_NAME="fact-checking-api"
$ENV_NAME="env-fact-checking"
$IMAGE_NAME="fact-checking-api"
$TAG="v1"
```

### Login y providers

```powershell
az login
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

### Crear Resource Group

```powershell
az group create `
  --name $RESOURCE_GROUP `
  --location $LOCATION
```

### Crear Azure Container Registry

```powershell
az acr create `
  --resource-group $RESOURCE_GROUP `
  --name $ACR_NAME `
  --sku Basic `
  --admin-enabled true
```

### Build y push a ACR

```powershell
az acr build `
  --registry $ACR_NAME `
  --image "$IMAGE_NAME`:$TAG" `
  -f dockerfile `
  .
```

### Verificar imagen

```powershell
az acr repository list `
  --name $ACR_NAME `
  --output table
```

```powershell
az acr repository show-tags `
  --name $ACR_NAME `
  --repository $IMAGE_NAME `
  --output table
```

### Crear entorno de Container Apps

```powershell
az containerapp env create `
  --name $ENV_NAME `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION
```

### Obtener credenciales ACR

```powershell
$ACR_USERNAME = az acr credential show `
  --name $ACR_NAME `
  --query username `
  -o tsv

$ACR_PASSWORD = az acr credential show `
  --name $ACR_NAME `
  --query "passwords[0].value" `
  -o tsv
```

### Crear Container App

```powershell
az containerapp create `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --environment $ENV_NAME `
  --image "$ACR_NAME.azurecr.io/$IMAGE_NAME`:$TAG" `
  --target-port 8000 `
  --ingress external `
  --cpu 2.0 `
  --memory 4Gi `
  --min-replicas 1 `
  --max-replicas 1 `
  --registry-server "$ACR_NAME.azurecr.io" `
  --registry-username $ACR_USERNAME `
  --registry-password $ACR_PASSWORD
```

### Obtener URL pública

```powershell
$APP_URL = az containerapp show `
  --name $APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --query properties.configuration.ingress.fqdn `
  -o tsv

"https://$APP_URL"
```

Endpoint final del torneo:

```text
https://<APP_URL>/predict
```

### Probar en Azure

```powershell
Invoke-RestMethod "https://$APP_URL/health"
```

```powershell
$body = @{
  claim = "So với cùng kỳ năm 2024, số vụ tai nạn giao thông tăng."
  contexts = @(
    "Tình hình tai nạn giao thông giảm cả 3 tiêu chí so với cùng kỳ năm 2024."
  )
} | ConvertTo-Json -Depth 5 -Compress

$utf8Body = [System.Text.Encoding]::UTF8.GetBytes($body)

Invoke-RestMethod `
  -Uri "https://$APP_URL/predict" `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $utf8Body
```

---

## Notas Técnicas

### Por qué se descartó mDeBERTa

Se probaron varias configuraciones de mDeBERTa, incluyendo `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` y `microsoft/mdeberta-v3-base`. En las pruebas completas apareció `Loss: NaN` y el modelo colapsó a una sola clase (`SUPPORTED`). Por estabilidad y desempeño, se descartó para la versión desplegable.

### Por qué usar PhoBERT/PhoBERT v2

PhoBERT es una familia de modelos especializados en vietnamita. El dataset del proyecto está en vietnamita, por lo que resultó más estable y adecuado usar modelos monolingües vietnamitas en lugar de un modelo multilingüe generalista.

### Sobre `NEI`

`NEI` se usa internamente durante entrenamiento e inferencia para detectar incertidumbre. Sin embargo, la especificación del torneo exige una decisión binaria. Por eso, cuando el modelo predice `NEI`, el sistema selecciona entre `SUPPORTED` y `REFUTED` según la probabilidad mayor.

### Sobre PyTorch CPU en Docker

Para evitar dependencias CUDA/NVIDIA innecesarias, Docker instala PyTorch CPU:

```dockerfile
RUN pip install --no-cache-dir torch==2.6.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

Esto evita errores relacionados con librerías como `libcusparseLt.so.0` y reduce problemas al ejecutar en Azure Container Apps con CPU.

---

## Estado Actual

- API local funcionando con Uvicorn.
- Docker local probado exitosamente.
- Endpoint `/health` devuelve `router_loaded=True`.
- Endpoint `/predict` devuelve únicamente `SUPPORTED` o `REFUTED`.
- Imagen lista para desplegar en Azure Container Apps.

