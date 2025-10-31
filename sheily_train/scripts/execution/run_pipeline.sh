#!/usr/bin/env bash
set -euo pipefail

# ==== CONFIG BÁSICA ====
BRANCH="${BRANCH:-noticias_diarias}"
OUTDIR="${OUTDIR:-./branches}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"

FEEDS_FILE="${FEEDS_FILE:-./feeds_es_news.txt}"
RSS_MAX="${RSS_MAX:-40}"
ONLY_LANG="${ONLY_LANG:-es}"

# LoRA SFT
LORA_OUT="${LORA_OUT:-./lora_out/${BRANCH}-mistral7b}"
SFT_BATCH="${SFT_BATCH:-1}"
SFT_ACCUM="${SFT_ACCUM:-16}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_LR="${SFT_LR:-1e-4}"
SFT_MAXLEN="${SFT_MAXLEN:-2048}"
SFT_BF16="${SFT_BF16:-true}"
SFT_GC="${SFT_GC:-true}"

# Retriever
RETRIEVER_BASE="${RETRIEVER_BASE:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
RETRIEVER_OUT="${RETRIEVER_OUT:-./retriever_out/${BRANCH}-minilm}"
RT_BATCH="${RT_BATCH:-64}"
RT_EPOCHS="${RT_EPOCHS:-1}"
RT_LR="${RT_LR:-2e-5}"

INSTALL_REQS="${INSTALL_REQS:-false}"

if [[ "${INSTALL_REQS}" == "true" ]]; then
  python -m pip install -r requirements.txt
fi

# PASO 1: Ingesta RSS → RAG
echo ">> Ingesta RSS para rama '${BRANCH}'..."
RSS_ARGS=()
if [[ -f "${FEEDS_FILE}" ]]; then
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    RSS_ARGS+=(--rss-url "$line")
  done < "${FEEDS_FILE}"
else
  echo "⚠️ No existe ${FEEDS_FILE}. Crea el archivo con tus feeds (uno por línea)."; exit 1
fi

python sheily_web2rag.py \
  --branch "${BRANCH}" \
  --connectors rss \
  "${RSS_ARGS[@]}" \
  --rss-max "${RSS_MAX}" \
  --no-crawl \
  --embed-model "${EMBED_MODEL}" \
  --outdir "${OUTDIR}" \
  --privacy hard \
  --only-lang "${ONLY_LANG}" \
  --near-dedup --near-thresh 0.92

RUN_DIR=$(ls -d "${OUTDIR}/${BRANCH}"/2* | sort -V | tail -n 1)
[[ -z "${RUN_DIR:-}" ]] && { echo "No se pudo resolver RUN_DIR"; exit 1; }

echo ">> RUN_DIR: ${RUN_DIR}"

# PASO 2: Retriever con triplets (si existen)
TRIPLETS="${RUN_DIR}/contrastive/triplets.jsonl"
if [[ -s "${TRIPLETS}" ]]; then
  echo ">> Entrenando retriever..."
  python train_retriever_triplet.py \
    --run_dir "${RUN_DIR}" \
    --base_encoder "${RETRIEVER_BASE}" \
    --output_dir "${RETRIEVER_OUT}" \
    --batch_size "${RT_BATCH}" \
    --epochs "${RT_EPOCHS}" \
    --lr "${RT_LR}"

  echo ">> Reindexando con el retriever ajustado..."
  python reindex_with_encoder.py \
    --run_dir "${RUN_DIR}" \
    --encoder_path "${RETRIEVER_OUT}"
else
  echo "⚠️ No hay triplets (contrastive/triplets.jsonl). Se omite entrenamiento de retriever."
fi

# PASO 3: LoRA SFT
echo ">> Entrenando LoRA SFT..."
python lora_sft_train.py \
  --run_dir "${RUN_DIR}" \
  --base_model "${BASE_MODEL}" \
  --output_dir "${LORA_OUT}" \
  --batch_size "${SFT_BATCH}" \
  --grad_accum "${SFT_ACCUM}" \
  --epochs "${SFT_EPOCHS}" \
  --lr "${SFT_LR}" \
  --max_seq_len "${SFT_MAXLEN}" \
  $([[ "${SFT_BF16}" == "true" ]] && echo --bf16) \
  $([[ "${SFT_GC}" == "true" ]] && echo --gradient_checkpointing)

printf "\n✅ Pipeline completado.\n  - RUN_DIR:        %s\n  - Retriever out:  %s (si se entrenó)\n  - LoRA adapter:   %s\n" "${RUN_DIR}" "${RETRIEVER_OUT}" "${LORA_OUT}"
