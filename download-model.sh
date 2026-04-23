#!/bin/bash
#
# Baixa um modelo do HuggingFace Hub para o diretório configurado em
# SGLANG_MODEL_PATH (.env). Usa o container oficial do huggingface_hub, então
# não exige Python/pip no host.
#
# Uso:
#   ./download-model.sh                              # usa .env
#   ./download-model.sh QuantTrio/Qwen3.6-35B-A3B-AWQ /data/sglang/models/meu-modelo
#
# Variáveis opcionais:
#   HF_TOKEN   token para modelos gated/privados (exporte antes de rodar)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Carrega .env se existir (para SGLANG_MODEL_PATH padrão)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

REPO_ID="${1:-QuantTrio/Qwen3.6-35B-A3B-AWQ}"
TARGET_DIR="${2:-${SGLANG_MODEL_PATH:-/data/sglang/models/qwen3.6-35b-a3b-awq}}"

echo "→ Repositório: $REPO_ID"
echo "→ Destino:     $TARGET_DIR"

mkdir -p "$TARGET_DIR"

# Usa HF_HUB_ENABLE_HF_TRANSFER=1 para acelerar downloads grandes.
docker run --rm \
  -v "$TARGET_DIR:/model" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  python:3.12-slim \
  bash -c "pip install --quiet --no-cache-dir 'huggingface_hub[cli,hf_transfer]' && \
           hf download '$REPO_ID' --local-dir /model"

echo
echo "✓ Modelo pronto em $TARGET_DIR"
echo "  Ajuste SGLANG_MODEL_PATH e SGLANG_SERVED_MODEL_NAME no .env se necessário."
echo "  Depois: docker compose up -d --force-recreate sglang"
