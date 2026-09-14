#!/bin/bash
#
# Baixa um modelo do HuggingFace Hub para o diretório configurado em
# SGLANG_MODEL_PATH (.env). Usa o container oficial do huggingface_hub, então
# não exige Python/pip no host.
#
# Uso:
#   ./download-model.sh                              # usa SGLANG_MODEL_REPO + SGLANG_MODEL_PATH do .env
#   ./download-model.sh QuantTrio/Qwen3.5-27B-AWQ /data/sglang/models/meu-modelo
#
# HF_TOKEN: lido do .env ou do ambiente. Se vazio e o terminal for interativo,
# o script pergunta (obter em https://huggingface.co/settings/tokens).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Extrai um único valor do .env sem `source` (vars podem conter espaços e
# tokens como `--flag`, que o bash tentaria executar como comando).
get_env() {
  local key="$1"
  [[ -f .env ]] || return 0
  grep -E "^${key}=" .env | tail -1 | sed -E "s/^${key}=//; s/^[\"'](.*)[\"']\$/\\1/"
}

REPO_ID="${1:-$(get_env SGLANG_MODEL_REPO)}"
REPO_ID="${REPO_ID:-QuantTrio/Qwen3.6-27B-AWQ}"

TARGET_DIR="${2:-$(get_env SGLANG_MODEL_PATH)}"
TARGET_DIR="${TARGET_DIR:-/data/sglang/models/qwen3.6-27b-awq}"

HF_TOKEN="${HF_TOKEN:-$(get_env HF_TOKEN)}"

echo "→ Repositório: $REPO_ID"
echo "→ Destino:     $TARGET_DIR"

# Pede HF_TOKEN se vazio e estivermos em terminal interativo.
if [[ -z "${HF_TOKEN:-}" && -t 0 ]]; then
  echo
  echo "HF_TOKEN não definido. Sem ele o HuggingFace aplica rate-limit agressivo"
  echo "e modelos gated falham. Obtenha um token 'Read' em:"
  echo "  https://huggingface.co/settings/tokens"
  read -r -s -p "HF_TOKEN (deixe vazio para seguir sem autenticação): " HF_TOKEN
  echo
  export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "⚠️  Baixando sem autenticação — sujeito a rate-limit."
fi

# Trava contra volume desmontado (incidente de 11/09/2026 em llm.nuvem: o LV do
# /dados não ativou no boot, o download caiu na raiz e encheu o disco). Se o
# destino começa por um diretório que costuma ser ponto de montagem (/dados,
# /data, /mnt, /srv) mas esse diretório está no mesmo filesystem da raiz, aborta.
# Para forçar (destino deliberadamente na raiz): ALLOW_ROOT_FS=1 ./download-model.sh …
top="/$(echo "$TARGET_DIR" | cut -d/ -f2)"
case "$top" in
  /dados|/data|/mnt|/srv)
    mp="$(findmnt -rn -T "$top" -o TARGET 2>/dev/null || true)"   # filesystem que contém $top
    if [[ "${ALLOW_ROOT_FS:-0}" != "1" ]] && [[ -d "$top" ]] && [[ "$mp" == "/" ]]; then
      echo "✗ $top não é um ponto de montagem: está no filesystem da raiz (/)." >&2
      echo "  O volume esperado não está montado — o download encheria o disco do sistema." >&2
      echo "  Confira: lsblk; sudo lvscan; findmnt $top. Para forçar: ALLOW_ROOT_FS=1 $0 …" >&2
      exit 1
    fi
    ;;
esac

mkdir -p "$TARGET_DIR"

# HF_XET_HIGH_PERFORMANCE=1 acelera downloads grandes (substitui o antigo
# HF_HUB_ENABLE_HF_TRANSFER, descontinuado no huggingface_hub 1.x). Flags de
# pip silenciam o aviso de versão e o warning de root.
docker run --rm \
  -v "$TARGET_DIR:/model" \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  -e PIP_ROOT_USER_ACTION=ignore \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  python:3.12-slim \
  bash -c "pip install --quiet --no-cache-dir 'huggingface_hub[hf_xet]>=1.0' && \
           hf download '$REPO_ID' --local-dir /model"

echo
echo "✓ Modelo pronto em $TARGET_DIR"
echo "  Ajuste SGLANG_MODEL_PATH e SGLANG_SERVED_MODEL_NAME no .env se necessário."
echo "  Depois: docker compose up -d --force-recreate sglang"
