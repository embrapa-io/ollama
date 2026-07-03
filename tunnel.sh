#!/bin/bash
#
# Túnel SSH para o SGLang do GPU Server da GTI:
#   estação local → embrapa@core.embrapa.io → llm.nuvem.ti.embrapa.br:11435
#
# O salto core→llm é um encaminhamento TCP feito pelo core (flag -L), portanto
# não exige chave SSH sua no llm — basta o core alcançar a porta 11435.
#
# Uso:
#   ./tunnel.sh          # abre (idempotente) e testa GET /v1/models
#   ./tunnel.sh down     # encerra
#   ./tunnel.sh status   # verifica túnel + endpoint

set -euo pipefail

JUMP="${TUNNEL_JUMP:-embrapa@core.embrapa.io}"
TARGET_HOST="${TUNNEL_HOST:-llm.nuvem.ti.embrapa.br}"
TARGET_PORT="${TUNNEL_PORT:-11435}"
LOCAL_PORT="${TUNNEL_LOCAL_PORT:-11435}"
CTRL="${HOME}/.ssh/ctl-sglang-tunnel.sock"

is_up() { ssh -O check -S "$CTRL" "$JUMP" 2>/dev/null; }

probe() {
  echo "→ GET http://localhost:${LOCAL_PORT}/v1/models"
  curl -sf --max-time 5 "http://localhost:${LOCAL_PORT}/v1/models" && echo || {
    echo "✗ endpoint não respondeu (túnel ok ≠ SGLang no ar — conferir logs no servidor)"
    return 1
  }
}

case "${1:-up}" in
  up)
    if is_up; then
      echo "✓ Túnel já ativo"
    else
      ssh -f -N -M -S "$CTRL" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -L "${LOCAL_PORT}:${TARGET_HOST}:${TARGET_PORT}" "$JUMP"
      echo "✓ Túnel aberto: localhost:${LOCAL_PORT} → ${TARGET_HOST}:${TARGET_PORT} (via ${JUMP})"
    fi
    probe
    ;;
  down)
    if is_up; then
      ssh -O exit -S "$CTRL" "$JUMP" 2>/dev/null
      echo "✓ Túnel encerrado"
    else
      echo "nenhum túnel ativo"
    fi
    ;;
  status)
    if is_up; then
      echo "✓ Túnel ativo (localhost:${LOCAL_PORT})"
      probe
    else
      echo "✗ Túnel inativo — rode ./tunnel.sh"
      exit 1
    fi
    ;;
  *)
    echo "uso: $0 [up|down|status]" >&2
    exit 2
    ;;
esac
