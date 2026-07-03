#!/bin/bash
#
# Túnel SSH para o LLM server do GPU Server da GTI, em DOIS saltos:
#   estação local ──ssh──▶ embrapa@core.embrapa.io ──ssh──▶ embrapa@llm:11435
#
# Dois saltos porque o firewall entre core e llm filtra a porta 11435 (só o
# SSH/22 atravessa) — o encaminhamento TCP direto pelo core não funciona.
# O 2º salto usa a chave SSH que o core já tem para o llm (BatchMode).
#
# Uso:
#   ./tunnel.sh          # abre (idempotente) e testa GET /v1/models
#   ./tunnel.sh down     # encerra
#   ./tunnel.sh status   # verifica túnel + endpoint

set -euo pipefail

JUMP="${TUNNEL_JUMP:-embrapa@core.embrapa.io}"
TARGET_SSH="${TUNNEL_TARGET_SSH:-embrapa@llm.nuvem.ti.embrapa.br}"
TARGET_PORT="${TUNNEL_PORT:-11435}"
LOCAL_PORT="${TUNNEL_LOCAL_PORT:-11435}"
RELAY_PORT="${TUNNEL_RELAY_PORT:-21435}"
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
      # Salto 1: local:LOCAL_PORT → core:127.0.0.1:RELAY_PORT (multiplexado em $CTRL)
      # Salto 2 (comando remoto no core): core:RELAY_PORT → llm:127.0.0.1:TARGET_PORT
      ssh -f -M -S "$CTRL" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -L "${LOCAL_PORT}:127.0.0.1:${RELAY_PORT}" "$JUMP" \
        "ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N -L ${RELAY_PORT}:127.0.0.1:${TARGET_PORT} ${TARGET_SSH}"
      echo "✓ Túnel aberto: localhost:${LOCAL_PORT} → ${JUMP} → ${TARGET_SSH}:${TARGET_PORT}"
      sleep 2
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
