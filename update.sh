#!/bin/bash
#
# Atualização diária da stack, chamada pelo cron do host (06h25 UTC), com a saída
# gravada em /var/log/ollama_update_<data>.log.
#
# Incidente de 09–12/10/2026 no GPU Server da GTI: o unattended-upgrades trocou o
# driver NVIDIA com o servidor no ar; o módulo do kernel ficou na versão antiga, o
# `--force-recreate` deste script derrubou o vLLM, que não subiu mais (NVRM: API
# mismatch), e o `set -e` abortou antes do prune — as imagens novas foram se
# acumulando no disco raiz até o alerta de "Disco Crítico". Daí as três regras:
#   1. preflight: não tocar nos contêineres com o driver inconsistente;
#   2. `up` sem --force-recreate: só recria o que mudou (o vLLM leva ~8 min para subir);
#   3. a limpeza roda sempre, mesmo quando o `up` falha.
# A lista de pacotes que o unattended-upgrades não deve tocar está em
# setup/unattended-upgrades-nvidia.conf (ver README, "Atualização diária").

set -u

cd "$(dirname "$0")"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

fail=0

# 1) Preflight: bibliotecas de usuário × módulo do kernel do driver NVIDIA.
if ! nvidia-smi -L >/dev/null 2>&1; then
  log "✗ nvidia-smi falhou: driver NVIDIA inconsistente (pacote nvidia-driver-* atualizado sem reboot?)."
  log "  Nenhum contêiner será recriado. Conferir: dmesg | grep NVRM; dkms status. Reiniciar o host na próxima janela."
  fail=1
else
  log "driver ok: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

  if git fetch --all && git pull; then :; else
    log "⚠ git pull falhou; segue com o código atual"
  fi

  # GPUs ocupadas por outra stack (ex.: o Forge no llm.nuvem desde 19/10/2026, com o Whisper na GPU 0 e o
  # docling na GPU 1): o vLLM reserva 90 % das duas GPUs e o `up` religaria um vLLM parado de propósito,
  # derrubando a outra stack por falta de memória (ou ele mesmo, sem memória para subir). Nesse caso, nada sobe.
  others=$(docker ps --format '{{.ID}}' | xargs -r docker inspect --format \
    '{{index .Config.Labels "com.docker.compose.project"}} {{.Name}} {{json .HostConfig.DeviceRequests}}' \
    | grep -i nvidia | grep -v '^ollama ' || true)
  if [ -n "$others" ]; then
    log "⚠ GPUs em uso por outra stack; a stack do LLM não será religada:"
    echo "$others" | awk '{print "    " $1 " " $2}'
    log "  Para voltar o vLLM a este host, pare antes a outra stack (ver README, \"Host com outra stack na GPU\")."
  else
    if ! docker compose pull --ignore-pull-failures --ignore-buildable; then
      log "⚠ pull de imagens falhou (rede ou registro?); segue com as imagens locais"
    fi

    # 2) Sem --force-recreate: o Compose recria só os serviços cuja imagem ou
    #    configuração mudou. --wait = detached + espera ficar running/healthy.
    if ! docker compose up --build --remove-orphans --wait; then
      log "✗ docker compose up falhou — ver: docker compose ps; docker compose logs --tail 100 vllm"
      fail=1
    fi
  fi
fi

# 3) Limpeza, sempre.
# -a: com tags pinadas a imagem antiga não vira dangling após um bump
docker image prune -af
# filtro: não apagar volumes órfãos de outras stacks do host
docker volume prune -f --filter label=com.docker.compose.project=ollama
docker builder prune -f
find /var/log -maxdepth 1 -name 'ollama_update_*.log' -mtime +30 -delete

log "raiz: $(df -h / | awk 'NR==2 {print $5" usado, "$4" livre"}')"

exit $fail
