#!/usr/bin/env bash
# Monitoramento de GPU: DCGM exporter + coleta pelo Grafana Alloy.
#
# Publica as métricas do NVIDIA DCGM em localhost:9400 e ensina o Alloy já
# instalado no host a coletá-las, de modo que utilização, memória, temperatura e
# potência de cada GPU cheguem ao Prometheus da plataforma com o mesmo rótulo
# 'instance' das demais métricas do host.
#
# Idempotente: pode ser reexecutado à vontade. Só recria o container quando a
# imagem muda e só toca no Alloy quando o bloco ainda não existe.
#
# Uso: sudo bash dcgm.sh
set -euo pipefail

IMAGE="${DCGM_IMAGE:-nvidia/dcgm-exporter:4.6.0-4.8.3-distroless}"
NAME="${DCGM_NAME:-dcgm-exporter}"
PORT="${DCGM_PORT:-9400}"
ALLOY_CONFIG="${ALLOY_CONFIG:-/etc/alloy/config.alloy}"

step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
fail() { echo "ERRO: $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || fail "execute como root (sudo bash dcgm.sh)"

step "Pré-requisitos"
command -v docker >/dev/null || fail "docker não encontrado — rode antes o setup/provision.sh"
command -v nvidia-smi >/dev/null || fail "nvidia-smi não encontrado — driver NVIDIA ausente"
docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia \
    || fail "runtime 'nvidia' não configurado no Docker — rode 'nvidia-ctk runtime configure --runtime=docker'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /'

step "DCGM exporter em localhost:${PORT}"
docker pull -q "$IMAGE" >/dev/null

DESIRED="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
CURRENT="$(docker inspect "$NAME" --format '{{.Image}}' 2>/dev/null || true)"

if [ "$CURRENT" != "$DESIRED" ]; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true

    # A porta é publicada apenas em loopback: quem coleta é o Alloy, no próprio host.
    # SYS_ADMIN é exigido pelo DCGM para ler os contadores de perfilamento da GPU.
    docker run -d \
        --name "$NAME" \
        --restart unless-stopped \
        --runtime nvidia \
        --gpus all \
        --cap-add SYS_ADMIN \
        --publish "127.0.0.1:${PORT}:9400" \
        "$IMAGE" >/dev/null

    echo "  container recriado a partir de ${IMAGE}"
else
    docker start "$NAME" >/dev/null 2>&1 || true
    echo "  container já está na imagem desejada"
fi

step "Aguardando as métricas responderem"
for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/metrics" >/dev/null 2>&1; then
        echo "  ok em ${i}s"
        break
    fi
    [ "$i" -eq 30 ] && fail "o exporter não respondeu em 30s — veja 'docker logs ${NAME}'"
    sleep 1
done

curl -fsS "http://127.0.0.1:${PORT}/metrics" | grep -c '^DCGM_FI_' | sed 's/^/  métricas DCGM expostas: /'

step "Coleta pelo Grafana Alloy"

if [ ! -f "$ALLOY_CONFIG" ]; then
    echo "  ${ALLOY_CONFIG} não existe — Alloy não instalado neste host."
    echo "  O exporter segue no ar em localhost:${PORT}; configure a coleta quando o Alloy for instalado."
    echo "DCGM-OK"
    exit 0
fi

if grep -q 'prometheus.scrape "dcgm"' "$ALLOY_CONFIG"; then
    echo "  bloco já presente em ${ALLOY_CONFIG}"
else
    # Encaminha para o mesmo receptor já usado pelo node_exporter, para que as
    # métricas de GPU carreguem os mesmos rótulos das demais métricas do host.
    if grep -q 'prometheus.relabel "add_ip"' "$ALLOY_CONFIG"; then
        RECEIVER="prometheus.relabel.add_ip.receiver"
    elif grep -q 'prometheus.remote_write "default"' "$ALLOY_CONFIG"; then
        RECEIVER="prometheus.remote_write.default.receiver"
    else
        fail "não encontrei em ${ALLOY_CONFIG} nem 'prometheus.relabel \"add_ip\"' nem 'prometheus.remote_write \"default\"' — ajuste o destino manualmente"
    fi

    cp -a "$ALLOY_CONFIG" "${ALLOY_CONFIG}.bak-$(date +%Y%m%d%H%M%S)"

    cat >> "$ALLOY_CONFIG" <<EOF

// Métricas das GPUs, expostas pelo DCGM exporter em localhost:${PORT}.
// O rótulo 'instance' recebe o hostname para casar com as demais métricas do host.
prometheus.scrape "dcgm" {
  targets = [{
    __address__ = "127.0.0.1:${PORT}",
    instance    = constants.hostname,
    job         = "dcgm",
  }]
  forward_to      = [${RECEIVER}]
  scrape_interval = "30s"
}
EOF

    echo "  bloco acrescentado, encaminhando para ${RECEIVER}"

    if command -v alloy >/dev/null && ! alloy fmt "$ALLOY_CONFIG" >/dev/null 2>&1; then
        fail "a configuração do Alloy ficou inválida — restaure o backup ao lado de ${ALLOY_CONFIG}"
    fi

    systemctl reload alloy 2>/dev/null || systemctl restart alloy 2>/dev/null || \
        echo "  aviso: recarregue o Alloy manualmente (systemctl reload alloy)"
fi

step "Concluído"
echo "  exporter:  http://127.0.0.1:${PORT}/metrics"
echo "  no Grafana: DCGM_FI_DEV_GPU_UTIL{instance=\"$(hostname)\"}"
echo "DCGM-OK"
