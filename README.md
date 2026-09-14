
# LLM Stack for Embrapa I/O

Configuração de _deploy_ de uma _stack_ de inferência LLM em GPU Servers do ecossistema do **Embrapa I/O**, combinando:

- **[vLLM](https://github.com/vllm-project/vllm)** nas GPUs — servindo **`QuantTrio/Qwen3.6-27B-AWQ`** (dense, GDN híbrida, vision-language, tool calling, SWE-Bench Verified 73.4%) com API OpenAI-compatible. Texto, **visão** e tool calling **validados em Turing em 03/jul/2026**. O serviço `sglang` permanece sob profile como referência — em Turing ele só roda text-only (ver **Roadmap**).
- **[Ollama](https://ollama.com)** em CPU — **somente _embeddings_** (decisão de 14/09/2026): `bge-m3`, `qwen3-embedding:0.6b`, `embeddinggemma:300m`, `granite-embedding:278m` e `nomic-embed-text` (mantido a pedido de uma equipe), aproveitando AVX-512. Chat, visão e tools ficam exclusivamente no vLLM; os modelos de chat em CPU foram removidos. A lista publicada no catálogo da plataforma (`io/boilerplate/metadata/clusters.json`, bloco "SEDE 03") deve acompanhar a desta seção.

## Arquitetura

```
Server (dual Xeon Gold 6254, 256 GB RAM, 2× Quadro RTX 6000 24 GB)
├── GPU 0 ──┐
│           ├── vLLM TP=2: Qwen3.6-27B-AWQ (~21 GB, 128K ctx, VL, GDN)
├── GPU 1 ──┘   http://<host>:${PORT_OLLAMA}/v1 (via nginx — rota aberta p/ ecossistema)
│               http://<host>:${PORT_SGLANG}/v1 (porta direta — filtrada pelo firewall)
│
└── CPU ──── Ollama (AVX-512, 72 threads)
              http://<host>:${PORT_OLLAMA}/api/* (nginx → bloqueia /api/pull etc.)
              • embeddings apenas (API nativa: /api/embeddings)
```

> 🔀 **Roteamento do nginx (portas 80 e 11434):** `/v1/*` → **vLLM** (API OpenAI-compatible: chat, visão, tools); todo o resto (`/api/*` etc.) → **Ollama** (API nativa). A **porta 80 é o padrão** (URL sem porta — `http://llm.nuvem.ti.embrapa.br/v1` —, alinhado aos demais GPU Servers da plataforma; liberação de firewall para as VMs do ecossistema solicitada à GTI em 04/08/2026). A **11434 segue funcional** durante a transição — clientes atuais (Open WebUI, n8n "OpenAI Chat Model") com `base_url` `http://llm.nuvem.ti.embrapa.br:11434/v1` e API key dummy continuam operando sem mudança. O firewall da TI filtra a 11435 para as VMs do ecossistema (confirmado em 03/07/2026 a partir da VM n8n); a liberação dela segue pendente como melhoria (diagnóstico).

## Requisitos

O GPU Server (_bare metal_) precisa ser configurado, preferencialmente, com **Ubuntu Server 24.04**.

Em seguida, é necessário instalar a [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian).

Para testar:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

> ⚠️ **Quadro RTX 6000 (Turing, sm_75) não suporta FP8 nativo nem BF16.** Apenas AWQ INT4 ou GGUF rodam. Kernels CUTLASS DSL e FlashInfer exigem sm_80+/sm_90+ — mas o Gated DeltaNet (Qwen3.5/3.6) ganhou _backend_ triton sem piso de _compute capability_ no SGLang v0.5.13+ (`--linear-attn-backend triton`), o que desbloqueou o Qwen3.6-27B-AWQ neste hardware (ver **Roadmap**).

> ⚠️ **CUDA da imagem × driver do host:** as imagens `v0.5.14*` default do SGLang são **CUDA 13.0** e exigem driver **≥ 580** no host (conferir com `nvidia-smi`). No Ubuntu 24.04 o ramo **R580** chega via apt e é o **último com suporte a Turing** — é o driver-alvo deste hardware. Com driver R570 ou anterior, usar a variante `cu129` da imagem (CUDA 12.9 roda em driver ≥ 525 via _minor version compatibility_; pode exigir `NVIDIA_DISABLE_REQUIRE=1`). O build local (`Dockerfile.sglang`) repõe dependências Python faltantes na `-runtime` ([sglang#29650](https://github.com/sgl-project/sglang/issues/29650)).

Para docker compose usar uma network nomeada já existente:

```bash
docker network create ollama
```

## Deploy

### 1. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Ajustar PORT_OLLAMA, PORT_SGLANG, LLM_PATH e SGLANG_MODEL_PATH conforme o host.
```

### 2. Baixar o modelo para o SGLang

```bash
# Antes de qualquer download: o volume de dados está montado?
findmnt /dados || echo 'VOLUME NÃO MONTADO — não baixe nada'

sudo mkdir -p /dados/sglang/models
sudo chown -R $USER:$USER /dados/sglang

./download-model.sh
# ou, para outro repositório/destino:
# ./download-model.sh QuantTrio/Qwen3.6-27B-AWQ /dados/sglang/models/qwen3.6-27b-awq
```

> O script lê `SGLANG_MODEL_REPO` e `SGLANG_MODEL_PATH` do `.env` e roda um container `python:3.12-slim` com `huggingface_hub[hf_xet]`, sem exigir Python no host. Para modelos _gated_ ou rate-limit, exporte `HF_TOKEN` ou coloque no `.env`.
>
> ⚠️ **Trava contra volume desmontado:** se o destino começa por `/dados`, `/data`, `/mnt` ou `/srv` e esse diretório está no mesmo filesystem da raiz, o script **aborta**. Foi o que aconteceu em 11/09/2026 no `llm.nuvem`: o LV do `/dados` não tinha sido ativado no boot, o modelo (21 GB) foi para a raiz de 105 GB e o disco chegou a 94 %. Confira `lsblk`, `sudo lvscan` e `findmnt /dados`; para forçar um destino deliberadamente na raiz, `ALLOW_ROOT_FS=1 ./download-model.sh …`.

### 3. Subir a stack

```bash
docker compose up --force-recreate --build --remove-orphans --wait
```

### 4. Modelos de embedding no Ollama CPU

```bash
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec ollama ollama pull embeddinggemma:300m
docker compose exec ollama ollama pull granite-embedding:278m
docker compose exec ollama ollama pull nomic-embed-text
```

> Não puxar modelos de chat para o Ollama: chat é no vLLM. Ao acrescentar ou remover um embedding, atualizar também o `clusters.json` do `io/boilerplate/metadata` (bloco "SEDE 03 — Turing 48GB", campo `embedding.models`).

> 🕐 O **primeiro boot do SGLang leva 10–15 minutos** compilando kernels (DeepGEMM/Triton/FlashInfer). Boots seguintes (com volume `sglang-cache` preservado) levam 1–2 minutos.

## Validação pós-deploy

**SGLang respondendo:**

```bash
curl http://localhost:${PORT_SGLANG}/v1/models
```

**Chat completion:**

```bash
curl http://localhost:${PORT_SGLANG}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-awq",
    "messages": [{"role":"user","content":"Olá! Diga em uma frase quem você é."}],
    "max_tokens": 100,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

**Vision (screenshot inline):**

```bash
curl http://localhost:${PORT_SGLANG}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-awq",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Resuma o que este screenshot mostra."},
        {"type": "image_url", "image_url": {"url": "https://exemplo/devtools.png"}}
      ]
    }],
    "max_tokens": 300
  }'
```

**Ollama (embeddings):**

```bash
curl http://localhost:${PORT_OLLAMA}/api/embeddings \
  -d '{"model": "bge-m3", "prompt": "Embrapa Gado de Corte"}'

docker compose exec ollama ollama list   # esperado: só os 5 modelos de embedding
```

**Métricas do SGLang:**

```bash
curl -s http://localhost:${PORT_SGLANG}/metrics | grep -E 'running|kv_cache'
```

## Comandos Úteis

### Monitoramento das GPUs

```bash
sudo bash setup/dcgm.sh          # DCGM exporter em localhost:9400 + coleta pelo Alloy
```

Idempotente. Depois de rodar, as métricas aparecem no Grafana da plataforma, por
exemplo em `DCGM_FI_DEV_GPU_UTIL{instance="<hostname>"}`.

### Ollama (CPU)

```bash
docker compose exec ollama ollama ls
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama rm <modelo>   # remover o que não for embedding
```

Modelos em: https://ollama.com/search

### SGLang (GPU)

Logs em tempo real:

```bash
docker compose logs -f sglang
```

Conferir métricas reportadas no boot (`max_total_num_tokens`, `available_gpu_mem`):

```bash
docker compose logs sglang | grep -E 'max_total_num_tokens|max_running_requests|available_gpu_mem'
```

Trocar o modelo:

```bash
./download-model.sh <org/repo> /dados/sglang/models/<pasta>
# Ajustar SGLANG_MODEL_PATH, SGLANG_MODEL_REPO e SGLANG_SERVED_MODEL_NAME em .env
docker compose up -d --force-recreate sglang
```

### Monitoramento de GPUs

```bash
docker run --rm -it --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 watch -n 1 nvidia-smi
```

## Ajuste fino

- **OOM no boot do SGLang** → reduzir `SGLANG_CONTEXT_LENGTH` (131072 → 65536), `SGLANG_MEM_FRACTION` (0.90 → 0.85) ou `SGLANG_MAX_RUNNING_REQUESTS` (6 → 4).
- **Contexto além dos 262K nativos** (até 1M) é possível via YaRN em `rope_parameters` do `text_config` — ver seção _Processing Ultra-Long Texts_ no model card do Qwen3.6-27B. Irrelevante em 2×24 GB.
- **MTP (Multi-Token Prediction)** — o checkpoint traz pesos de MTP (não quantizados, BF16), mas **não habilitar** `--speculative-*` neste hardware: consome VRAM extra e há crash conhecido ([sglang#28431](https://github.com/sgl-project/sglang/issues/28431)).
- **Thinking mode** (reasoning) é controlado _por request_ via `chat_template_kwargs.enable_thinking` no cliente — **não** há flag de servidor para desligar globalmente.
- **Cache de compilação** do SGLang vive no volume nomeado `sglang-cache`; preservá-lo entre deploys é essencial (evita o warmup de 10–15 minutos a cada restart).
- **OpenMP** (`OMP_NUM_THREADS=16`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`) está ajustado pro dual Xeon Gold 6254. Ajustar se o hardware mudar.

## Restrições observadas em Turing (sm_75)

Histórico de tentativas que **não funcionam** neste hardware via SGLang — mantido como referência pra não repetir:

| Modelo | Motivo |
|---|---|
| `Qwen3.6-35B-A3B-AWQ` (QuantTrio) | MoE — kernel AWQ-MoE fundido exige sm_80+. Experts caem em FP16 e estouram a VRAM |
| `Qwen3.5-27B-AWQ` (QuantTrio/cyankiwi) | Arquitetura `qwen3_5.py` com `Qwen3_5GatedDeltaNet` exige sm_80+ |
| `Qwen3.5-35B-A3B-AWQ` | MoE + GatedDeltaNet + VL — triplo bloqueio |
| `Qwen3.6-27B-AWQ-INT4` (cyankiwi) | GatedDeltaNet + VL encoder + scheme compressed-tensors sem fallback em sm_75 |
| `Qwen3.6-27B-FP8` (oficial) | sm_75 não suporta FP8 nativo |
| `Qwen3.6-27B-AWQ` (QuantTrio) **com `--enable-multimodal`** | Vision encoder do `qwen3_vl.py` chama kernels fused do sgl-kernel sem cubins sm_75 → `CUDA error: no kernel image is available` no warmup (v0.5.14, 03/jul/2026) |

> 🚨 **SGLang removeu oficialmente o suporte a Turing** — confirmado pelos mantenedores em [#24224](https://github.com/sgl-project/sglang/issues/24224) e [#24302](https://github.com/sgl-project/sglang/issues/24302) (a documentação que ainda cita sm_75+ está desatualizada). Os wheels novos do `sgl-kernel` não trazem mais cubins sm_75 para todos os kernels fused. O que ainda roda aqui depende de os paths escolhidos (triton) não tocarem nesses kernels — sem garantia de que continue rodando em versões futuras.

> ℹ️ As linhas de GatedDeltaNet refletem o estado de ~abril/2026. Desde a modularização dos _backends_ de atenção linear no SGLang (`--linear-attn-backend triton`, sem piso de _compute capability_ — os pisos duros são só do `flashinfer` SM90+ e `cutedsl` Blackwell), esse bloqueio deixou de ser estrutural. A conv1d do GDN também tem versão triton.

**O que funciona confirmadamente em sm_75 via SGLang:**
- `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (rollback validado)
- Famílias Qwen 2.5 / QwQ em geral (`qwen2.py` / `qwen2_vl.py`)

**Em validação:**
- `QuantTrio/Qwen3.6-27B-AWQ` (alvo atual — ver Roadmap)

## Roadmap

### Qwen3.6-27B-AWQ (QuantTrio) — ✅ validado via vLLM

O AWQ saiu: [`QuantTrio/Qwen3.6-27B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ) (abril/2026, ~21 GiB, AWQ clássico gemm). E as duas condições do bloqueio anterior caíram:

1. **GatedDeltaNet** — o SGLang v0.5.13+ modularizou os _backends_ de atenção linear; o kernel `gdn_triton` (default) não exige sm_80+ e a conv1d usa `causal_conv1d_triton`.
2. **Quantização** — diferente do quant da cyankiwi (compressed-tensors), o da QuantTrio usa AWQ clássico (`quant_method: awq`, gemm), o mesmo _path_ já validado neste servidor com o Qwen2.5-VL-32B. Visual encoder, `q/k/v_proj`, camada 0 e MTP ficam em BF16 (SGLang converte para FP16 em sm_75).

A configuração padrão do repositório (`.env.example`, `docker-compose.yaml`) já aponta para ele: imagem pinada `v0.5.14-runtime`, `--linear-attn-backend triton` explícito, contexto 128K, parser `qwen3_coder`.

**Status da validação em Turing (03/jul/2026) — ✅ concluída via vLLM v0.24.0:**
- ✅ **Texto (GDN + AWQ)** — primeiro forward do `GatedDeltaNet` em sm_75 OK (backend `TRITON_ATTN` auto-selecionado; FA2 rejeitado corretamente por exigir sm_80+).
- ✅ **VISÃO** — leitura de documento em imagem base64 validada. O encoder roda via `TORCH_SDPA` (`MMEncoderAttention`), ops portáveis — a diferença decisiva para o SGLang, cujos kernels fused perderam os cubins sm_75.
- ✅ **Tool calling** — `tool_calls` populado com parser `qwen3_coder`.
- ✅ **Thinking per-request** — `chat_template_kwargs.enable_thinking` funciona; atenção: no vLLM o raciocínio vem no campo **`reasoning`** (no SGLang era `reasoning_content`). Clientes devem ler `message.reasoning`.
- ❌ **SGLang segue inviável para visão em Turing** (crash `no kernel image` no warmup do vision encoder) e não há _downgrade_ possível: o fallback triton do GDN (v0.5.13+, jun/2026) chegou depois da remoção do sm_75 do sgl-kernel (antes de mai/2026) — nenhuma versão tem os dois. O serviço `sglang` fica sob profile como referência text-only:
  ```bash
  docker compose stop vllm && docker compose --profile sglang up -d sglang
  ```

**Se falhar, rollback no `.env`:**

```bash
SGLANG_MODEL_PATH=/data/sglang/models/qwen2.5-vl-32b-instruct-awq
SGLANG_MODEL_REPO=Qwen/Qwen2.5-VL-32B-Instruct-AWQ
SGLANG_SERVED_MODEL_NAME=qwen2.5-vl-32b-instruct-awq
SGLANG_CONTEXT_LENGTH=65536
SGLANG_TOOL_CALL_PARSER=qwen
```

### Quando o servidor for trocado por Ampere+ (sm_80+)

- Esvaziar `SGLANG_EXTRA_ARGS` no `.env` (remover `--disable-cuda-graph`, `--attention-backend triton`, `--sampling-backend pytorch`) — recupera path nativo flashinfer + CUDA graph.
- Revisar `SGLANG_MEM_FRACTION` (pode subir) e contexto (128K passa a caber).
