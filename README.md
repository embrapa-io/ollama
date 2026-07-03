
# LLM Stack for Embrapa I/O

Configuração de _deploy_ de uma _stack_ de inferência LLM em GPU Servers do ecossistema do **Embrapa I/O**, combinando:

- **[SGLang](https://github.com/sgl-project/sglang)** nas GPUs — servindo **`QuantTrio/Qwen3.6-27B-AWQ`** (dense, GDN híbrida, vision-language, tool calling, SWE-Bench Verified 73.4%) com API OpenAI-compatible. Em caso de falha em Turing, _rollback_ para `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (ver **Roadmap** abaixo).
- **[Ollama](https://ollama.com)** em CPU — duas funções:
  - _embeddings_ (bge-m3, nomic-embed-text, mxbai-embed-large, …) aproveitando AVX-512
  - **`qwen3.6:35b-a3b`** (MoE, 3 B ativos) como chat/agentic de contingência em CPU

## Arquitetura

```
Server (dual Xeon Gold 6254, 256 GB RAM, 2× Quadro RTX 6000 24 GB)
├── GPU 0 ──┐
│           ├── SGLang TP=2: Qwen3.6-27B-AWQ (~21 GB, 128K ctx, VL, GDN)
├── GPU 1 ──┘   http://<host>:${PORT_SGLANG}/v1  (porta host direta, OpenAI-compatible)
│
└── CPU ──── Ollama (AVX-512, 72 threads)
              http://<host>:${PORT_OLLAMA}      (atrás do nginx → bloqueia /api/pull etc.)
              • embeddings
              • qwen3.6:35b-a3b (chat/agentic de contingência)
```

## Requisitos

O GPU Server (_bare metal_) precisa ser configurado, preferencialmente, com **Ubuntu Server 24.04**.

Em seguida, é necessário instalar a [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian).

Para testar:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

> ⚠️ **Quadro RTX 6000 (Turing, sm_75) não suporta FP8 nativo nem BF16.** Apenas AWQ INT4 ou GGUF rodam. Kernels CUTLASS DSL e FlashInfer exigem sm_80+/sm_90+ — mas o Gated DeltaNet (Qwen3.5/3.6) ganhou _backend_ triton sem piso de _compute capability_ no SGLang v0.5.13+ (`--linear-attn-backend triton`), o que desbloqueou o Qwen3.6-27B-AWQ neste hardware (ver **Roadmap**).

> ⚠️ As imagens `v0.5.14*` do SGLang são **CUDA 13.0** — exigem driver NVIDIA **≥ 580** no host (conferir com `nvidia-smi`). O ramo R580 é também o **último com suporte a Turing**, então ele cobre os dois requisitos. O serviço `sglang` usa build local (`Dockerfile.sglang`) sobre a `-runtime` para repor dependências Python faltantes ([sglang#29650](https://github.com/sgl-project/sglang/issues/29650)).

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
sudo mkdir -p /data/sglang/models
sudo chown -R $USER:$USER /data/sglang

./download-model.sh
# ou, para outro repositório/destino:
# ./download-model.sh QuantTrio/Qwen3.6-27B-AWQ /data/sglang/models/qwen3.6-27b-awq
```

> O script lê `SGLANG_MODEL_REPO` e `SGLANG_MODEL_PATH` do `.env` e roda um container `python:3.12-slim` com `huggingface_hub + hf_transfer`, sem exigir Python no host. Para modelos _gated_ ou rate-limit, exporte `HF_TOKEN` ou coloque no `.env`.

### 3. Subir a stack

```bash
docker compose up --force-recreate --build --remove-orphans --wait
```

### 4. (opcional) Puxar o Qwen3.6 no Ollama CPU

```bash
# Chat/agentic MoE em CPU — ~22 GB, aproveita o AVX-512, ~15–25 tok/s
docker compose exec ollama ollama pull qwen3.6:35b-a3b-q4_K_M

# Embeddings
docker compose exec ollama ollama pull bge-m3
```

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

**Ollama (embeddings e chat):**

```bash
curl http://localhost:${PORT_OLLAMA}/api/embeddings \
  -d '{"model": "bge-m3", "prompt": "Embrapa Gado de Corte"}'

curl http://localhost:${PORT_OLLAMA}/api/generate \
  -d '{"model": "qwen3.6:35b-a3b-q4_K_M", "prompt": "Explique em uma frase o que é Embrapa."}'
```

**Métricas do SGLang:**

```bash
curl -s http://localhost:${PORT_SGLANG}/metrics | grep -E 'running|kv_cache'
```

## Comandos Úteis

### Ollama (CPU)

```bash
docker compose exec ollama ollama ls
docker compose exec ollama ollama run qwen3.6:35b-a3b-q4_K_M
docker compose exec ollama ollama pull bge-m3
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

> ℹ️ As linhas de GatedDeltaNet refletem o estado de ~abril/2026. Desde a modularização dos _backends_ de atenção linear no SGLang (`--linear-attn-backend triton`, sem piso de _compute capability_ — os pisos duros são só do `flashinfer` SM90+ e `cutedsl` Blackwell), esse bloqueio deixou de ser estrutural. A conv1d do GDN também tem versão triton.

**O que funciona confirmadamente em sm_75 via SGLang:**
- `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (rollback validado)
- Famílias Qwen 2.5 / QwQ em geral (`qwen2.py` / `qwen2_vl.py`)

**Em validação:**
- `QuantTrio/Qwen3.6-27B-AWQ` (alvo atual — ver Roadmap)

## Roadmap

### Qwen3.6-27B-AWQ (QuantTrio) — em validação

O AWQ saiu: [`QuantTrio/Qwen3.6-27B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ) (abril/2026, ~21 GiB, AWQ clássico gemm). E as duas condições do bloqueio anterior caíram:

1. **GatedDeltaNet** — o SGLang v0.5.13+ modularizou os _backends_ de atenção linear; o kernel `gdn_triton` (default) não exige sm_80+ e a conv1d usa `causal_conv1d_triton`.
2. **Quantização** — diferente do quant da cyankiwi (compressed-tensors), o da QuantTrio usa AWQ clássico (`quant_method: awq`, gemm), o mesmo _path_ já validado neste servidor com o Qwen2.5-VL-32B. Visual encoder, `q/k/v_proj`, camada 0 e MTP ficam em BF16 (SGLang converte para FP16 em sm_75).

A configuração padrão do repositório (`.env.example`, `docker-compose.yaml`) já aponta para ele: imagem pinada `v0.5.14-runtime`, `--linear-attn-backend triton` explícito, contexto 128K, parser `qwen3_coder`.

**Riscos residuais (só verificáveis no host):**
- Kernels triton do `fla/chunk` (prefill GDN) podem exceder os 64 KB de _shared memory_ do sm_75 → erro `out of resource: shared memory` em runtime. Não há guarda de capability no código — o boot vai tentar.
- Visual encoder do `qwen3_5.py` usa a infra `VisionAttention` (mesma do Qwen2.5-VL, que funciona aqui), mas nunca foi exercitada em Turing com este modelo.

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
