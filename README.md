
# LLM Stack for Embrapa I/O

Configuração de _deploy_ de uma _stack_ de inferência LLM em GPU Servers do ecossistema do **Embrapa I/O**, combinando:

- **[SGLang](https://github.com/sgl-project/sglang)** nas GPUs — servindo o modelo `Qwen3.5-27B-AWQ` (dense, tool calling, reasoning) com API OpenAI-compatible.
- **[Ollama](https://ollama.com)** em CPU — dedicado aos modelos de _embedding_ (bge-m3, nomic-embed-text, mxbai-embed-large, etc.), aproveitando o AVX-512.

## Arquitetura

```
Server (dual Xeon Gold 6254, 256 GB RAM, 2× Quadro RTX 6000 24 GB)
├── GPU 0 ──┐
│           ├── SGLang TP=2: Qwen3.5-27B-AWQ dense (~15 GB, 128K ctx, 6 slots)
├── GPU 1 ──┘   http://<host>:${PORT_SGLANG}/v1  (porta host direta, OpenAI-compatible)
│
└── CPU ──── Ollama (embeddings via AVX-512)
              http://<host>:${PORT_OLLAMA}      (atrás do nginx → bloqueia /api/pull etc.)
```

## Requisitos

O GPU Server (_bare metal_) precisa ser configurado, preferencialmente, com **Ubuntu Server 24.04**.

Em seguida, é necessário instalar a [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian).

Para testar:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

> ⚠️ **Quadro RTX 6000 (Turing, sm_75) não suporta FP8 nativo.** Apenas quantizações AWQ INT4 ou GGUF funcionam no SGLang desse hardware.

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
# ./download-model.sh QuantTrio/Qwen3.5-27B-AWQ /data/sglang/models/qwen3.6-35b-a3b-awq
```

> O script lê `SGLANG_MODEL_PATH` do `.env` e usa um container `python:3.12-slim` com `huggingface_hub[cli,hf_transfer]`, sem exigir Python no host. Para modelos _gated_, exporte `HF_TOKEN` antes.

### 3. Subir a stack

```bash
docker compose up --force-recreate --build --remove-orphans --wait
```

> 🕐 O **primeiro boot do SGLang leva 10–15 minutos** compilando kernels (DeepGEMM/Triton/FlashInfer). Boots seguintes (com volume `sglang-cache` preservado) levam 1–2 minutos.

## Validação pós-deploy

**SGLang respondendo:**

```bash
curl http://localhost:${PORT_SGLANG}/v1/models
```

**Chat completion (thinking desabilitado):**

```bash
curl http://localhost:${PORT_SGLANG}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b-awq",
    "messages": [{"role":"user","content":"Olá! Diga em uma frase quem você é."}],
    "max_tokens": 100,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

**Ollama embeddings em CPU:**

```bash
curl http://localhost:${PORT_OLLAMA}/api/embeddings \
  -d '{"model": "bge-m3", "prompt": "Embrapa Gado de Corte"}'
```

**Métricas do SGLang:**

```bash
curl -s http://localhost:${PORT_SGLANG}/metrics | grep -E 'running|kv_cache'
```

## Comandos Úteis

### Ollama (embeddings)

Ver LLMs instaladas:

```bash
docker compose exec ollama ollama ls
```

Instalar modelo de embedding:

```bash
docker compose exec ollama ollama pull bge-m3
```

Ver os modelos em: https://ollama.com/search

### SGLang

Logs em tempo real:

```bash
docker compose logs -f sglang
```

Conferir métricas reportadas no boot (`max_total_num_tokens`, `max_running_requests`, `available_gpu_mem`):

```bash
docker compose logs sglang | grep -E 'max_total_num_tokens|max_running_requests|available_gpu_mem'
```

Atualizar o modelo:

```bash
./download-model.sh QuantTrio/<novo-modelo> /data/sglang/models/<novo-modelo>
# Ajustar SGLANG_MODEL_PATH e SGLANG_SERVED_MODEL_NAME em .env
docker compose up -d --force-recreate sglang
```

### Monitoramento de GPUs

```bash
docker run --rm -it --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 watch -n 1 nvidia-smi
```

## Ajuste fino

- **OOM no boot do SGLang** → reduzir `SGLANG_MEM_FRACTION` para `0.85`, `SGLANG_CONTEXT_LENGTH` para `65536` ou ativar CPU offload (`SGLANG_CPU_OFFLOAD_GB=20`).
- **Modelos MoE + AWQ em Turing (sm_75)**: o kernel AWQ-MoE fundido não existe; os experts caem em FP16 dequantizado e estouram a VRAM. Evitar variantes como `Qwen3.5-35B-A3B-AWQ` ou `Qwen3.6-35B-A3B-AWQ` — preferir dense.
- **Multimodal**: `Qwen3.5-27B-AWQ` é text-only. Para modelos VL (vision-language), adicionar `--enable-multimodal` ao command do SGLang.
- **Thinking mode** é controlado _por request_ via `chat_template_kwargs.enable_thinking` no cliente — **não** há flag de servidor para desligar globalmente.
- Cache de compilação do SGLang vive no _volume_ nomeado `sglang-cache`; preservá-lo entre deploys é essencial.
