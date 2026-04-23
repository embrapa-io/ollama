
# LLM Stack for Embrapa I/O

Configuração de _deploy_ de uma _stack_ de inferência LLM em GPU Servers do ecossistema do **Embrapa I/O**, combinando:

- **[SGLang](https://github.com/sgl-project/sglang)** nas GPUs — servindo **`Qwen/Qwen2.5-VL-32B-Instruct-AWQ`** (dense, vision-language, tool calling) com API OpenAI-compatible. Lê screenshots (Chrome DevTools, UIs) e responde texto.
- **[Ollama](https://ollama.com)** em CPU — duas funções:
  - _embeddings_ (bge-m3, nomic-embed-text, mxbai-embed-large, …) aproveitando AVX-512
  - **`qwen3.6:35b-a3b`** (MoE, 3 B ativos) como chat/agentic interino enquanto o AWQ oficial do Qwen3.6 não sai (ver **Roadmap** abaixo)

## Arquitetura

```
Server (dual Xeon Gold 6254, 256 GB RAM, 2× Quadro RTX 6000 24 GB)
├── GPU 0 ──┐
│           ├── SGLang TP=2: Qwen2.5-VL-32B-Instruct-AWQ (~19 GB, 64K ctx, VL)
├── GPU 1 ──┘   http://<host>:${PORT_SGLANG}/v1  (porta host direta, OpenAI-compatible)
│
└── CPU ──── Ollama (AVX-512, 72 threads)
              http://<host>:${PORT_OLLAMA}      (atrás do nginx → bloqueia /api/pull etc.)
              • embeddings
              • qwen3.6:35b-a3b (chat/agentic interino)
```

## Requisitos

O GPU Server (_bare metal_) precisa ser configurado, preferencialmente, com **Ubuntu Server 24.04**.

Em seguida, é necessário instalar a [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian).

Para testar:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

> ⚠️ **Quadro RTX 6000 (Turing, sm_75) não suporta FP8 nativo nem BF16.** Apenas AWQ INT4 ou GGUF rodam. Vários kernels modernos (CUTLASS DSL, Gated DeltaNet) também exigem sm_80+ — isso restringe quais modelos bootam no SGLang neste hardware (ver **Roadmap**).

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
# ./download-model.sh Qwen/Qwen2.5-VL-32B-Instruct-AWQ /data/sglang/models/qwen2.5-vl-32b-instruct-awq
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
    "model": "qwen2.5-vl-32b-instruct-awq",
    "messages": [{"role":"user","content":"Olá! Diga em uma frase quem você é."}],
    "max_tokens": 100
  }'
```

**Vision (screenshot inline):**

```bash
curl http://localhost:${PORT_SGLANG}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-vl-32b-instruct-awq",
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

- **OOM no boot do SGLang** → reduzir `SGLANG_MEM_FRACTION` (0.90 → 0.85), `SGLANG_CONTEXT_LENGTH` (65536 → 32768) ou ativar CPU offload (`SGLANG_CPU_OFFLOAD_GB=20`).
- **Context até 128K com YaRN** — adicionar em `SGLANG_EXTRA_ARGS`:
  ```
  --json-model-override-args '{"rope_scaling":{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
  ```
  E reduzir `SGLANG_MAX_RUNNING_REQUESTS` para 2.
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

**O que funciona confirmadamente em sm_75 via SGLang:**
- `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (atual)
- Famílias Qwen 2.5 / QwQ em geral (`qwen2.py` / `qwen2_vl.py`)

## Roadmap

### Quando Qwen3.6 AWQ oficial sair

O time Qwen deve publicar [`Qwen/Qwen3.6-27B-AWQ`](https://huggingface.co/Qwen/Qwen3.6-27B) nas próximas semanas (hoje só existe BF16 e FP8). Monitorar:

- **Google Alert**: `"Qwen3.6" AWQ`
- **HuggingFace Hub**: seguir [Qwen](https://huggingface.co/Qwen)
- **r/LocalLLaMA** e **GitHub releases do SGLang**

Quando sair, tentar trocar. Mesmo assim, provavelmente continuará bloqueado em Turing enquanto o SGLang não adicionar fallback triton para `Qwen3_5GatedDeltaNet` e para o visual encoder do `qwen3_vl.py`.

### Quando o servidor for trocado por Ampere+ (sm_80+)

- Esvaziar `SGLANG_EXTRA_ARGS` no `.env` (remover `--disable-cuda-graph`, `--attention-backend triton`, `--sampling-backend pytorch`) — recupera path nativo flashinfer + CUDA graph.
- Revisar `SGLANG_MEM_FRACTION` (pode subir) e contexto (128K passa a caber).
