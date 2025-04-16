
# Ollama for Embrapa I/O

Configuração de _deploy_ do [Ollama](https://ollama.com) (LLM Runtime) em GPU Servers do ecossistema do **Embrapa I/O**.

## Requisitos

O GPU Server (_bare metal_) precisa ser configurado, preferencialmente, com **Ubuntu Server 24.04**.

Em seguida, é necessário instalar a [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian).

Para testar:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Deploy

Configure as variáveis de ambiente: `cp .env.example .env`. Por fim, suba a _stack_ de containers:

```bash
docker compose up --force-recreate --build --remove-orphans --wait
```

## Referências

