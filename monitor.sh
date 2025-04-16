#!/bin/bash

docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 watch -n 1 nvidia-smi
