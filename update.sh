#!/bin/bash

set -ex

SCRIPT_DIR=$(dirname "$0")

cd "$SCRIPT_DIR"

pwd

git fetch --all

git pull

docker compose pull --ignore-pull-failures --ignore-buildable

docker compose up --build --force-recreate --remove-orphans --wait

docker image prune -f

docker volume prune -f

docker builder prune -f
