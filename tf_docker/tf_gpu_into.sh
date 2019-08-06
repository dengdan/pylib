#!/usr/bin/env bash

SHELL=zsh
if [ "$1" = "bash" ]; then
  SHELL=bash
fi
DOCKER_NAME="${USER}_tensorflow_gpu"

xhost +local:${USER} 1>/dev/null 2>&1
docker exec \
    -u ${USER} \
    -it $DOCKER_NAME \
    /bin/$SHELL
xhost -${USER} 1>/dev/null 2>&1
