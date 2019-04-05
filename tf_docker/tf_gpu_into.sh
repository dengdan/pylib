#!/usr/bin/env bash

SHELL=zsh
if [ "$1" = "bash" ]; then
  SHELL=bash
fi
DOCKER_NAME="${USER}_gui_test"

xhost +local:root 1>/dev/null 2>&1
docker exec \
    -u root \
    -it $DOCKER_NAME \
    /bin/$SHELL
xhost -local:root 1>/dev/null 2>&1
