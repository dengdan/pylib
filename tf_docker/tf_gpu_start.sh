#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
#DOCKER_REPO=dengdan/tensorflow-gpu
VERSION=latest
ARCH=$(uname -m)
DOCKER_HOME="/root"
DATE=$(date +%F)
IMG=gui_test
DOCKER_NAME="${USER}_gui_test"


PYLIB_PATH="$(pwd)/.."
function local_volumes() {
  case "$(uname -s)" in
    Linux)
      volumes="${volumes} -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                          -v /media:/media \
                          -v /home/$USER:/home/$USER\
                          -v /etc/localtime:/etc/localtime:ro \
                          -v /private:/private \
                          -v /data:/data"
      ;;
    Darwin)
      chmod -R a+wr ~/.cache/bazel
      ;;
  esac

  echo "${volumes}"
}

function main(){
    docker ps -a --format "{{.Names}}" | grep "${DOCKER_NAME}" 1>/dev/null
    if [ $? == 0 ]; then
        docker stop ${DOCKER_NAME} 1>/dev/null
        docker rm -f ${DOCKER_NAME} 1>/dev/null
    fi
    local display=""
    if [[ -z ${DISPLAY} ]];then
        display=":0"
    else
        display="${DISPLAY}"
    fi

    USER_ID=$(id -u)
    GRP=$(id -g -n)
    GRP_ID=$(id -g)
    LOCAL_HOST=`hostname`
    if [ -z "$(command -v nvidia-smi)" ]; then
        echo "Nvidia GPU can NOT be used in the docker! Please install the driver first in the host machine if you want to use gpu in the docker."
        CMD="docker"
    else
        CMD="nvidia-docker"
    fi

    eval ${CMD} run -it \
        -d \
        --name ${DOCKER_NAME}\
        -e DISPLAY=$display \
        $(local_volumes) \
        -p :2222:22 \
        -p :6060:6060 \
        --hostname $DOCKER_NAME \
        --shm-size 2G \
        --security-opt seccomp=unconfined \
        $IMG 
        
	
    docker exec ${DOCKER_NAME} service ssh start
      
 }

main
