#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
DOCKER_REPO=dengdan/tensorflow-gpu
VERSION=tf2.0 #py36 #latest
ARCH=$(uname -m)
DOCKER_HOME="/root"
DATE=$(date +%F)
IMG=${DOCKER_REPO}:$VERSION

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_tensorflow2_gpu"
fi


PYLIB_PATH="$(pwd)/.."
function local_volumes() {
  case "$(uname -s)" in
    Linux)
      volumes="${volumes} -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                          -v /media:/media \
                          -v $HOME/.ssh:${DOCKER_HOME}/.ssh \
                          -v /onboard_data:/onboard_data \
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

function add_user() {
  add_script="addgroup --gid ${GRP_ID} ${GRP} && \
      adduser --disabled-password --gecos '' ${USER} \
        --uid ${USER_ID} --gid ${GRP_ID} 2>/dev/null && \
      usermod -aG sudo ${USER} && \
      echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
      cp -r /etc/skel/. /home/${USER} && \
      chsh -s /usr/bin/zsh ${USER} && \
      chown -R ${USER}:${GRP} '/home/${USER}'"
  echo "${add_script}"
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
        -p :2234:22 \
        -p :7086:6060 \
        --hostname $DOCKER_NAME \
        --shm-size 2G \
        --security-opt seccomp=unconfined \
        $IMG 
        
  
    docker exec ${DOCKER_NAME} service ssh start
    if [ "${USER}" != "root" ]; then
        docker exec ${DOCKER_NAME} bash -c "$(add_user)"
    fi

    if [ -z "$(command -v nvidia-smi)" ]; then
        docker exec ${DOCKER_NAME} ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
    fi
      
    #docker exec -u $USER ${DOCKER_NAME} sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
    docker cp -L ~/.gitconfig ${DOCKER_NAME}:${DOCKER_HOME}/.gitconfig
    docker cp -L ~/.vimrc ${DOCKER_NAME}:${DOCKER_HOME}/.vimrc
    docker cp -L ~/.vim ${DOCKER_NAME}:${DOCKER_HOME}
    docker exec -d -u ${USER} ${DOCKER_NAME} pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
 }

main
