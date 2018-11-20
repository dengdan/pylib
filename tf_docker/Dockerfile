#FROM tensorflow/tensorflow:latest-devel-gpu-py3
FROM dengdan/tensorflow-gpu
RUN apt-get update && apt-get install -y openssh-server
RUN apt-get install -y tmux htop vim zsh git locales libcv-dev
RUN pip install opencv-python matplotlib setproctitle
RUN mkdir /var/run/sshd
RUN echo 'root:1' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
RUN echo "alias p='python'" >> /etc/profile
RUN echo "alias n='nvidia-smi'" >> /etc/profile
RUN echo "alias wn='watch nvidia-smi'" >> /etc/profile
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

