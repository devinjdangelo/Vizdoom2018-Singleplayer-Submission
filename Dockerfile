FROM vizdoom/vizdoom:1.1.6-ubuntu16.04-cuda9.0

RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    g++ \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng12-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libzmq3-dev \
    nano \
    nasm \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    locales \
    zlib1g-dev \
    python3-dev \
    python3 \
    python3-pip 

# Python3
RUN pip3 install pip --upgrade

RUN  apt-get install -y net-tools

RUN pip3 install tensorflow
RUN pip3 install imageio
RUN pip3 install scikit-image


# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Enables X11 sharing and creates user home directory
ENV USER_NAME crowdai
ENV HOME_DIR /home/$USER_NAME
#
# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1000
ENV HOST_GID 1000

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

COPY config config
COPY model model
COPY doomfiles doomfiles
COPY random_agent.py .
COPY mock.wad .
COPY run.sh .
COPY EvalArgs.py .
COPY _vizdoom.ini .
COPY 67300.ckpt.data-00000-of-00001 .
COPY 67300.ckpt.index .
COPY 67300.ckpt.meta .
RUN sudo chown ${HOST_UID}:${HOST_GID} -R *
RUN sudo chmod 775 -R *

# Uncomment to use doom2.wad:
COPY doom2.wad /usr/local/lib/python3.5/dist-packages/vizdoom
COPY doom2.wad ./config

ENTRYPOINT ["./run.sh"]
