FROM nvcr.io/nvidia/cuda:11.5.1-devel-ubuntu20.04
MAINTAINER samuel21119

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y \
    && apt-get install -y tzdata && ln -fs /usr/share/zoneinfo/Asia/Taiwan /etc/localtime && dpkg-reconfigure -f noninteractive tzdata \
    && apt install -y sudo vim wget gcc libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" \
    && conda init bash \
    && . /root/.bashrc \
    && conda create -n mva_team1 -c conda-forge -y python==3.10 \
    && conda activate mva_team1 \
    && pip3 install torch==1.11.0+cu115 torchvision==0.12.0+cu115  -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install openmim timm opencv-python termcolor yacs pyyaml scipy \
    && pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html

RUN echo "conda activate mva_team1" >> /root/.bashrc

WORKDIR /root/MVATeam1

CMD ["/bin/bash"]