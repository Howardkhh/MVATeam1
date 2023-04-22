FROM nvcr.io/nvidia/cuda:11.3.0-devel-ubuntu20.04
MAINTAINER samuel21119

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y \
    && apt-get install -y tzdata && ln -fs /usr/share/zoneinfo/Asia/Taiwan /etc/localtime && dpkg-reconfigure -f noninteractive tzdata \
    && apt install -y sudo vim wget gcc libgl1-mesa-glx libglib2.0-0 zip openmpi-bin \
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
    && pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113 \
    && pip install openmim opencv-python termcolor yacs pyyaml scipy gdown \
    && mim install mmcv-full==1.6.0 \
    && pip install timm==0.6.11

RUN echo "conda activate mva_team1" >> /root/.bashrc

WORKDIR /root/MVATeam1

CMD ["/bin/bash"]