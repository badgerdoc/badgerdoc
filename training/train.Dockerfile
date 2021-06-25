FROM nvidia/cuda:11.0-devel-ubuntu20.04

# ------ eReader base ------
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --yes locales build-essential python3-dev git\
    python3-distutils python3-pip \
    cmake wget curl && rm -rf /var/lib/apt/lists/*

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf && \
    locale-gen en_US.UTF-8

RUN pip3 install torch==1.7.0 torchvision==0.8.1 gdown click

RUN mkdir /mmcv && wget -P /mmcv https://download.openmmlab.com/mmcv/dist/1.2.1/torch1.7.0/cpu/mmcv_full-1.2.1%2Btorch1.7.0%2Bcpu-cp38-cp38-manylinux1_x86_64.whl

RUN git clone --branch v2.7.0 'https://github.com/open-mmlab/mmdetection.git' /mmdetection && \
    cd /mmdetection && \
    python3 /mmdetection/setup.py install && \
    cd -

RUN mkdir /models && \
    gdown "https://drive.google.com/uc?id=1YmO5O8kBPI9XZWASTWqP1Qh4skqQu7US" -O /models/3_cls_w18_e30.pth

ENV CASCADE_MODEL_PATH="/models/3_cls_w18_e30.pth"

RUN mkdir /mmcv2 && wget -P /mmcv2 https://download.openmmlab.com/mmcv/dist/1.2.2/torch1.7.0/cu110/mmcv_full-1.2.2%2Btorch1.7.0%2Bcu110-cp38-cp38-manylinux1_x86_64.whl
RUN pip3 uninstall mmcv
RUN pip3 install /mmcv2/mmcv_full-1.2.2+torch1.7.0+cu110-cp38-cp38-manylinux1_x86_64.whl

COPY . /table-extractor
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/table-extractor"
ENV CUDA_HOME="/usr/local/cuda"
ENV FORCE_CUDA="1"

WORKDIR /table-extractor

CMD ["/bin/bash"]
