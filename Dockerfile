FROM python:3.8

ENV MODELS_DIR=models
ENV IMG_DIR=images
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        curl \
        tar \
        gzip \
        git \
        # python3 \
        # python3-pip \
        python3-opencv \
        libgl1-mesa-glx \
        tesseract-ocr-all \
        pkg-config \
        libtesseract-dev \
        libleptonica-dev \
        poppler-utils libpoppler-cpp-dev poppler-data \
        cmake make g++ python3-dev wget default-jre libreoffice libreoffice-java-common \
        && apt-get autoremove -fy \
        && apt-get clean \
        && apt-get autoclean -y

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r .requirements.txt 

RUN pip install mmcv-full==1.2.1 -f https://download.openmmlab.com/mmcv/dist/index.html
RUN pip install 'git+https://github.com/open-mmlab/mmdetection.git@v2.7.0'

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader words
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

RUN mkdir $MODELS_DIR && \
    mkdir $IMG_DIR && \
    gdown "https://drive.google.com/uc?id=17Xtqh3X9_Hu4BWFTiJpgmYipouFMy0sG" -O $MODELS_DIR/epoch_20_w18.pth && \
    wget --output-document $MODELS_DIR/ch_ppocr_mobile_v2.0_det_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar && \
    tar xf $MODELS_DIR/ch_ppocr_mobile_v2.0_det_infer.tar -C $MODELS_DIR && \
    rm -rf $MODELS_DIR/ch_ppocr_mobile_v2.0_det_infer.tar && \
    wget --output-document $MODELS_DIR/ch_ppocr_server_v2.0_det_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar && \
    tar xf $MODELS_DIR/ch_ppocr_server_v2.0_det_infer.tar -C $MODELS_DIR && \
    rm -rf $MODELS_DIR/ch_ppocr_server_v2.0_det_infer.tar && \
    wget --output-document $MODELS_DIR/ch_ppocr_mobile_v2.0_cls_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar && \
    tar xf $MODELS_DIR/ch_ppocr_mobile_v2.0_cls_infer.tar -C $MODELS_DIR && \
    rm -rf $MODELS_DIR/ch_ppocr_mobile_v2.0_cls_infer.tar
