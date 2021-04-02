export MODELS_DIR=models
export IMG_DIR=images

mkdir -p $MODELS_DIR
mkdir -p $IMG_DIR

python -m nltk.downloader stopwords
python -m nltk.downloader words
python -m nltk.downloader punkt
python -m nltk.downloader wordnet

# Demo image
wget --output-document $IMG_DIR/demo.png https://raw.githubusercontent.com/DevashishPrasad/CascadeTabNet/master/Demo/demo.png

rm -rf $MODELS_DIR && mkdir $MODELS_DIR && \
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
