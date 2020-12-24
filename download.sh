export MODELS_DIR=models
export IMG_DIR=images

mkdir $MODELS_DIR
mkdir $IMG_DIR

# Model configured to be used with the mmdetection>=2.0.0
gdown "https://drive.google.com/uc?id=1EsrTmKm5_Px2XpDMUiERkv0HSEfstRVg" -O $MODELS_DIR/epoch_41_acc_94_mmd_v2.pth

# Config for the model
wget --output-document $MODELS_DIR/cascadetabnet_config.py https://gist.githubusercontent.com/EgorOs/6bc38bc9b4c7b9eb6dbe0b9cd4ab2915/raw/e511f1e488046da173e5061b0476d11244c03a47/gistfile1.txt

# Demo image
wget --output-document $IMG_DIR/demo.png https://raw.githubusercontent.com/DevashishPrasad/CascadeTabNet/master/Demo/demo.png

cd paddle_detector && ./download.sh