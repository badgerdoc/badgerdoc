pip install -r .requirements.txt -f https://download.openmmlab.com/mmcv/dist/index.html

mkdir mmcv && wget -P mmcv https://download.openmmlab.com/mmcv/dist/1.2.1/torch1.7.0/cpu/mmcv_full-1.2.1%2Btorch1.7.0%2Bcpu-cp38-cp38-manylinux1_x86_64.whl
pip install mmcv/mmcv_full-1.2.1+torch1.7.0+cpu-cp38-cp38-manylinux1_x86_64.whl

git clone --branch v2.7.0 'https://github.com/open-mmlab/mmdetection.git'
cd mmdetection
python setup.py install
cd -
rm -rf mmdetection
