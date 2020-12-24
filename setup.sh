pip install -r .requirements.txt -f https://download.openmmlab.com/mmcv/dist/index.html

git clone --branch v2.7.0 'https://github.com/open-mmlab/mmdetection.git'
cd mmdetection
python setup.py install
cd -
rm -rf mmdetection
