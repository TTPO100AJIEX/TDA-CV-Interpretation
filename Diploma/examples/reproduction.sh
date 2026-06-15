set -x

python3.10 -m venv env
source env/bin/activate
python3.10 -m pip install --no-cache-dir cv-tda cv-tda[classification] cv-tda[facerecognition] cv-tda[autoencoder] cv-tda[segmentation]
python3.10 -m pip install --no-cache-dir -e git+https://github.com/c-hofer/torchph.git@master#egg=torchph
python3.10 reproduction.py --quick --clean > quick.txt
