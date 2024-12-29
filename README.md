# Uncertainty BEV Mapping

### PointBEV Setup

Create a 'data' folder in the repo directory, download efficinet weights and put it there:

```
mkdir data
cd data
wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth
wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
cd ..
```

Enable your virtualenv/conda environment, install CUDA toolkits and pytorch, then build the required modules:

```
cd uncertainty_bev_mapping/bev_models/backbones/pointbev/ops/

cd defattn/
chmod +x make.sh
./make.sh

cd ..
cd gs
pip install -e .
```
