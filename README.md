# Uncertainty BEV Mapping

## Setup

Create a virtual environment with your favorite tools and run `pip install -r requirments.txt`. Then run `pip install -e .` to setup this repo.

### PointBEV Setup (Optional)

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

## Training

Entrypoint scripts are in the `scripts` folder. To replicate the top-k results, use `scripts/train_bev_topk2_model.py`.

Before running the script, you'll need to download the NuScenes dataset into `../../Datasets/`.

The parameters are controlled by the configuration files in `configs/topk2`. For example, to run the simplebev experiments on NuScenes dataset, run `python scripts/train_bev_topk2_model.py configs/topk2/nuscenes_simplebev_evidential_topk2_vehicle.yaml`.

To replicate the CARLA experiments, you'll need to compile a custom version of CARLA. Follow the official instructions from Carla, but instead of using the official version please use this version instead: [carla with ood objects support](https://github.com/UTD-Autopilot/carla). Then setup [UTD-Autopilot](https://github.com/UTD-Autopilot/UTD-Autopilot) and use the `auto_data_collection.py` to run the data collection.
