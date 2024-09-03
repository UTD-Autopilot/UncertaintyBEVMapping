from .carla import compile_data as compile_data_carla
from .nuscenes import compile_data as compile_data_nuscenes

datasets = {
    'nuscenes': compile_data_nuscenes,
    'carla': compile_data_carla,
}
