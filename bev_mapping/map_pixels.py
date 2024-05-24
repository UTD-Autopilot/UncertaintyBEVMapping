import os
import pickle
import json
from PIL import Image, ImageDraw
import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import Delaunay
import tqdm

DEPTH_CAMERA_FAR_PLANE = 1000.00

def transform_to_matrix(x, y, z, roll, pitch, yaw):
    # see https://msl.cs.uiuc.edu/planning/node102.html
    alpha, beta, gamma = np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll)
    transformation_matrix = np.array([
		[(np.cos(alpha)*np.cos(beta)), (np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)), (np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)), x],
		[(np.sin(alpha)*np.cos(beta)), (np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)), (np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)), y],
		[(-np.sin(beta))             , (np.cos(beta)*np.sin(gamma))                                            , (np.cos(beta) * np.cos(gamma))                                          , z],
		[0,0,0,1]
    ])
    return transformation_matrix

def get_intrinsic_matrix(image_size_y, image_size_x, fov):
    focal_length = image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))
    intrinsic = np.identity(3)
    intrinsic[0, 2] = image_size_x / 2.0
    intrinsic[1, 2] = image_size_y / 2.0
    intrinsic[0, 0] = intrinsic[1, 1] = focal_length
    return intrinsic

def pixels_to_point_cloud(depth: np.ndarray, intrinsic: np.ndarray):
    image_size_y, image_size_x = depth.shape[0], depth.shape[1]
    image_xy = np.mgrid[0:image_size_x, 0:image_size_y].transpose(2, 1, 0)
    image_xyz = np.concatenate([image_xy, np.ones((image_size_y, image_size_x, 1))], axis=-1)

    camera_xyz = np.matmul(np.linalg.inv(intrinsic), image_xyz.reshape(-1, 3).T)
    camera_xyz *= depth.reshape(-1)
    camera_xyz = camera_xyz.T

    return camera_xyz

def points_to_bev(points: np.ndarray, payload: np.ndarray, bev_range, bev_size_y=500, bev_size_x=500, subsample_rate=0.1, save_suffix=''):
    points = points.copy()[..., 0:2]
    points_not_in_range = np.logical_or.reduce([points[..., 0]>bev_range, points[..., 1]>bev_range, points[..., 0]<-bev_range, points[..., 1]<-bev_range])

    points = np.delete(points, points_not_in_range, axis=0)
    payload = np.delete(payload, points_not_in_range, axis=0)
    
    # # flip image
    t = points[..., 1].copy()
    points[..., 1] = -points[..., 0]
    points[..., 0] = t

    pixel_coords_x = ((points[..., 0] + bev_range) / (2 * bev_range) * bev_size_x).astype(np.int32)
    pixel_coords_y = ((points[..., 1] + bev_range) / (2 * bev_range) * bev_size_y).astype(np.int32)
    pixel_coords = np.stack([pixel_coords_x, pixel_coords_y], axis=-1)

    img = np.zeros((bev_size_y, bev_size_x, 4), dtype=np.uint8)
    color_rgba = np.concatenate([payload, np.full((*payload.shape[:-1], 1), 255)], axis=-1)
    img[pixel_coords_y, pixel_coords_x] = color_rgba

    subsample_interval = int(1.0/subsample_rate)

    points = points[::subsample_interval]
    payload = payload[::subsample_interval]
    pixel_coords = pixel_coords[::subsample_interval]
    triang = Delaunay(points).simplices

    bg = Image.new('RGB', (bev_size_y, bev_size_x))
    draw = ImageDraw.Draw(bg)

    for triangle in triang:
        draw.polygon([tuple(pixel_coords[triangle[0]]), tuple(pixel_coords[triangle[1]]), tuple(pixel_coords[triangle[2]])], fill = tuple(payload[triangle[0]].astype(np.int32)))

    bg.save(f'temp/triangles_{save_suffix}.png')
    fg = Image.fromarray(img, 'RGBA')
    x, y = ((bg.width - fg.width) // 2 , (bg.height - fg.height) // 2)
    bg.paste(fg, (x, y), fg)

    fg.save(f'temp/pixels_{save_suffix}.png')
    bg.save(f'temp/bev_{save_suffix}.png')
    return np.array(bg)

def to_right_handed(points: np.ndarray):
    points = points.copy()
    points[..., 1] = -points[..., 1]
    return points

def bev_map_frame(agent_path, frame):
    rr.set_time_sequence(f'frame', frame)
    sensor_info_path = os.path.join(agent_path, f'sensors.json')
    with open(sensor_info_path, 'r') as f:
        sensor_info = json.load(f)
    
    # Load lidar points for reference. We don't use these for the mapping
    lidar_path = os.path.join(agent_path, f'lidar/{frame}.bin')
    with open(lidar_path, 'rb') as f:
        data = f.read()
    lidar_points = np.frombuffer(data, dtype=np.dtype('f4'))
    lidar_points = np.reshape(lidar_points, (int(lidar_points.shape[0] / 4), 4))
    lidar_info = sensor_info['sensors']['lidar']
    loc = lidar_info['transform']['location']
    rot = lidar_info['transform']['rotation']
    x, y, z, roll, pitch, yaw = loc[0], loc[1], loc[2], rot[0], rot[1], rot[2]
    extrinsic = transform_to_matrix(x, y, z, roll, pitch, yaw)
    points_xyz1 = np.concatenate([lidar_points[..., :3], np.ones((*lidar_points.shape[:-1], 1))], axis=-1)
    lidar_points = np.matmul(extrinsic, points_xyz1.T).T[:, :3]

    rr.log(f'lidar', rr.Points3D(to_right_handed(lidar_points)))

    all_points = []
    all_semantic_pixels = []
    
    cameras = ['front_camera', 'left_front_camera', 'left_back_camera', 'back_camera', 'right_back_camera', 'right_front_camera']
    semantic_cameras = ['front_semantic_camera', 'left_front_semantic_camera', 'left_back_semantic_camera', 'back_semantic_camera', 'right_back_semantic_camera', 'right_front_semantic_camera']
    depth_cameras = ['front_depth_camera', 'left_front_depth_camera', 'left_back_depth_camera', 'back_depth_camera', 'right_back_depth_camera', 'right_front_depth_camera']
    # cameras = ['front_camera', 'back_camera',]
    # depth_cameras = ['front_depth_camera', 'back_depth_camera',]
    for camera, semantic_camera, depth_camera in zip(cameras, semantic_cameras, depth_cameras):
        camera_info = sensor_info['sensors'][camera]
        image_size_x = camera_info['sensor_options']['image_size_x']
        image_size_y = camera_info['sensor_options']['image_size_y']
        fov = camera_info['sensor_options']['fov']

        loc = camera_info['transform']['location']
        rot = camera_info['transform']['rotation']
        x, y, z, roll, pitch, yaw = loc[0], loc[1], loc[2], rot[0], rot[1], rot[2]

        intrinsic = get_intrinsic_matrix(image_size_y, image_size_x, fov)
        extrinsic = transform_to_matrix(x, y, z, roll, pitch, yaw)

        img_path = os.path.join(agent_path, f'{camera}/{frame}.png')
        semantic_path = os.path.join(agent_path, f'{semantic_camera}/{frame}.png')
        depth_path = os.path.join(agent_path, f'{depth_camera}/{frame}.pkl')

        with Image.open(img_path) as pil_img:
            pixels = np.array(pil_img)
        
        with Image.open(semantic_path) as pil_img:
            semantic_pixels = np.array(pil_img)
        
        with open(depth_path, 'rb') as f:
            depth = pickle.load(f)

        camera_xyz = pixels_to_point_cloud(depth, intrinsic)
        camera_xyz1 = np.concatenate([camera_xyz, np.ones((*camera_xyz.shape[:-1], 1))], axis=-1).T

        # Camera space to Unreal space, see https://github.com/carla-simulator/carla/issues/553
        adjustment_matrix = transform_to_matrix(0, 0, 0, -90, 0, 90)
        adjusted_xyz1 = camera_xyz1
        adjusted_xyz1[2] = -adjusted_xyz1[2]
        adjusted_xyz1 = np.matmul(adjustment_matrix, adjusted_xyz1)
        vehicle_xyz = np.matmul(extrinsic, adjusted_xyz1).T[:, :3]

        max_depth = 200.0
        valid_pixels_mask = (depth.reshape(-1)>=max_depth)
        vehicle_xyz = np.delete(vehicle_xyz, valid_pixels_mask, axis=0)
        pixels = np.delete(pixels.reshape(-1, 3), valid_pixels_mask, axis=0)
        semantic_pixels = np.delete(semantic_pixels.reshape(-1, 3), valid_pixels_mask, axis=0)

        all_points.append(vehicle_xyz)
        all_semantic_pixels.append(semantic_pixels)

        subsample_interval = 10
        rr.log(f'camera/{camera}/points_xyz', rr.Points3D(to_right_handed(vehicle_xyz.reshape(-1, 3)[::subsample_interval]), colors=pixels.reshape(-1, 3)[::subsample_interval]/255.0))
        rr.log(f'semantic_camera/{semantic_camera}/points_xyz', rr.Points3D(to_right_handed(vehicle_xyz.reshape(-1, 3)[::subsample_interval]), colors=semantic_pixels.reshape(-1, 3)[::subsample_interval]/255.0))

    

    all_points = np.concatenate(all_points, axis=0)
    all_semantic_pixels = np.concatenate(all_semantic_pixels, axis=0)

    bev_image = points_to_bev(all_points, all_semantic_pixels, bev_range=50.0, save_suffix=f'{frame}')
    rr.log(f'bev_semantic_map/reconstruct', rr.ImageEncoded(path=f'temp/bev_{frame}.png'))
    rr.log(f'bev_semantic_map/pixels', rr.ImageEncoded(path=f'temp/pixels_{frame}.png'))
    rr.log(f'bev_semantic_map/triangles', rr.ImageEncoded(path=f'temp/triangles_{frame}.png'))
    rr.log(f'bev_semantic_map/gt', rr.ImageEncoded(path=os.path.join(agent_path, 'birds_view_semantic_camera', f'{frame}.png')))
    rr.log(f'bev_range', rr.Boxes3D(centers=[0, 0, 0], half_sizes=[50.0, 50.0, 1.0]))

if __name__ == '__main__':
    rr.init('map_pixels_debug')
    dataset_path = '../datasets/carla/depth_test_2'
    agent_path = os.path.join(dataset_path, 'agents/0')
    frames = list(map(lambda s: int(s.split('.')[0]), os.listdir(os.path.join(agent_path, 'front_camera'))))

    for frame in tqdm.tqdm(frames):
        bev_map_frame(agent_path, frame)

    rr.save('map_pixels_debug_2.rrd')

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # points = image_xyz.reshape(-1, 3)[::100]
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.savefig('image_xyz.png')
    # plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # points = vehicle_xyz.reshape(-1, 3)[::100]
    # color = pixels.reshape(-1, 3)[::100] / 255.0
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # # ax.set_xlim(-100, 100)
    # # ax.set_ylim(-100, 100)
    # # ax.set_zlim(-100, 100)
    # plt.savefig('vehicle_xyz.png')
    # plt.close()
