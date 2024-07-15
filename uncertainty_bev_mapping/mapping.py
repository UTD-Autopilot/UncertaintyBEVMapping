import os
import json
from PIL import Image, ImageDraw
import numpy as np
import rerun as rr
from scipy.spatial import Delaunay

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

def triangle_area(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def points_to_bev(points: np.ndarray, payload: np.ndarray, bev_range, bev_size_y=500, bev_size_x=500, subsample_rate=0.1, max_triangle_size=2.0):
    points = points.copy()[..., 0:2]
    points_not_in_range = np.logical_or.reduce([points[..., 0]>bev_range, points[..., 1]>bev_range, points[..., 0]<-bev_range, points[..., 1]<-bev_range])

    points = np.delete(points, points_not_in_range, axis=0)
    payload = np.delete(payload, points_not_in_range, axis=0)
    
    # # flip image
    t = points[..., 1].copy()
    points[..., 1] = -points[..., 0]
    points[..., 0] = t

    # Directly map the points to the bev pixels.
    pixel_coords_x = ((points[..., 0] + bev_range) / (2 * bev_range) * bev_size_x).astype(np.int32)
    pixel_coords_y = ((points[..., 1] + bev_range) / (2 * bev_range) * bev_size_y).astype(np.int32)
    pixel_coords = np.stack([pixel_coords_x, pixel_coords_y], axis=-1)

    img = np.zeros((bev_size_y, bev_size_x, 4), dtype=np.uint8)
    color_rgba = np.concatenate([payload, np.full((*payload.shape[:-1], 1), 255)], axis=-1)
    img[pixel_coords_y, pixel_coords_x] = color_rgba

    # Delaunay triangularization and fill in missing pixels.
    subsample_interval = int(1.0/subsample_rate)

    points = points[::subsample_interval]
    payload = payload[::subsample_interval]
    pixel_coords = pixel_coords[::subsample_interval]

    # It's possible that there's no point when caller trying to map only one class and that class does not appear in the image.
    # scipy.spatial.Delaunay will raise ValueError otherwise
    if points.shape[0] >= 3:
        triang = Delaunay(points).simplices
    else:
        triang = []

    bg = Image.new('RGBA', (bev_size_y, bev_size_x))
    draw = ImageDraw.Draw(bg)

    for triangle in triang:
        triangle_vertices = np.array([points[triangle[0]][:2], points[triangle[1]][:2], points[triangle[2]][:2]])
        # if triangle_area(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]) > max_triangle_size:
        #     continue
        if np.max(np.max(triangle_vertices, axis=0) - np.min(triangle_vertices, axis=0)) > max_triangle_size:
            continue
        draw.polygon([tuple(pixel_coords[triangle[0]]), tuple(pixel_coords[triangle[1]]), tuple(pixel_coords[triangle[2]])], fill = tuple(payload[triangle[0]].astype(np.int32))+tuple([255]))

    triangles = bg.copy()
    fg = Image.fromarray(img, 'RGBA')
    x, y = ((bg.width - fg.width) // 2 , (bg.height - fg.height) // 2)
    bg.paste(fg, (x, y), fg)

    # mapped bev, pixels, triangles
    return bg, fg, triangles

def to_right_handed(points: np.ndarray):
    points = points.copy()
    points[..., 1] = -points[..., 1]
    return points

def bev_map_frame(agent_path, frame, save_path=None, log_to_rerun=False):
    if save_path is None:
        save_path = agent_path

    if log_to_rerun:
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

    if log_to_rerun:
        rr.log(f'lidar', rr.Points3D(to_right_handed(lidar_points)))

    all_points = []
    all_semantic_pixels = []
    
    cameras = ['front_camera', 'left_front_camera', 'left_back_camera', 'back_camera', 'right_back_camera', 'right_front_camera']
    semantic_cameras = ['front_semantic_camera', 'left_front_semantic_camera', 'left_back_semantic_camera', 'back_semantic_camera', 'right_back_semantic_camera', 'right_front_semantic_camera']
    depth_cameras = ['front_depth_camera', 'left_front_depth_camera', 'left_back_depth_camera', 'back_depth_camera', 'right_back_depth_camera', 'right_front_depth_camera']

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
        depth_path = os.path.join(agent_path, f'{depth_camera}/{frame}.npz')

        with Image.open(img_path) as pil_img:
            pixels = np.array(pil_img)
        
        with Image.open(semantic_path) as pil_img:
            semantic_pixels = np.array(pil_img)
        
        with open(depth_path, 'rb') as f:
            depth = np.load(f)['data']

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

        
        if log_to_rerun:
            subsample_interval = 10
            rr.log(f'camera/{camera}/points_xyz', rr.Points3D(to_right_handed(vehicle_xyz.reshape(-1, 3)[::subsample_interval]), colors=pixels.reshape(-1, 3)[::subsample_interval]/255.0))
            rr.log(f'semantic_camera/{semantic_camera}/points_xyz', rr.Points3D(to_right_handed(vehicle_xyz.reshape(-1, 3)[::subsample_interval]), colors=semantic_pixels.reshape(-1, 3)[::subsample_interval]/255.0))

    all_points = np.concatenate(all_points, axis=0)
    all_semantic_pixels = np.concatenate(all_semantic_pixels, axis=0)

    bev_image, bev_pixels, bev_triangles = points_to_bev(all_points, all_semantic_pixels, bev_range=50.0)

    bev_image_path = os.path.join(save_path, f'bev_mapping', f'{frame}.png')
    bev_pixels_path = os.path.join(save_path, f'bev_mapping_pixels', f'{frame}.png')
    bev_triangles_path = os.path.join(save_path, f'bev_mapping_triangles', f'{frame}.png')

    os.makedirs(os.path.join(save_path, f'bev_mapping'), exist_ok=True)
    os.makedirs(os.path.join(save_path, f'bev_mapping_pixels'), exist_ok=True)
    os.makedirs(os.path.join(save_path, f'bev_mapping_triangles'), exist_ok=True)

    bev_image.save(bev_image_path)
    bev_pixels.save(bev_pixels_path)
    bev_triangles.save(bev_triangles_path)

    if log_to_rerun:
        rr.log(f'bev_semantic_map/reconstruct', rr.ImageEncoded(path=bev_image_path))
        rr.log(f'bev_semantic_map/pixels', rr.ImageEncoded(path=bev_pixels_path))
        rr.log(f'bev_semantic_map/triangles', rr.ImageEncoded(path=bev_triangles_path))
        rr.log(f'bev_semantic_map/gt', rr.ImageEncoded(path=os.path.join(agent_path, 'birds_view_semantic_camera', f'{frame}.png')))
        rr.log(f'bev_range', rr.Boxes3D(centers=[0, 0, 0], half_sizes=[50.0, 50.0, 1.0]))
    return np.array(bev_image)
