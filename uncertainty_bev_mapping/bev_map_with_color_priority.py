import cv2
from .mapping import *
from .utils import split_path_into_folders

fg_colors = [
    (0, 0, 142), # car
    (0, 0, 70), # truck
    (0, 60, 100), # bus
    (50, 100, 144), # bear deer cow
    # (130, 130, 130), # garbage_bag stand_food trash_can
]

def bev_map_frame_with_color_priority(agent_path, frame, save_path=None, log_to_rerun=False):
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

    # Do standard mapping first
    bev_image, bev_pixels, bev_triangles = points_to_bev(all_points, all_semantic_pixels, bev_range=50.0)

    # For each class overlay it to the mapping
    for color in fg_colors:
        color_semantic_pixles_index = np.all(all_semantic_pixels == color, axis=-1)
        color_points = all_points[color_semantic_pixles_index]
        color_semantic_pixles = all_semantic_pixels[color_semantic_pixles_index]
        _, _, color_triangles = points_to_bev(color_points, color_semantic_pixles, bev_range=50.0)
        bev_triangles.paste(color_triangles, (0, 0), color_triangles)

    bev_image.paste(bev_triangles, (0, 0), bev_triangles)

    bev_image_path = os.path.join(save_path, f'bev_mapping', f'bev_{frame}.png')
    bev_pixels_path = os.path.join(save_path, f'bev_mapping_pixels', f'pixels_{frame}.png')
    bev_triangles_path = os.path.join(save_path, f'bev_mapping_triangles', f'triangles{frame}.png')

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

def id_array_to_rgb(pixel_ids:np.ndarray, offset=0):
    pixel_ids_rgb = np.zeros((pixel_ids.shape[0], pixel_ids.shape[1], 3), dtype=np.uint8)
    pixel_ids_rgb[..., 0] = (pixel_ids + offset) % 256
    pixel_ids_rgb[..., 1] = ((pixel_ids + offset) // 256) % 256
    pixel_ids_rgb[..., 2] = ((pixel_ids + offset) // 256 // 256) % 256
    return pixel_ids_rgb

def rgb_array_to_id(pixel_ids_rgb:np.ndarray, offset=0):
    pixel_ids = np.zeros((pixel_ids_rgb.shape[0], pixel_ids_rgb.shape[1]), dtype=np.int32)
    pixel_ids = pixel_ids_rgb[..., 0].astype(np.int32) + (pixel_ids_rgb[..., 1].astype(np.int32) * 256) + (pixel_ids_rgb[..., 2].astype(np.int32) * 256 * 256) - offset
    return pixel_ids

def bev_map_frame_with_color_priority_uncertainty(agent_path, frame, uncertainty_data_path, save_path=None, log_to_rerun=False):
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
    all_pixel_ids = []
    all_semantic_pixels = []

    all_color_images = []
    all_semantic_images = []
    all_epistemic_images = []
    all_aleatoric_images = []

    cameras = ['front_camera', 'left_front_camera', 'left_back_camera', 'back_camera', 'right_back_camera', 'right_front_camera']
    semantic_cameras = ['front_semantic_camera', 'left_front_semantic_camera', 'left_back_semantic_camera', 'back_semantic_camera', 'right_back_semantic_camera', 'right_front_semantic_camera']
    depth_cameras = ['front_depth_camera', 'left_front_depth_camera', 'left_back_depth_camera', 'back_depth_camera', 'right_back_depth_camera', 'right_front_depth_camera']

    for camera_idx, (camera, semantic_camera, depth_camera) in enumerate(zip(cameras, semantic_cameras, depth_cameras)):
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
        semantic_path = os.path.join(uncertainty_data_path, f'{camera}/{frame}_pred.png')
        depth_path = os.path.join(agent_path, f'{depth_camera}/{frame}.npz')
        aleatoric_path = os.path.join(uncertainty_data_path, f'{camera}/{frame}_aleatoric.npy')
        epistemic_path = os.path.join(uncertainty_data_path, f'{camera}/{frame}_epistemic.npy')

        with Image.open(img_path) as pil_img:
            color = np.array(pil_img)

        with Image.open(semantic_path) as pil_img:
            semantic_pixels = np.array(pil_img)

        with open(depth_path, 'rb') as f:
            depth = np.load(f)['data']
        
        aleatoric_image = np.load(aleatoric_path)
        epistemic_image = np.load(epistemic_path)

        cam_h=depth.shape[0]
        cam_w=depth.shape[1]

        color = cv2.resize(color, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
        semantic_pixels = cv2.resize(semantic_pixels, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
        aleatoric_image = cv2.resize(aleatoric_image, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
        epistemic_image = cv2.resize(epistemic_image, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)

        all_color_images.append(color)
        all_semantic_images.append(semantic_pixels)
        all_aleatoric_images.append(aleatoric_image)
        all_epistemic_images.append(epistemic_image)

        pixel_ids = np.linspace(0, cam_h*cam_w-1, cam_h*cam_w, dtype=np.int32).reshape((cam_h, cam_w))
        pixel_ids_rgb = id_array_to_rgb(pixel_ids, offset=camera_idx*cam_h*cam_w)

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
        pixel_ids_rgb = np.delete(pixel_ids_rgb.reshape(-1, 3), valid_pixels_mask, axis=0)
        semantic_pixels = np.delete(semantic_pixels.reshape(-1, 3), valid_pixels_mask, axis=0)

        all_points.append(vehicle_xyz)
        all_pixel_ids.append(pixel_ids_rgb)
        all_semantic_pixels.append(semantic_pixels)

        if log_to_rerun:
            subsample_interval = 10
            rr.log(f'semantic_camera/{semantic_camera}/points_xyz', rr.Points3D(to_right_handed(vehicle_xyz.reshape(-1, 3)[::subsample_interval]), colors=semantic_pixels.reshape(-1, 3)[::subsample_interval]/255.0))

    all_points = np.concatenate(all_points, axis=0)
    all_pixel_ids = np.concatenate(all_pixel_ids, axis=0)
    all_semantic_pixels = np.concatenate(all_semantic_pixels, axis=0)

    all_color_images = np.stack(all_color_images)
    all_semantic_images = np.stack(all_semantic_images)
    all_aleatoric_images = np.stack(all_aleatoric_images)
    all_epistemic_images = np.stack(all_epistemic_images)

    # Do standard mapping first
    bev_ids_image, bev_ids_pixels, bev_ids_triangles = points_to_bev(all_points, all_pixel_ids, bev_range=50.0)

    # For each class overlay it to the mapping
    for color in fg_colors:
        color_semantic_pixles_index = np.all(all_semantic_pixels == color, axis=-1)
        color_points = all_points[color_semantic_pixles_index]
        _, _, color_triangles = points_to_bev(color_points, all_pixel_ids[color_semantic_pixles_index], bev_range=50.0)
        bev_ids_triangles.paste(color_triangles, (0, 0), color_triangles)

    bev_ids_image.paste(bev_ids_triangles, (0, 0), bev_ids_triangles)

    bev_ids = rgb_array_to_id(np.array(bev_ids_image))

    # map based on bev_ids
    camera_space_idx = (bev_ids//(cam_h*cam_w), (bev_ids % (cam_h*cam_w))//cam_w, bev_ids%cam_w)

    bev_color_image = all_color_images[camera_space_idx]
    bev_semantic_image = all_semantic_images[camera_space_idx]
    bev_aleatoric_image = all_aleatoric_images[camera_space_idx]
    bev_epistemic_image = all_epistemic_images[camera_space_idx]

    bev_color_path = os.path.join(save_path, f'bev_mapping_color', f'{frame}.png')
    bev_semantic_path = os.path.join(save_path, f'bev_mapping_pred', f'{frame}.png')
    bev_aleatoric_path = os.path.join(save_path, f'bev_mapping_aleatoric', f'{frame}.npy')
    bev_epistemic_path = os.path.join(save_path, f'bev_mapping_epistemic', f'{frame}.npy')

    os.makedirs(os.path.join(save_path, f'bev_mapping_color'), exist_ok=True)
    os.makedirs(os.path.join(save_path, f'bev_mapping_pred'), exist_ok=True)
    os.makedirs(os.path.join(save_path, f'bev_mapping_aleatoric'), exist_ok=True)
    os.makedirs(os.path.join(save_path, f'bev_mapping_epistemic'), exist_ok=True)

    bev_color = Image.fromarray(bev_color_image)
    bev_color.save(bev_color_path)
    bev_semantic = Image.fromarray(bev_semantic_image)
    bev_semantic.save(bev_semantic_path)
    np.save(bev_aleatoric_path, bev_aleatoric_image)
    np.save(bev_epistemic_path, bev_epistemic_image)

    if log_to_rerun:
        rr.log(f'bev_semantic_map/pred', rr.ImageEncoded(path=bev_semantic_path))
        rr.log(f'bev_semantic_map/gt', rr.ImageEncoded(path=os.path.join(agent_path, 'birds_view_semantic_camera', f'{frame}.png')))
        rr.log(f'bev_range', rr.Boxes3D(centers=[0, 0, 0], half_sizes=[50.0, 50.0, 1.0]))
    return np.array(bev_semantic)
