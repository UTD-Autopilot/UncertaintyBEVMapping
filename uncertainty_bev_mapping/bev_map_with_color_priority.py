from .mapping import *

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
