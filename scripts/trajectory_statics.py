import os
import json
import pickle
import argparse
import numpy as np
import plotly.graph_objects as go

def collect_ego_trajectory(dataset_path, distance_to_obstacle_considering_an_avoidance=10.0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    trajectories = {}
    obstacles = {}
    statics = {}
    for agent in os.listdir(os.path.join(dataset_path, 'agents')):
        agent_path = os.path.join(os.path.join(dataset_path, 'agents', agent))
        location_record_path = os.path.join(agent_path, 'gt_location', 'data.jsonl')
        location_records = []
        num_frames = 0
        lane_follow_frames = 0
        trun_left_right_frames = 0
        change_lane_frames = 0
        change_lane_near_ood_frames = 0
        with open(location_record_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                location_records.append(json.loads(line))
        for fn in os.listdir(os.path.join(agent_path, 'gt_ood_bbox')):
            obstacle_record_path = os.path.join(agent_path, 'gt_ood_bbox', fn)
            with open(obstacle_record_path, 'rb') as f:
                obstacle_record = pickle.load(f)
            for obstacle in obstacle_record['oods']:
                # in our current verison of the dataset, ood object does not move. We just need the location so we can simply overwrite.
                obstacles[obstacle['id']] = obstacle['location']
        agent_trajectory = []
        for record in location_records:
            traj = record['location']
            agent_trajectory.append(traj)
            num_frames += 1
            current_action = record['current_action']
            if current_action == 'LaneFollow':
                lane_follow_frames += 1
            elif current_action == 'Left' or current_action == 'Right':
                trun_left_right_frames += 1
            elif record['current_action'] == 'ChangeLaneLeft' or record['current_action'] == 'ChangeLaneRight':
                change_lane_frames += 1
                for obstacle in obstacles.values():
                    if np.linalg.norm(np.asarray(traj) - np.asarray(obstacle)) < distance_to_obstacle_considering_an_avoidance:
                        change_lane_near_ood_frames += 1
        agent_trajectory = np.asarray(agent_trajectory)
        
        trajectories[agent] = agent_trajectory
        statics[agent] = {
            'num_frames': num_frames,
            'lane_follow_frames': lane_follow_frames,
            'turn_left_right_frames': trun_left_right_frames,
            'change_lane_frames': change_lane_frames,
            'change_lane_near_ood_frames': change_lane_near_ood_frames,
        }

    obstacles = np.asarray(list(obstacles.values()))
    return trajectories, obstacles, statics

def main():
    parser = argparse.ArgumentParser(
        prog='Depth based BEV mapper',
    )

    parser.add_argument('dataset_path', type=str, help='path to the dataset')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_path = 'outputs/trajectory_statics'
    os.makedirs(output_path, exist_ok=True)

    trajectories = []
    statics = []
    for run_name in os.listdir(dataset_path):
        run_path = os.path.join(dataset_path, run_name)
        if not os.path.isfile(os.path.join(run_path, 'info.json')):
            continue
        run_trajectories, run_obstacles, run_statics = collect_ego_trajectory(run_path)
        trajectories.extend(list(run_trajectories.values()))
        statics.extend(list(run_statics.values()))
        for agent, trajectory in run_trajectories.items():
            fig = go.Figure()
            plot_path = os.path.join(output_path, f'{run_name}_{agent}.png')
            fig.add_trace(go.Scatter(
                x=trajectory[:, 0], y=trajectory[:, 1],
                mode='lines',
            ))
            fig.add_trace(go.Scatter(
                x=run_obstacles[:, 0], y=run_obstacles[:, 1],
                mode='markers',
            ))
            fig.write_image(plot_path)

    trajectories_lengths = [len(traj) for traj in trajectories]
    print(f'len: {np.mean(trajectories_lengths)} stddev: {np.std(trajectories_lengths)}')
    total_turn_left_right_frames = np.sum([d['turn_left_right_frames'] for d in statics])
    total_lane_follow_frame = np.sum([d['lane_follow_frames'] for d in statics])
    total_change_lane_frame = np.sum([d['change_lane_frames'] for d in statics])
    change_lane_near_ood_frames = np.sum(d['change_lane_near_ood_frames'] for d in statics)
    print(f'total frame: {sum(trajectories_lengths)} turn left or right: {total_turn_left_right_frames} lane follow: {total_lane_follow_frame} change lane: {total_change_lane_frame} change lane near ood: {change_lane_near_ood_frames}')


# python scripts/trajectory_statics.py ../../Datasets/carla

if __name__ == '__main__':
    main()
