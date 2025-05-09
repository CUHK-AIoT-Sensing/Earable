import matplotlib.pyplot as plt
import numpy as np
import os
from random import uniform, sample 
from tqdm import tqdm
import pyroomacoustics as pra
import json
from wasn import simulation_to_wasn, simulation_to_infrastructure, wasn_to_slam
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def random_room(num_room=200):
    rooms = []
    for _ in range(num_room):
        width = uniform(5, 15)
        length = uniform(5, 20)
        height = uniform(2, 3)
        room_dim = [width, length, height]
        rt60 = uniform(0.25, 0.75)
        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        rooms.append([room_dim, absorption, max_order])
    return rooms

def random_conversation_pairs(num_people):
    """Generate random pairs for conversation."""
    indices = np.arange(num_people)
    np.random.shuffle(indices)
    pairs = {}
    for i in range(0, len(indices), 2):
        if i + 1 < len(indices):
            pairs[indices[i]] = indices[i + 1]
            pairs[indices[i + 1]] = indices[i]
    return pairs

def initialize_positions(room_size, num_people, close_distance=1.0):
    """Initialize positions ensuring conversation pairs are close to each other."""
    positions = np.zeros((num_people, 2))
    pairs = random_conversation_pairs(num_people)
    
    for person in range(num_people):
        if person in pairs:
            partner = pairs[person]
            # Random position for the first person in the pair
            positions[person] = np.random.rand(2) * (np.array(room_size) - close_distance) + 0.5
            
            # Calculate a position for the partner close to the first person
            angle = np.random.rand() * 2 * np.pi  # Random angle
            offset = np.array([np.cos(angle), np.sin(angle)]) * (close_distance / 2)
            positions[partner] = positions[person] + offset
            
    # Ensure all positions are within bounds
    positions = np.clip(positions, 0.5, np.array(room_size) - 0.5)
    
    return positions, pairs

def simulate_movement(room_size, num_people, num_moving, speed, time_steps, min_distance=0.5):
    # Initialize positions
    # positions = np.random.rand(num_people, 2) * np.array(room_size)
    # positions[:, 0] = np.clip(positions[:, 0], 0.5, room_size[0]-0.5)
    # positions[:, 1] = np.clip(positions[:, 1], 0.5, room_size[1]-0.5)
    positions, conversation_pairs = initialize_positions(room_size, num_people, close_distance=min_distance)

    directions = np.random.rand(num_people) * 2 * np.pi  # Random initial directions
    trajectory = np.ones((time_steps, num_people, 3))  # [T, N, 3] where 3 for x, y, and moving status
    directions_trajectory = np.zeros((time_steps, num_people))

    for t in range(time_steps):
        # Generate random conversation pairs
        # if t % 250 == 0:  # Change pairs every 50 time steps
        #     conversation_pairs = random_conversation_pairs(num_people)

        random_people = np.random.choice(num_people, num_moving, replace=False)
        for i in random_people:
            if i in conversation_pairs:
                partner = conversation_pairs[i]
                # Calculate direction towards partner
                direction_to_partner = np.arctan2(positions[partner][1] - positions[i][1],
                                                   positions[partner][0] - positions[i][0])
                new_direction = direction_to_partner + np.random.uniform(-np.pi/8, np.pi/8)
                directions[i] = new_direction
                distance_to_partner = np.linalg.norm(positions[partner] - positions[i])
                # Move towards partner
                if distance_to_partner > min_distance:
                    shift = np.array([np.cos(direction_to_partner), np.sin(direction_to_partner)]) * speed
                else:
                    # shift = np.array([0, 0])  # Stay still if too close
                    shift = np.random.normal(0, speed/10, 2)  # Random small movement
            else:
                # Current direction equals to the previous direction + random angle
                direction_shift = np.random.uniform(-np.pi/8, np.pi/8)
                new_direction = directions[i] + direction_shift
                directions[i] = new_direction
                shift = np.array([np.cos(directions[i]), np.sin(directions[i])]) * speed

            new_position = positions[i] + shift
            # Check bounds
            new_position[0] = np.clip(new_position[0], 0.5, room_size[0]-0.5)
            new_position[1] = np.clip(new_position[1], 0.5, room_size[1]-0.5)

            positions[i] = new_position
            
        # Store positions and moving status
        trajectory[t, :, :2] = positions
        directions_trajectory[t] = directions

    return trajectory, directions_trajectory

def plot_trajectory(trajectory, room_size, file_name):
    time_steps, num_people, _ = trajectory.shape
    plt.figure(figsize=(10, 10))

    for i in range(num_people):
        x = trajectory[:, i, 0]  # x positions
        y = trajectory[:, i, 1]  # y positions
        
        # Plot each user's trajectory
        plt.plot(x, y, marker='o', label=f'Person {i+1}')
        
    plt.title('User Trajectories in the Room')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid()
    plt.legend()
    plt.savefig(f'{file_name}.png')
    plt.close()

class simulator():
    def __init__(self) -> None:
        self.room_dims = random_room()

    def __call__(self, save_folder, num_data, config):
        '''
        For sound scape simulation, no need to simulate the waveform (same as sound event), but need to simulate the trajectory of multiple users
        '''
        max_source = config['max_source']; num_user = config['num_user']; num_source = config['num_source']
        meta_folder = save_folder + '/meta'; track_folder = save_folder + '/track'
        os.makedirs(meta_folder, exist_ok=True); os.makedirs(track_folder, exist_ok=True)

        if num_data is None:
            num_data = 100
        else: 
            num_data = num_data

        for i in range(num_data):
            print(f"Generating room {i}")
            room_dim, absorption, max_order = sample(self.room_dims, 1)[0]
            os.makedirs(f"{meta_folder}/room_{i}", exist_ok=True)

            trajectory, directions = simulate_movement(room_dim[:2], num_user, num_moving=1, speed=0.1, time_steps=num_source)
            plot_trajectory(trajectory, room_dim[:2], f"{track_folder}/room_{i}")

            for j in range(num_source): # number of sources
                source_idx = sample(range(num_user), 1)
                meta_data = {'room_dim': room_dim, 'absorption': absorption, 'mic_centers': trajectory[j].tolist(), 
                             'mic_orientations': directions[j].tolist(), 'source_idx': source_idx}
                with open(f"{meta_folder}/room_{i}/{j}.json", 'w') as f:
                    json.dump(meta_data, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', type=str, required=False, default='../../dataset')
    parser.add_argument('--num_data', type=int, default=100)
    config = {'max_source': 1, 'num_user': 5, 'num_source': 500}

    args = parser.parse_args()
    dataset_folder = args.save_folder + '/simulation/{}_{}_{}_{}/'.format('conversation', config['max_source'], config['num_user'], config['num_source'])

    train_folder = os.path.join(dataset_folder, 'train'); test_folder = os.path.join(dataset_folder, 'test')
    os.makedirs(train_folder, exist_ok=True); os.makedirs(test_folder, exist_ok=True) 
    conversation_simulator = simulator()
    TRAIN_NUM = args.num_data; TEST_NUM = None if args.num_data is None else args.num_data//5

    conversation_simulator(train_folder, num_data=TRAIN_NUM, config=config)
    conversation_simulator(test_folder, num_data=TEST_NUM, config=config)
    
    print('store in wasn format')
    output_json = {'datasets': {}}
    train_wasn_dict = simulation_to_wasn(train_folder); output_json['datasets']['train'] = train_wasn_dict
    test_wasn_dict = simulation_to_wasn(test_folder); output_json['datasets']['test'] = test_wasn_dict
    json.dump(output_json, open(f'{dataset_folder}/wasn.json', 'w'), indent=4, cls=NumpyEncoder)

    # print('store in slam format')
    # train_slam_dict = wasn_to_slam(train_wasn_dict)
    # json.dump(train_slam_dict, open(f'{dataset_folder}/slam_train.json', 'w'), indent=4, cls=NumpyEncoder)
    # test_slam_dict = wasn_to_slam(test_wasn_dict)
    # json.dump(test_slam_dict, open(f'{dataset_folder}/slam_test.json', 'w'), indent=4, cls=NumpyEncoder)

    print('store in infrastructure format')
    output_json = {'datasets': {}}
    infrastructure_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    infrastructure_orientations = np.array([0, 0, 0, 0])
    wasn_dict = simulation_to_infrastructure(train_folder, node_positions=infrastructure_positions, node_orientations=infrastructure_orientations); output_json['datasets']['train'] = wasn_dict
    wasn_dict = simulation_to_infrastructure(test_folder, node_positions=infrastructure_positions, node_orientations=infrastructure_orientations); output_json['datasets']['test'] = wasn_dict
    json.dump(output_json, open(f'{dataset_folder}/infra.json', 'w'), indent=4, cls=NumpyEncoder)