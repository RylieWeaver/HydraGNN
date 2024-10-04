import torch
import torch.nn.functional as F
import os, json
import random
import math
from torch_geometric.data import Data
from hydragnn.preprocess import update_predicted_values

# Function to create a random tetrahedron with chiral configurations and positions
def create_chiral_instance():
    # Step 1: Choose 4 random atomic numbers without replacement
    atomic_numbers = random.sample(range(1, 119), 4)
    
    # Step 2: Choose one atom to be the chiral center
    chiral_center = random.choice(atomic_numbers)
    
    # Step 3: Remove the chiral center from the others
    other_atoms = [atom for atom in atomic_numbers if atom != chiral_center]
    
    # Step 4: Randomly decide clockwise or counterclockwise placement
    clockwise = random.choice([True, False])

    # Step 5: Arrange atoms
    if clockwise:
        # Sort other atoms in ascending order (smallest to largest)
        other_atoms.sort()
        chiral_tag = [0, 1, 0]  # Assign y value [0,1,0] for clockwise
    else:
        # Sort other atoms in descending order (largest to smallest)
        other_atoms.sort(reverse=True)
        chiral_tag = [0, 0, 1]  # Assign y value [0,0,1] for counterclockwise

    # Step 6: Create edge connections
    edges = [[0, 1], [0, 2], [0, 3]]  # Chiral center connected to the other 3 atoms
    
    # Step 7: Create node features (atomic numbers) and y labels
    atomic_numbers = torch.tensor([chiral_center] + other_atoms, dtype=torch.int64).view(-1, 1)  # Use int64 for one-hot encoding
    atomic_one_hot = F.one_hot(atomic_numbers.squeeze() - 1, num_classes=118)  # One-hot encode atomic number
    chirality_one_hot = torch.tensor([chiral_tag] + [[1, 0, 0]] * 3, dtype=torch.float)
    atomic_features = torch.cat([atomic_one_hot.float(), chirality_one_hot], dim=1)  # Concatenating along the last dimension
    
    # Step 8: Assign positions: Chiral center at [0,0,1] and other atoms on the XY plane at angles 0, 120, 240 degrees
    positions = [
        [0.0, 0.0, 1.0],  # Chiral center
        [1.0, 0.0, 0.0],  # First atom at 0 degrees
        [math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3), 0.0],  # Second atom at 120 degrees
        [math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3), 0.0],  # Third atom at 240 degrees
    ]
    pos = torch.tensor(positions, dtype=torch.float)

    # Step 9: Create edge index
    edge_index = torch.tensor(edges + [[i, j] for j, i in edges], dtype=torch.long).t().contiguous()

    # Step 10: Compute the scalar triple product for the chiral center
    # Step 10.1: Get relative vectors for src to chiral
    chiral_center_edges = edge_index[:, 3:6]
    chiral_src, chiral_dst = chiral_center_edges[0], chiral_center_edges[1]
    # Step 10.2: Get one-hot encoded vector for each src atom indicating its atomic number
    relative_vectors = torch.zeros(118, 3)  # Initialize with zeros
    for src, dst in zip(chiral_src, chiral_dst):
        index = int(atomic_numbers[src].item() - 1)  # Atomic number - 1 gives the index
        relative_vectors[index] = pos[dst] - pos[src]

    # Step 10.5: Filter out the zero rows (non-contributing atomic numbers)
    non_zero_rows = torch.any(relative_vectors != 0, dim=1)
    relative_vectors = relative_vectors[non_zero_rows]  # Only keep the non-zero rows

    # Step 10.6: Compute the scalar triple product using the relative vectors
    if relative_vectors.shape[0] == 3:  # Only proceed if there are exactly 3 non-zero vectors
        a, b, c = relative_vectors[0], relative_vectors[1], relative_vectors[2]
        scalar_triple_value = torch.dot(a, torch.cross(b, c))
    elif relative_vectors.shape[0] == 1:
        scalar_triple_value = torch.tensor(0.0)  # Default if we only have 1 non-zero vector
    else:
        scalar_triple_value = torch.tensor(0.0)  # Default if we don't have exactly 3 non-zero vectors
        print('Error: Expected 3 or 1 non-zero vectors but got', relative_vectors.shape[0])
    
    # Step 11: Construct PyTorch Geometric data object
    data = Data(x=atomic_features, edge_index=edge_index, pos=pos)
    data.scalar_triple = scalar_triple_value
    
    # Step 12: Update data with y_loc
    data = process_data(data)

    return data

def process_data(data):
    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chirality_custom.json")
    with open(filename, "r") as f:
        config = json.load(f)
    # Define the configuration parameters based on your needs
    output_names = config["NeuralNetwork"]["Variables_of_interest"]["output_names"]
    type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
    index = config["NeuralNetwork"]["Variables_of_interest"]["output_index"]
    graph_feature_dim = None
    node_feature_dim = config["Dataset"]["node_features"]["dim"]
    # Call the function to update data.y and data.y_loc
    update_predicted_values(type, index, graph_feature_dim, node_feature_dim, data)
    data.x = data.x[:, :118]
    return data

# Generate the dataset
def generate_chiral_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        dataset.append(create_chiral_instance())
    return dataset

# Example: Create a dataset with 10000 samples
num_samples = 10000
dataset = generate_chiral_dataset(num_samples)

# Save the dataset as a .pt file
torch.save(dataset, 'chiral_tetrahedron_dataset_with_positions_and_triples.pt')

print(f'Dataset of {num_samples} samples created and saved as chiral_tetrahedron_dataset_with_positions_and_triples.pt')
