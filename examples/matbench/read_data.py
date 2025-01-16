from matbench.bench import MatbenchBenchmark
from torch_geometric.data import Data
import torch
import ase
from ase.neighborlist import neighbor_list
from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_pbc, get_radius_graph_pbc



def check_structures_or_compositions(df):
    # Check if the dataset contains structures or compositions
    if "structure" in df.columns:
        return "structure"
    elif "composition" in df.columns:
        return "composition"
    else:
        raise ValueError(
            "The dataset must contain either 'structure' or 'composition' column."
        )


def process_structure_data(df, radius, max_neighbors):
    """Process a DataFrame containing structure data into PyTorch Geometric Data objects."""
    data_list = []
    for _, row in df.iterrows():
        structure = row.structure
        target = row[1]

        # Extract attributes
        atomic_numbers = torch.tensor(
            [site.specie.Z for site in structure], dtype=torch.float
        ).view(-1, 1)
        positions = torch.tensor(structure.cart_coords, dtype=torch.float)
        if positions.size(0) < 2:
            print("Skipping single-node graph.")
            continue
        pbc = torch.tensor(
            structure.lattice.pbc, dtype=torch.bool
        )  # PBC as [bool, bool, bool]
        cell = None
        if pbc.any():
            cell = torch.tensor(
                structure.lattice.matrix, dtype=torch.float
            )  # 3x3 unit cell matrix

        # Create PyG Data object
        data = Data(
            x=atomic_numbers,
            pos=positions,
            pbc=pbc,
            cell=cell,
            y=torch.tensor([target]),
        )
        radius_graph = get_radius_graph_pbc(radius=radius, max_neighbours=max_neighbors) if pbc.any() else get_radius_graph(radius=radius, max_neighbours=max_neighbors)
        data = radius_graph(data)
        data_list.append(data)
    return data_list


if __name__ == "__main__":
    # Specify the list of datasets
    structure_datasets = [
        "matbench_mp_e_form",
        "matbench_mp_gap",
        "matbench_perovskites",
        "matbench_log_gvrh",
        "matbench_log_kvrh",
        "matbench_dielectric",
        "matbench_phonons",
        "matbench_jdft2d",
    ]
    composition_datasets = [
        "matbench_glass",
        "matbench_expt_is_metal",
        "matbench_expt_gap",
        "matbench_steels",
    ]
    mixed_datasets = []
    classification_structure_datasets = ["matbench_mp_is_metal"]

    # Initialize Matbench with local dataset directory
    mb = MatbenchBenchmark(autoload=False)

    for dataset_name in structure_datasets:
        # Find the task corresponding to the dataset name
        task = next(task for task in mb.tasks if task.dataset_name == dataset_name)
        task.load()

        # Process the dataset for fold 0
        fold_number = 0  # Specify the fold number
        train_val_df = task.get_train_and_val_data(
            fold_number=fold_number, as_type="df"
        )

        # Check the type of data and process accordingly
        data_type = check_structures_or_compositions(train_val_df)
        if data_type == "structure":
            print(f"Processing {dataset_name} with structures...")
            train_val_data_list = process_structure_data(train_val_df)
        elif data_type == "composition":
            print(f"Skipping {dataset_name} with compositions...")
            continue  # Skip compositions for now

        # Save the processed data
        torch.save(train_val_data_list, f"processed_{dataset_name}_train_val.pt")

        print(
            f"Processed {len(train_val_data_list)} train and val samples {dataset_name}."
        )