from matbench.bench import MatbenchBenchmark
from torch_geometric.data import Data
import torch
from torch_geometric.transforms import RadiusGraph
import ase
from ase.neighborlist import neighbor_list


class RadiusGraphDynamic(RadiusGraph):
    r"""Creates edges based on node positions :obj:`pos` to all points within a
    given distance. Ensures every node has at least one neighbor by dynamically
    increasing the radius.
    """

    def __call__(self, data):
        initial_radius = self.r
        increment = 0.2  # Radius increment step
        max_radius = 48.0  # Set an upper limit to prevent infinite loops

        while True:
            # Generate edges based on the current radius
            edge_index = torch_geometric.nn.radius_graph(
                data.pos, r=self.r, loop=self.loop
            )

            # Check if every node has at least one neighbor
            unique_nodes_with_edges = torch.unique(edge_index)
            if unique_nodes_with_edges.size(0) == data.pos.size(0):
                # At least one neighbor for each node in the graph
                break
            else:
                # Increase the radius and try again
                self.r += increment
                print(
                    f"Increasing radius to {self.r:.2f} to ensure all nodes have neighbors."
                )
                if self.r > max_radius:
                    raise ValueError(
                        f"Exceeded maximum radius of {max_radius}. Some nodes may still be disconnected."
                    )

        # Step 1: Find unique edges in edge_index
        unique_edge_index, unique_indices = torch.unique(
            edge_index, dim=1, return_inverse=True
        )

        # Step 2: Update the data object
        data.edge_index = unique_edge_index

        # Reset radius to its initial value for the next call
        self.r = initial_radius

        return data


class RadiusGraphPBC(RadiusGraph):
    r"""Creates edges based on node positions :obj:`pos` to all points within a
    given distance, including periodic images. Ensures every node has at least one neighbor.
    """

    def __call__(self, data):
        data.edge_attr = None
        data.edge_shifts = None
        assert (
            "batch" not in data
        ), "Periodic boundary conditions not currently supported on batches."
        assert hasattr(
            data, "pbc"
        ), "The data must contain data.pbc as a bool (True) or list of bools for the dimensions ([True, False, True]) to apply periodic boundary conditions."

        initial_radius = self.r
        increment = 0.2  # Radius increment step
        max_radius = 48.0  # Set an upper limit to prevent infinite loops

        while True:
            ase_atom_object = ase.Atoms(
                positions=data.pos,
                cell=data.cell,
                pbc=data.pbc,
            )
            # 'i' : first atom index
            # 'j' : second atom index
            # 'd' : absolute distance
            # 'S' : shift vector
            # https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
            (
                edge_src,
                edge_dst,
                edge_length,
                edge_cell_shifts,
            ) = ase.neighborlist.neighbor_list(
                "ijdS", a=ase_atom_object, cutoff=self.r, self_interaction=self.loop
            )
            edge_index = torch.stack(
                [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)],
                dim=0,  # Shape: [2, n_edges]
            )

            # Check if every node has at least one neighbor
            unique_nodes_with_edges = torch.unique(edge_index)
            if unique_nodes_with_edges.size(0) == data.pos.size(0):
                # At least one neighbor for each node in graph
                break
            else:
                # Increase the radius and try again
                self.r += increment
                print(
                    f"Increasing radius to {self.r:.2f} to ensure all nodes have neighbors."
                )
                if self.r > max_radius:
                    raise ValueError(
                        f"Exceeded maximum radius of {max_radius}. Some nodes may still be disconnected."
                    )

        # Step 1: Find unique edges in edge_index
        unique_edge_index, unique_indices = torch.unique(
            edge_index, dim=1, return_inverse=True
        )

        # Step 2: Map the unique indices to edge_attr and edge_shifts
        unique_edge_attr = torch.tensor(edge_length, dtype=torch.float).unsqueeze(1)[
            unique_indices.unique()
        ]
        unique_edge_cell_shifts = torch.tensor(edge_cell_shifts, dtype=torch.float)[
            unique_indices.unique()
        ]

        # Step 3: Update the data object
        data.edge_index = unique_edge_index
        data.edge_attr = unique_edge_attr
        data.edge_shifts = torch.matmul(
            torch.tensor(unique_edge_cell_shifts, dtype=torch.float),
            data.cell.float(),  # Shape: [3, 3]
        )  # Shape: [n_unique_edges, 3]

        assert (
            data.edge_index.size(0) > 0
        ), "No edges were found within the cutoff radius."

        # Reset radius to its initial value for the next call
        self.r = initial_radius

        return data


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


def process_structure_data(df):
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
        radius_graph = RadiusGraphPBC(r=2.0) if pbc.any() else RadiusGraphDynamic(r=2.0)
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


# All initial radius: 2.0
