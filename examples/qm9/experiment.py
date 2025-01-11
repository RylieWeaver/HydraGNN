import os, json

import torch

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import torch_geometric
from tqdm import tqdm

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn
from hydragnn.utils.model import load_existing_model, save_model
from hydragnn.utils.distributed import get_device


# Update each sample prior to loading.
def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    # data.y = data.y[:, 10] / len(data.x)
    data.y = data.y[:, 10]
    graph_features_dim = [1]
    node_feature_dim = [1]
    return data


def qm9_pre_filter(data):
    return data.idx < 10000


# Make dataset
def create_dataset(config):
    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Use built-in torch_geometric datasets.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    dataset = torch_geometric.datasets.QM9(
        root="dataset/qm9", pre_transform=qm9_pre_transform, pre_filter=qm9_pre_filter
    )
    
    return dataset


def run_model(model_type, num_samples, config, train, val, test):
    verbosity = config["Verbosity"]["level"]
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    
    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    log_name = f"qm9_test_{model_type}_{num_samples}"
    hydragnn.utils.print.print_utils.setup_log(log_name)
    
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # Run training with the given model and qm9 datasets.
    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
    )
    
    hydragnn.utils.model.save_model(model, optimizer, log_name)
    load_existing_model(model, log_name, path="./logs/")
    model.eval()
    
    test_MAE = 0.0
    test_MSE = 0.0
    # pred_list = []
    # true_list = []
    for data_id, data in enumerate(tqdm(test)):
        # data.pos.requires_grad = True
        pred = model(data.to(get_device()))[0]
        test_MAE += torch.norm(pred - data.y, p=1).item()
        test_MSE += torch.norm(pred - data.y, p=2).item() ** 2
        # pred_list.append(pred)
        # true_list.append(data.y)
    
    return test_MAE / len(test), test_MSE / len(test)
    

def run_experiment():
    # Make sure to clean logs directory
    os.system("rm -rf logs")
    
    num_repeats = 3
    # list(map(int, np.logspace(np.log10(100), np.log10(10000), num=10)))
    # num_samples = [100, 166, 278, 464, 774, 1291, 2154, 3593, 5994, 10000]
    model_types = ["CGCNN", "EGNN", "GIN", "GAT", "MFC", "SAGE", "PNA", "SchNet", "DimeNet", "PNAPlus", "PNAEq", "PAINN", "MACE"]
    results = {model: [] for model in model_types}

    # Read Config
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
    with open(filename, "r") as f:
        config = json.load(f)
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    
    # Run Experiment
    dataset = create_dataset(config)
    for num_sample in num_samples:
        print(f"\n\n\nRunning experiment with {num_sample} samples")
        for repeat in range(num_repeats):
            print(f"\n--- Repetition {repeat + 1} ---")
            # Split dataset
            sample_dataset = dataset[:num_sample]
            train, val, test = hydragnn.preprocess.split_dataset(
                sample_dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
            )
            # Run dataset for all models
            for model_type in model_types:
                print(f"\nRunning model: {model_type}")
                MAE, MSE = run_model(model_type, num_sample, config, train, val, test)
                results[model_type].append({
                    "num_samples": num_sample,
                    "repetition": repeat + 1,
                    "MAE": MAE,
                    "MSE": MSE,
                })
            
    # Save results to file
    output_file = "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
            

if __name__ == "__main__":
    run_experiment()