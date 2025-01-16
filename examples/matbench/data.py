from matbench.bench import MatbenchBenchmark

# Initialize the Matbench framework
mb = MatbenchBenchmark()

# Loop through all tasks to download and load datasets
for task in mb.tasks:
    print(f"Downloading dataset for task: {task.dataset_name}")
    task.load()  # Downloads and prepares the dataset
