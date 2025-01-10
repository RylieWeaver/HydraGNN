import optuna
import time
from datetime import datetime


# Objective function to optimize
def objective(trial):
    # Log the start time of the trial
    start_time = datetime.now()
    print(f"Trial {trial.number} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Simulate a sleep time to mimic a time-consuming task
    time.sleep(100)

    # Log the end time of the trial
    end_time = datetime.now()
    print(f"Trial {trial.number} completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Example objective function: quadratic function
    x = trial.suggest_float("x", -10.0, 10.0)  # Example: Suggest a float parameter
    return (x - 2) ** 2  # Example: Simple quadratic function (minimum at x = 2)


def main():
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Run optimization with parallel workers
    # Adjust the number of trials as needed
    study.optimize(
        objective, n_trials=160, n_jobs=16
    )  # n_jobs specifies parallel workers

    # Output results
    print("Best value:", study.best_value)
    print("Best parameters:", study.best_params)


if __name__ == "__main__":
    main()
