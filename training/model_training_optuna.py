import optuna
from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml
import os
curDir = os.getcwd()

dataset_path = f'{curDir}/training/data.yaml'

def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        # 'epochs': trial.suggest_int('epochs', 100, 300),
        'epochs': trial.suggest_int('epochs', 100, 300),
        
        'imgsz': 640,
        'device':"mps",
        'batch':8,
        'optimizer': trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW']),
        'optimizer': 'SGD',
        
        'lr0': trial.suggest_loguniform('lr0', 1e-5, 1e-1),
        'lrf': trial.suggest_loguniform('lrf', 0.01, 1.0),
        'momentum': trial.suggest_uniform('momentum', 0.6, 0.98),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 5),
        'warmup_momentum': trial.suggest_uniform('warmup_momentum', 0, 0.95),
        
        'box': trial.suggest_uniform('box', 0.2, 4.0),
        'cls': trial.suggest_uniform('cls', 0.2, 4.0),
        'hsv_h': trial.suggest_uniform('hsv_h', 0, 1.0),
        'hsv_s': trial.suggest_uniform('hsv_s', 0, 1.0),
        'hsv_v': trial.suggest_uniform('hsv_v', 0, 1.0),
        'degrees': trial.suggest_uniform('degrees', -180, 180),
        'translate': trial.suggest_uniform('translate', 0.0, 1.0),
        'scale': trial.suggest_uniform('scale', 0, 0.9),
        'shear': trial.suggest_uniform('shear', -180, 180),
        'perspective': trial.suggest_uniform('perspective', 0.0, 0.001),
        'flipud': trial.suggest_uniform('flipud', 0.0, 1.0),
        'fliplr': trial.suggest_uniform('fliplr', 0.0, 1.0),
        'mosaic': trial.suggest_uniform('mosaic', 0.0, 1.0),
        'mixup': trial.suggest_uniform('mixup', 0.0, 1.0),
        'copy_paste': trial.suggest_uniform('copy_paste', 0.0, 1.0),
    }

    # Create a YOLO model
    model = YOLO("yolov8s-seg.pt")

    # Train the model with the suggested hyperparameters
    results = model.train(
        data=dataset_path,
        project="training/runs",
        **params
    )

    # Return the metric to optimize (e.g., mAP50-95)
    return (1-results.results_dict['metrics/mAP50-95(B)'])

# Create an output directory for results
os.makedirs('optuna_results', exist_ok=True)

study_name = "precision-optimization"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

# Create an Optuna study
study = optuna.create_study(study_name=study_name,storage=storage_name, load_if_exists=True, direction= optuna.study.StudyDirection.MINIMIZE)

plt.figure(figsize=(14, 10))
optuna.visualization.plot_optimization_history(study).write_image('optuna_results/optimization_history.png')
plt.close()

plt.figure(figsize=(14, 10))
optuna.visualization.plot_param_importances(study).write_image('optuna_results/param_importances.png')
plt.close()

plt.figure(figsize=(14, 10))
optuna.visualization.plot_parallel_coordinate(study).write_image('optuna_results/parallel_coordinate.png')
plt.close()
# Optimize
# study.enqueue_trial(study.trials[-1].params)
study.optimize(objective, n_trials=10)  # Adjust n_trials as needed
# Print and save the best parameters and value achieved
print("Best parameters:", study.best_params)
print("Best value achieved:", study.best_value)

with open('optuna_results/best_results.txt', 'w') as f:
    f.write(f"Best parameters: {study.best_params}\n")
    f.write(f"Best value achieved: {study.best_value}\n")

# Visualize and save the optimization results

# Save the best parameters to a YAML file
with open('optuna_results/best_params.yaml', 'w') as f:
    yaml.dump(study.best_params, f)


print("Final model training completed.")