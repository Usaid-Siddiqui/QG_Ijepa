import yaml
import subprocess
import os

# Define experimental matrix
experiments = [
    {
        "name": "JEPA_Linear_Probe_100pct",
        "ckpt": "checkpoints/jepa_latest.pth", # Adjust to your actual path
        "freeze": True,
        "data_fraction": 1.0,
        "lr": 0.001
    },
    {
        "name": "JEPA_Finetune_10pct",
        "ckpt": "checkpoints/jepa_latest.pth",
        "freeze": False,
        "data_fraction": 0.1,
        "lr": 0.0001 # Pretrained models need smaller LR
    },
    {
        "name": "Scratch_10pct",
        "ckpt": "", # Empty string triggers the 'scratch' logic
        "freeze": False,
        "data_fraction": 0.1,
        "lr": 0.001 # Scratch needs a higher LR to learn from zero
    }
]

def update_config(changes):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Apply changes to the finetune section
    config['finetune']['checkpoint_to_load'] = changes['ckpt']
    config['finetune']['freeze_encoder'] = changes['freeze']
    config['finetune']['lr'] = changes['lr']
    
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    for exp in experiments:
        print(f"\n>>> STARTING EXPERIMENT: {exp['name']}")
        update_config(exp)
        
        # Call evaluation script as a separate process
        # This ensures CUDA memory is cleared between runs
        subprocess.run(["python", "train_finetune.py"])