import yaml
import subprocess

# Define experimental matrix
experiments = [
    {
        "name": "I-JEPA_Linear_Probe",
        "ckpt": "checkpoints/jepa_latest.pth",
        "freeze": True,
        "lr": 1e-3
    },
    {
        "name": "I-JEPA_Full_Finetune",
        "ckpt": "checkpoints/jepa_latest.pth",
        "freeze": False,
        "lr": 1e-5
    },
    {
        "name": "From_Scratch_Baseline",
        "ckpt": "",  # Empty string triggers your 'scratch' logic
        "freeze": False,
        "lr": 1e-3
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
        subprocess.run(["python", "your_evaluation_script.py"])