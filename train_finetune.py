import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import datetime
from tqdm import tqdm
from utils.misc import generate_patches
from utils import QG_Dataset, load_config
from models import VisionTransformer
import shutil

torch.manual_seed(42)

class MLPProbe(nn.Module):
    def __init__(self, encoder, embed_dim, head_layers, pool_type="mean", freeze_encoder=True, unfreeze_last=0):
        super().__init__()
        self.encoder = encoder

        # control parameter freezing behaviour
        if not freeze_encoder:
            # leave every parameter trainable
            for param in self.encoder.parameters():
                param.requires_grad = True
        elif unfreeze_last > 0:
            # only unfreeze last few transformer blocks
            for block in self.encoder.blocks[-unfreeze_last:]:
                for param in block.parameters():
                    param.requires_grad = True
            # freeze everything else
            for block in self.encoder.blocks[:-unfreeze_last]:
                for param in block.parameters():
                    param.requires_grad = False
        else:
            # freeze entire encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pool_type = pool_type

        # build the head from a list of hidden sizes; final output is scalar
        layers = []
        in_dim = embed_dim
        for hid in head_layers:
            layers.append(nn.Linear(in_dim, hid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hid
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        x_patches = generate_patches(x, patch_size=8)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                features = self.encoder(x_patches)
        else:
            features = self.encoder(x_patches)
        if self.pool_type == "mean":
            global_feat = features.mean(dim=1)
        elif self.pool_type == "max":
            global_feat, _ = features.max(dim=1)
        elif self.pool_type == "cls":
            # assume first token is CLS if used
            global_feat = features[:, 0]
        else:
            raise ValueError(f"Unknown pool type {self.pool_type}")
        return self.head(global_feat)

def run_evaluation():
    # setup
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg['finetune']['checkpoint_to_load']
    
    # Define save paths immediately
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    mode = "frozen" if cfg['finetune'].get('freeze_encoder', True) else "full-ft"
    if cfg['finetune'].get('unfreeze_last', 0) > 0:
        mode = f"unfreeze-{cfg['finetune']['unfreeze_last']}"
    
    run_id = f"{timestamp}_{mode}"
    
    # Define and create save path
    final_save_path = os.path.join(cfg['finetune']['finetune_dir'], run_id)
    os.makedirs(final_save_path, exist_ok=True)

    # initialize model and load checkpoint
    encoder = VisionTransformer(
        patch_dim=cfg['model']['patch_dim'], 
        embed_dim=cfg['model']['embed_dim'], 
        depth=cfg['model']['depth'], 
        num_heads=cfg['model']['num_heads']
    ).to(device)
    
    # If no checkpoint is found, training will be done from scratch
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[*] Loading I-JEPA weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(checkpoint)
        is_scratch = False
    else:
        print("[!] No checkpoint found. TRAINING FROM SCRATCH.")
        is_scratch = True

    # Adjust save path if training from scratch to avoid overwriting finetune results
    if is_scratch:
        run_id = f"{timestamp}_SCRATCH_lr{cfg['finetune']['lr']}"
        final_save_path = os.path.join(cfg['finetune']['finetune_dir'], run_id)
        os.makedirs(final_save_path, exist_ok=True)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint)

    # configuration for probe head and training
    head_layers = cfg['finetune'].get('head_layers', [512])
    pool_type = cfg['finetune'].get('pool', 'mean')
    freeze_encoder = cfg['finetune'].get('freeze_encoder', True)
    unfreeze_last = cfg['finetune'].get('unfreeze_last', 0)

    model = MLPProbe(
        encoder,
        cfg['model']['embed_dim'],
        head_layers=head_layers,
        pool_type=pool_type,
        freeze_encoder=freeze_encoder,
        unfreeze_last=unfreeze_last,
    ).to(device)

    # prepare data loaders
    train_loader = DataLoader(QG_Dataset(cfg['finetune']['train_h5_path']), 
                              batch_size=cfg['finetune']['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(QG_Dataset(cfg['finetune']['test_h5_path']), 
                             batch_size=cfg['finetune']['batch_size'], shuffle=False, num_workers=2)

    optimizer = optim.AdamW(model.head.parameters(), lr=cfg['finetune']['lr'])
    criterion = nn.BCEWithLogitsLoss()

    # training loop for head with interrupt handling
    print(f"--- Training MLP Head | Results path: {final_save_path} ---")
    print("probe config:",
          f"head={head_layers}",
          f"pool={pool_type}",
          f"freeze={freeze_encoder}",
          f"unfreeze_last={unfreeze_last}")
    try:
        model.train()
        for epoch in range(cfg['finetune']['epochs']):
            correct, total = 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=loss.item(), acc=correct/total)
    except KeyboardInterrupt:
        print("\n[!] User interrupted. Calculating metrics on partial training...")

    # evaluate on test set
    print("--- Computing Final ROC AUC ---")
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            outputs = model(images.to(device))
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    # plot ROC curve and save results
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'MLP Probe (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Performance: {checkpoint_folder_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save to the checkpoint-specific folder
    plt.savefig(os.path.join(final_save_path, "finetune_roc.png"), bbox_inches='tight')
    plt.show()

    # log metrics and save head weights
    torch.save(model.head.state_dict(), os.path.join(final_save_path, "finetune_head.pth"))

    # log configs used for this specific run
    try:
        shutil.copy("config.yaml", os.path.join(final_save_path, "run_config.yaml"))
        print(f"[*] Config backed up to {final_save_path}")
    except Exception as e:
        print(f"[!] Warning: Could not back up config: {e}")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(final_save_path, "finetune_results.txt"), "a") as f:
        f.write(f"[{timestamp}] AUC: {auc:.4f} | Epochs: {cfg['finetune']['epochs']} | LR: {cfg['finetune']['lr']}\n")

    print(f"\nFinal AUC: {auc:.4f}")
    print(f"Results saved to: {final_save_path}")

if __name__ == "__main__":
    run_evaluation()