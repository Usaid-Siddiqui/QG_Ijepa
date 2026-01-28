import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
from tqdm import tqdm

from utils import QG_Dataset, load_config
from models import VisionTransformer

class LinearProbe(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        # Strictly freeze the pre-trained weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # The only trainable part: a single linear mapping
        self.head = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            # Global Average Pooling (GAP) across the patch dimension
            # Shape: [Batch, Tokens, Dim] -> [Batch, Dim]
            global_feat = features.mean(dim=1) 
        return self.head(global_feat)

def run_evaluation():
    # Load config and setup device
    cfg = load_config("colab_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. INITIALIZE ENCODER & LOAD PRE-TRAINED WEIGHTS
    encoder = VisionTransformer(
        patch_dim=cfg['model']['patch_dim'], 
        embed_dim=cfg['model']['embed_dim'], 
        depth=cfg['model']['depth'], 
        num_heads=cfg['model']['num_heads']
    ).to(device)
    
    ckpt_path = cfg['finetune']['checkpoint_to_load']
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print(f"--- Loaded Pre-trained Encoder from {ckpt_path} ---")

    # 2. INITIALIZE LINEAR PROBE
    model = LinearProbe(encoder, cfg['model']['embed_dim']).to(device)

    # 3. PREPARE DATASETS FROM SEPARATE FILES
    train_ds = QG_Dataset(cfg['finetune']['train_h5_path'])
    test_ds = QG_Dataset(cfg['finetune']['test_h5_path'])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['finetune']['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg['finetune']['batch_size'], shuffle=False, num_workers=2)
    
    print(f"--- Data Ready: {len(train_ds)} train samples, {len(test_ds)} test samples ---")

    # 4. OPTIMIZER & LOSS
    optimizer = optim.AdamW(model.head.parameters(), lr=cfg['finetune']['lr'])
    criterion = nn.BCEWithLogitsLoss()

    # 5. TRAIN LINEAR HEAD
    print(f"--- Training Linear Head for {cfg['finetune']['epochs']} epochs ---")
    model.train()
    for epoch in range(cfg['finetune']['epochs']):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct/total)

    # 6. EVALUATE & COMPUTE ROC AUC
    print("--- Computing Final ROC AUC ---")
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            # Use sigmoid to get 0.0 - 1.0 probability range
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    # 7. PLOT ROC CURVE
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'I-JEPA Linear Probe (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Gluon Misidentification)')
    plt.ylabel('True Positive Rate (Quark Efficiency)')
    plt.title('Quark vs Gluon Discrimination Performance')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(os.path.dirname(ckpt_path), 'roc_curve.png')
    plt.savefig(plot_path)
    plt.show()

    print(f"\nFinal ROC AUC Score: {auc:.4f}")
    print(f"ROC curve saved to: {plot_path}")

if __name__ == "__main__":
    run_evaluation()