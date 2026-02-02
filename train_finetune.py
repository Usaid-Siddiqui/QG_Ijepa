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

class MLPProbe(nn.Module):
    def __init__(self, encoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        # Strictly freeze the pre-trained weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Non-linear head to improve AUC
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x_patches = generate_patches(x, patch_size=8)
        with torch.no_grad():
            features = self.encoder(x_patches)
            global_feat = features.mean(dim=1) 
        return self.head(global_feat)

def run_evaluation():
    # 1. SETUP
    cfg = load_config("colab_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = cfg['finetune']['checkpoint_to_load']
    
    # Define save paths immediately
    checkpoint_folder_name = os.path.basename(os.path.dirname(ckpt_path))
    final_save_path = os.path.join("/content/drive/MyDrive/QG_IJEPA/finetune", checkpoint_folder_name)
    os.makedirs(final_save_path, exist_ok=True)

    # 2. INITIALIZE MODEL
    encoder = VisionTransformer(
        patch_dim=cfg['model']['patch_dim'], 
        embed_dim=cfg['model']['embed_dim'], 
        depth=cfg['model']['depth'], 
        num_heads=cfg['model']['num_heads']
    ).to(device)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint)
    model = MLPProbe(encoder, cfg['model']['embed_dim']).to(device)

    # 3. DATA
    train_loader = DataLoader(QG_Dataset(cfg['finetune']['train_h5_path']), 
                              batch_size=cfg['finetune']['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(QG_Dataset(cfg['finetune']['test_h5_path']), 
                             batch_size=cfg['finetune']['batch_size'], shuffle=False, num_workers=2)

    optimizer = optim.AdamW(model.head.parameters(), lr=cfg['finetune']['lr'])
    criterion = nn.BCEWithLogitsLoss()

    # 4. TRAINING LOOP WITH EMERGENCY SAVE
    print(f"--- Training MLP Head | Results path: {final_save_path} ---")
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

    # 5. FINAL EVALUATION
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

    # 6. PLOT & SAVE (Order: Save then Show)
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

    # 7. LOGGING & WEIGHTS
    torch.save(model.head.state_dict(), os.path.join(final_save_path, "finetune_head.pth"))
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(final_save_path, "finetune_results.txt"), "a") as f:
        f.write(f"[{timestamp}] AUC: {auc:.4f} | Epochs: {cfg['finetune']['epochs']} | LR: {cfg['finetune']['lr']}\n")

    print(f"\nFinal AUC: {auc:.4f}")
    print(f"Results saved to: {final_save_path}")

if __name__ == "__main__":
    run_evaluation()