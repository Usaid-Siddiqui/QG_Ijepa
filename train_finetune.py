import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import logging
from tqdm import tqdm

from utils.misc import generate_patches
from utils import QG_Dataset, load_config
from models import VisionTransformer

# --- Updated Logic to Create the Finetuning Path ---
def get_finetune_dir(encoder_path):
    # Get the folder name of the encoder (e.g., vit_small_ijepa_...)
    encoder_folder_name = os.path.basename(os.path.dirname(encoder_path))
    
    # Define the new base finetuning directory
    # Adjust 'QG_IJEPA' if your main project folder is named differently
    base_finetune_dir = "/content/drive/MyDrive/QG_IJEPA/finetuning"
    
    # Combine them
    final_dir = os.path.join(base_finetune_dir, encoder_folder_name)
    os.makedirs(final_dir, exist_ok=True)
    return final_dir

class QG_Classifier(nn.Module):
    def __init__(self, encoder, embed_dim, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.head = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        x_patches = generate_patches(x, patch_size=8)
        if next(self.encoder.parameters()).requires_grad:
            features = self.encoder(x_patches)
        else:
            with torch.no_grad():
                features = self.encoder(x_patches)
        global_feat = features.mean(dim=1) 
        return self.head(global_feat)

def setup_finetune_logger(log_dir):
    log_path = os.path.join(log_dir, "finetune_log.txt")
    logger = logging.getLogger("Finetune")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def run_training():
    cfg = load_config("colab_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. GENERATE DESIGNATED FOLDER
    finetune_save_dir = get_finetune_dir(cfg['finetune']['checkpoint_to_load'])
    logger = setup_finetune_logger(finetune_save_dir)
    
    logger.info(f"Finetuning results will be stored in: {finetune_save_dir}")
    logger.info(f"Mode: {'Frozen' if cfg['finetune']['freeze_encoder'] else 'Unfrozen'}")

    # 2. INITIALIZE ENCODER & MODEL
    encoder = VisionTransformer(
        patch_dim=cfg['model']['patch_dim'], 
        embed_dim=cfg['model']['embed_dim'], 
        depth=cfg['model']['depth'], 
        num_heads=cfg['model']['num_heads']
    ).to(device)
    
    checkpoint = torch.load(cfg['finetune']['checkpoint_to_load'], map_location=device)
    encoder.load_state_dict(checkpoint)
    
    model = QG_Classifier(
        encoder, 
        cfg['model']['embed_dim'], 
        freeze_encoder=cfg['finetune']['freeze_encoder']
    ).to(device)

    # 3. DATA
    train_loader = DataLoader(QG_Dataset(cfg['finetune']['train_h5_path']), 
                              batch_size=cfg['finetune']['batch_size'], shuffle=True)
    test_loader = DataLoader(QG_Dataset(cfg['finetune']['test_h5_path']), 
                             batch_size=cfg['finetune']['batch_size'], shuffle=False)

    # 4. OPTIMIZER
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=float(cfg['finetune']['lr']))
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    # 5. TRAINING LOOP
    for epoch in range(cfg['finetune']['epochs']):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(acc=correct/total)

        # 6. EVALUATE & LOG
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.numpy())

        current_auc = roc_auc_score(all_labels, all_probs)
        logger.info(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {correct/total:.4f} | Test AUC: {current_auc:.4f}")

        # 7. SAVE BEST MODEL & ROC PLOT TO THE NEW FOLDER
        if current_auc > best_auc:
            best_auc = current_auc
            logger.info(f"*** New Best AUC: {best_auc:.4f} - Saving to {finetune_save_dir} ***")
            
            torch.save(model.state_dict(), os.path.join(finetune_save_dir, 'finetuned_best.pth'))
            
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {best_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {os.path.basename(finetune_save_dir)}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(finetune_save_dir, 'best_roc_curve.png'))
            plt.close()

if __name__ == "__main__":
    run_training()