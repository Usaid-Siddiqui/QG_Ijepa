"""
Full-scale ViT training comparison: pretrained encoder vs. from-scratch.

Runs two full end-to-end fine-tuning jobs back-to-back using the same
hyperparameters, then plots their ROC curves on a single figure so the
two runs can be directly compared.

Config keys used (all under 'vit_comparison' in config.yaml, with fallback
to the 'finetune' section values):

  vit_comparison:
    best_encoder_path:  path to best_encoder.pth from pretraining
    train_h5_path:      training HDF5 (defaults to finetune.train_h5_path)
    test_h5_path:       test HDF5     (defaults to finetune.test_h5_path)
    results_dir:        output directory
    epochs:             training epochs  (default 20)
    lr:                 learning rate    (default 1e-4)
    warmup_epochs:      LR warmup epochs (default 5)
    weight_decay:       AdamW weight decay (default 1e-4)
    batch_size:         (default 64)
    head_layers:        MLP head hidden sizes (default [512, 256])
    pool:               patch pooling mode  (default 'mean')
    dropout:            head dropout        (default 0.1)
    use_amp:            mixed precision     (default True)
    grad_clip:          gradient clip norm  (default 1.0)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
import datetime
import shutil
from tqdm import tqdm

from utils.misc import generate_patches, load_config
from utils.optim import adjust_learning_rate
from utils import QG_Dataset
from models import VisionTransformer

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FullViT(nn.Module):
    """End-to-end trainable ViT with an MLP classification head."""

    def __init__(self, encoder: VisionTransformer, embed_dim: int,
                 head_layers: list, pool_type: str = "mean", dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.pool_type = pool_type

        layers = []
        in_dim = embed_dim
        for hid in head_layers:
            layers.append(nn.Linear(in_dim, hid))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hid
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        patches = generate_patches(x, patch_size=8)   # [B, 256, patch_dim]
        features = self.encoder(patches)               # [B, 256, embed_dim]

        if self.pool_type == "mean":
            z = features.mean(dim=1)
        elif self.pool_type == "max":
            z, _ = features.max(dim=1)
        elif self.pool_type == "cls":
            z = features[:, 0]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        return self.head(z)                            # [B, 1]


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        outputs = model(images.to(device))
        all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
        all_labels.extend(labels.numpy())

    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    return auc, fpr, tpr, np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(label: str, encoder_path, cfg_model, cfg_run, train_loader,
               test_loader, device, save_dir):
    """
    Trains one ViT classifier to completion and returns (auc, fpr, tpr).

    label          -- human-readable name, e.g. "Pretrained" or "Scratch"
    encoder_path   -- path to best_encoder.pth, or None for random init
    cfg_model      -- config['model'] dict
    cfg_run        -- merged vit_comparison config dict
    """
    # Build encoder
    encoder = VisionTransformer(
        patch_dim=cfg_model['patch_dim'],
        embed_dim=cfg_model['embed_dim'],
        depth=cfg_model['depth'],
        num_heads=cfg_model['num_heads'],
    ).to(device)

    if encoder_path and os.path.exists(encoder_path):
        state = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(state)
        print(f"[{label}] Loaded encoder weights from {encoder_path}")
    else:
        print(f"[{label}] Training from random initialization")

    model = FullViT(
        encoder=encoder,
        embed_dim=cfg_model['embed_dim'],
        head_layers=cfg_run['head_layers'],
        pool_type=cfg_run['pool'],
        dropout=cfg_run['dropout'],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg_run['lr'],
        weight_decay=cfg_run['weight_decay'],
    )
    criterion = nn.BCEWithLogitsLoss()
    use_amp = cfg_run['use_amp'] and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    epochs = cfg_run['epochs']
    warmup = cfg_run['warmup_epochs']
    grad_clip = cfg_run['grad_clip']

    best_auc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_auc": []}

    try:
        for epoch in range(epochs):
            lr = adjust_learning_rate(optimizer, epoch, warmup, epochs, cfg_run['lr'])
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                        device, scaler, grad_clip)
            auc, _, _, _, _ = evaluate(model, test_loader, device)

            history["train_loss"].append(loss)
            history["train_acc"].append(acc)
            history["val_auc"].append(auc)

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), os.path.join(save_dir, f"{label}_best_model.pth"))

            print(f"[{label}] Epoch {epoch+1:>3}/{epochs} | "
                  f"LR {lr:.2e} | Loss {loss:.4f} | Acc {acc:.4f} | AUC {auc:.4f}"
                  + (" *" if auc == best_auc else ""))

    except KeyboardInterrupt:
        print(f"\n[{label}] Interrupted — evaluating at current state...")

    # Final evaluation using best checkpoint
    best_ckpt = os.path.join(save_dir, f"{label}_best_model.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"[{label}] Loaded best checkpoint (AUC {best_auc:.4f}) for final eval")

    final_auc, fpr, tpr, probs, labels_arr = evaluate(model, test_loader, device)
    print(f"[{label}] Final AUC: {final_auc:.4f}")

    # Save per-run metrics
    with open(os.path.join(save_dir, f"{label}_results.txt"), "w") as f:
        f.write(f"Label: {label}\n")
        f.write(f"Encoder path: {encoder_path}\n")
        f.write(f"Best AUC (during training): {best_auc:.4f}\n")
        f.write(f"Final AUC: {final_auc:.4f}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"LR: {cfg_run['lr']}\n")
        f.write(f"Warmup: {warmup}\n")
        f.write(f"Head layers: {cfg_run['head_layers']}\n")
        f.write(f"Pool: {cfg_run['pool']}\n")
        for i, (tl, ta, va) in enumerate(
                zip(history["train_loss"], history["train_acc"], history["val_auc"])):
            f.write(f"  Epoch {i+1:>3}: loss={tl:.4f}  acc={ta:.4f}  auc={va:.4f}\n")

    return final_auc, fpr, tpr, history


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Merge comparison config over finetune defaults
    ft = cfg.get('finetune', {})
    vc = cfg.get('vit_comparison', {})

    run_cfg = {
        'best_encoder_path': vc.get('best_encoder_path', ft.get('checkpoint_to_load', '')),
        'train_h5_path':     vc.get('train_h5_path',     ft.get('train_h5_path')),
        'test_h5_path':      vc.get('test_h5_path',      ft.get('test_h5_path')),
        'results_dir':       vc.get('results_dir',
                                    os.path.join(os.path.dirname(ft.get('finetune_dir', 'results')),
                                                 'vit_comparison')),
        'epochs':            vc.get('epochs',        20),
        'lr':                vc.get('lr',             1e-4),
        'warmup_epochs':     vc.get('warmup_epochs',  5),
        'weight_decay':      vc.get('weight_decay',   1e-4),
        'batch_size':        vc.get('batch_size',     ft.get('batch_size', 64)),
        'head_layers':       vc.get('head_layers',    [512, 256]),
        'pool':              vc.get('pool',           ft.get('pool', 'mean')),
        'dropout':           vc.get('dropout',        0.1),
        'use_amp':           vc.get('use_amp',        True),
        'grad_clip':         vc.get('grad_clip',      1.0),
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(run_cfg['results_dir'], f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Back up config
    try:
        shutil.copy("config.yaml", os.path.join(save_dir, "run_config.yaml"))
    except Exception as e:
        print(f"[!] Could not back up config: {e}")

    # Data loaders (shared between both runs)
    train_ds = QG_Dataset(run_cfg['train_h5_path'])
    test_ds  = QG_Dataset(run_cfg['test_h5_path'])
    train_loader = DataLoader(train_ds, batch_size=run_cfg['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=run_cfg['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

    cfg_model = cfg['model']
    encoder_path = run_cfg['best_encoder_path']

    # -----------------------------------------------------------------------
    # Run 1: Pretrained encoder
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("RUN 1: Full ViT fine-tuning with PRETRAINED encoder")
    print("="*60)
    auc_pre, fpr_pre, tpr_pre, hist_pre = run_single(
        label="Pretrained",
        encoder_path=encoder_path,
        cfg_model=cfg_model,
        cfg_run=run_cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        save_dir=save_dir,
    )

    # -----------------------------------------------------------------------
    # Run 2: From scratch
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("RUN 2: Full ViT fine-tuning FROM SCRATCH")
    print("="*60)
    auc_scr, fpr_scr, tpr_scr, hist_scr = run_single(
        label="Scratch",
        encoder_path=None,
        cfg_model=cfg_model,
        cfg_run=run_cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        save_dir=save_dir,
    )

    # -----------------------------------------------------------------------
    # Combined plots
    # -----------------------------------------------------------------------

    # -- ROC curve comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(fpr_pre, tpr_pre, color='steelblue', lw=2,
            label=f'Pretrained (AUC = {auc_pre:.4f})')
    ax.plot(fpr_scr, tpr_scr, color='tomato',    lw=2,
            label=f'Scratch    (AUC = {auc_scr:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # -- AUC-per-epoch curves
    ax = axes[1]
    epochs_axis = range(1, len(hist_pre['val_auc']) + 1)
    ax.plot(epochs_axis, hist_pre['val_auc'], color='steelblue', lw=2,
            marker='o', markersize=4, label='Pretrained')
    epochs_axis_scr = range(1, len(hist_scr['val_auc']) + 1)
    ax.plot(epochs_axis_scr, hist_scr['val_auc'], color='tomato', lw=2,
            marker='s', markersize=4, label='Scratch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation AUC')
    ax.set_title('AUC vs. Epoch')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "comparison_roc.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Comparison plot saved to: {plot_path}")

    # -- Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Pretrained AUC : {auc_pre:.4f}")
    print(f"  Scratch AUC    : {auc_scr:.4f}")
    delta = auc_pre - auc_scr
    print(f"  Delta          : {delta:+.4f} ({'pretrained wins' if delta > 0 else 'scratch wins' if delta < 0 else 'tie'})")
    print(f"  Results dir    : {save_dir}")

    # Write summary file
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(f"Run timestamp : {ts}\n")
        f.write(f"Encoder path  : {encoder_path}\n")
        f.write(f"Epochs        : {run_cfg['epochs']}\n")
        f.write(f"LR            : {run_cfg['lr']}\n")
        f.write(f"Head layers   : {run_cfg['head_layers']}\n")
        f.write(f"Pool          : {run_cfg['pool']}\n")
        f.write(f"\nPretrained AUC: {auc_pre:.4f}\n")
        f.write(f"Scratch AUC   : {auc_scr:.4f}\n")
        f.write(f"Delta         : {delta:+.4f}\n")


if __name__ == "__main__":
    run_comparison()
