import torch
import torch.optim as optim
import os
import datetime
from utils import QG_Dataset, generate_patches, load_config, adjust_learning_rate, setup_logger, save_checkpoint, load_checkpoint, masked_mse_loss, init_experiment, QG_MaskCollator
from models import IJEPA, VisionTransformer, MaskPredictor
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler # Updated import
import torch.nn.functional as F

# Initialize Scaler with the new non-deprecated syntax
scaler = GradScaler('cuda')

# LOAD CONFIGURATION
cfg = load_config("colab_config.yaml")

# INITIALIZE LOGGING
logger, checkpoint_dir, device = init_experiment(cfg)

# SETUP DATA
dataset = QG_Dataset(cfg['data']['h5_path'])
logger.info(f"--- Dataset Loaded (Size: {len(dataset)}) ---")
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=cfg['data']['batch_size'], 
    shuffle=True,
    num_workers=cfg['data']['num_workers'],
    pin_memory=True # Added for faster data transfer
)

# SETUP MODELS
encoder = VisionTransformer(
    patch_dim=cfg['model']['patch_dim'], 
    embed_dim=cfg['model']['embed_dim'], 
    depth=cfg['model']['depth'], 
    num_heads=cfg['model']['num_heads']
)
predictor = MaskPredictor(
    encoder_dim=cfg['model']['embed_dim'], 
    predictor_dim=cfg['model']['predictor_embed_dim'], 
    depth=cfg['model']['predictor_depth']
)
model = IJEPA(encoder, predictor, ema_momentum=cfg['train']['ema_momentum']).to(device)

# OPTIMIZATION
collator = QG_MaskCollator(grid_size=16)
optimizer = optim.AdamW(
    model.parameters(), 
    lr=cfg['train']['base_lr'], 
    weight_decay=cfg['train']['weight_decay']
)

# RESUME
checkpoint_path = f"{cfg['train']['checkpoint_dir']}/latest_checkpoint.pth"
start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device)

# TRAINING LOOP
print("--- Starting Training Loop ---")
for epoch in range(start_epoch, cfg['train']['epochs']):
    current_lr = adjust_learning_rate(
        optimizer, epoch, 
        cfg['train']['warmup_epochs'], 
        cfg['train']['epochs'], 
        cfg['train']['base_lr']
    )
    
    epoch_loss = 0
    model.train()
    
    print("Starting epoch...")
    for images, labels, ctx_idx, trg_idx in dataloader:
        images = images.to(device, non_blocking=True)
        ctx_idx = ctx_idx.to(device, non_blocking=True)
        trg_idx = trg_idx.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        # --- MIXED PRECISION FORWARD PASS ---
        with autocast('cuda'):
            patches = generate_patches(images) # [B, 256, D]

            # Teacher gets full patches, but we index the output
            with torch.no_grad():
                target_encoder_out = model.target_encoder(patches)
                # Expand target indices to match latent dimension
                trg_idx_exp = trg_idx.unsqueeze(-1).expand(-1, -1, target_encoder_out.size(-1))
                target_latents = torch.gather(target_encoder_out, 1, trg_idx_exp)

            # Student only gets context patches via index selection
            context_latents = model.context_encoder(patches, indices=ctx_idx)
            preds = model.predictor(context_latents, trg_idx)

            loss = F.mse_loss(preds, target_latents) # Standard MSE now works!    

        # --- SCALED BACKWARD PASS ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.update_target_encoder()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch [{epoch+1}/{cfg['train']['epochs']}] | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
    
    # 8. CHECKPOINTING
    is_best = avg_loss < best_loss
    if is_best: best_loss = avg_loss
    
    if (epoch + 1) % cfg['train']['save_freq'] == 0 or is_best:
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': model.context_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, is_best, checkpoint_dir=checkpoint_dir)