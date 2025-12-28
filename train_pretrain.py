import torch
import torch.optim as optim
import os
import datetime
from utils.data import QG_Dataset
from utils.misc import generate_patches, load_config
from utils.masking import BlockMaskGenerator
from utils.optim import adjust_learning_rate
from utils.logging import setup_logger
from utils.checkpointing import save_checkpoint, load_checkpoint
from models import IJEPA, VisionTransformer, MaskPredictor

# 1. LOAD CONFIGURATION
cfg = load_config("colab_config.yaml")

# 2. INITIALIZE LOGGING
# Generate a unique run ID based on time
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{cfg['model']['name']}_{timestamp}"

# Create unique subdirectories in Drive
checkpoint_dir = os.path.join(cfg['train']['checkpoint_dir'], run_name)
log_dir = os.path.join(cfg['train']['log_dir'], run_name)

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Update the logger and save_checkpoint calls to use these paths
logger = setup_logger(log_dir=log_dir, model_name=cfg['model']['name'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3. SETUP DATA
dataset = QG_Dataset(cfg['data']['h5_path'])
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=cfg['data']['batch_size'], 
    shuffle=True,
    num_workers=cfg['data']['num_workers']
)

# 4. SETUP MODELS
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

# 5. OPTIMIZATION
optimizer = optim.AdamW(
    model.parameters(), 
    lr=cfg['train']['base_lr'], 
    weight_decay=cfg['train']['weight_decay']
)
masker = BlockMaskGenerator(
    grid_size=cfg['masking']['grid_size'],
    num_targets=cfg['masking']['num_targets'],
    target_size=tuple(cfg['masking']['target_size']),
    context_size=tuple(cfg['masking']['context_size'])
)

# 6. RESUME
checkpoint_path = f"{cfg['train']['checkpoint_dir']}/latest_checkpoint.pth"
start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device)

# 7. TRAINING LOOP
for epoch in range(start_epoch, cfg['train']['epochs']):
    current_lr = adjust_learning_rate(
        optimizer, epoch, 
        cfg['train']['warmup_epochs'], 
        cfg['train']['epochs'], 
        cfg['train']['base_lr']
    )
    
    epoch_loss = 0
    model.train()
    
    for images, _ in dataloader:
        images = images.to(device)
        patches = generate_patches(images) 
        c_mask, t_mask = masker.generate_batch_masks(images.size(0), device)
        
        with torch.no_grad():
            target_latents = model.target_encoder(patches) 
        
        context_latents = model.context_encoder(patches, c_mask)
        preds = model.predictor(context_latents, t_mask)
        
        loss = torch.nn.functional.mse_loss(preds, target_latents[t_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
        }, is_best, checkpoint_dir=cfg['train']['checkpoint_dir'])