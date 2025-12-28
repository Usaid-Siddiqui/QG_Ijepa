import torch
import os
import logging

def save_checkpoint(state, is_best, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the latest state for resuming
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(state, latest_path)
    
    # Save a specific epoch version
    epoch_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{state['epoch']+1}.pth")
    torch.save(state, epoch_path)

    # Export just the encoder weights for finetuning
    encoder_path = os.path.join(checkpoint_dir, "best_encoder.pth")
    if is_best:
        torch.save(state['encoder_state_dict'], encoder_path)
        logging.info(f"--- New best encoder saved at epoch {state['epoch']+1} ---")

def load_checkpoint(checkpoint_path, model, optimizer, device):
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Successfully resumed from epoch {checkpoint['epoch']+1}")
    return checkpoint['epoch'] + 1, checkpoint.get('best_loss', float('inf'))