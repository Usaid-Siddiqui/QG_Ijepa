import torch
from models import VisionTransformer  # or however you import your encoder

def check_collapse(checkpoint_path, val_loader, device='cuda'):
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = VisionTransformer() # Init with your config
    model.load_state_dict(checkpoint['encoder_state_dict'])
    model.to(device).eval()

    all_embeddings = []
    with torch.no_grad():
        for images, _ in val_loader:
            # Get embeddings: [Batch, Tokens, Dim]
            emb = model(images.to(device))
            # Average over tokens to get one vector per jet
            all_embeddings.append(emb.mean(dim=1).cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    # Calculate variance per dimension, then average
    std = all_embeddings.std(dim=0).mean().item()
    
    print(f"Average Latent Std Dev: {std:.4f}")
    if std < 0.01:
        print("ALERT: Representation Collapse detected.")
    else:
        print("Latent space looks healthy and diverse.")