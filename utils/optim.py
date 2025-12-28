import math

def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs, base_lr):
    """
    Computes and sets the learning rate based on a Cosine Decay schedule with Warmup.
   
    """
    if epoch < warmup_epochs:
        # Linear warmup to prevent gradient explosion in the ViT
        lr = base_lr * epoch / warmup_epochs 
    else:
        # Cosine decay from base_lr down to near zero
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr