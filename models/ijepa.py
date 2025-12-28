import torch
import torch.nn as nn
import copy

class IJEPA(nn.Module):
    def __init__(self, encoder, predictor, ema_momentum=0.999):
        super().__init__()
        self.context_encoder = encoder
        self.predictor = predictor
        
        # Create Target Encoder as a deep copy of Context Encoder
        self.target_encoder = copy.deepcopy(encoder)
        
        # Freeze Target Encoder (it is updated via EMA, not SGD)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_target_encoder(self):
        """Standard EMA update: target = m * target + (1 - m) * context"""
        for c_param, t_param in zip(self.context_encoder.parameters(), 
                                    self.target_encoder.parameters()):
            t_param.data.mul_(self.ema_momentum).add_(c_param.data, alpha=1 - self.ema_momentum)