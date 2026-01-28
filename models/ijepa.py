import torch
import torch.nn as nn
import copy

class IJEPA(nn.Module):
    def __init__(self, encoder, predictor, ema_momentum=0.996):
        super().__init__()
        self.context_encoder = encoder
        self.predictor = predictor
        
        # Create Target Encoder as a deep copy of Context Encoder
        self.target_encoder = copy.deepcopy(encoder)
        
        # Freeze Target Encoder (updated via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # For a 25-epoch run, 0.996 allows the teacher to evolve 
        # fast enough to keep the student challenged.
        self.ema_momentum = ema_momentum

    def forward(self, patches, context_indices, target_indices):
        """
        patches: [B, N_total, D]
        context_indices: [B, N_ctx]
        target_indices: [B, N_trg]
        """
        # 1. Encode context (Student)
        # Note: Your encoder likely takes (patches, indices)
        context_latents = self.context_encoder(patches, context_indices)
        
        # 2. Predict targets (Student)
        # We now pass BOTH sets of indices so the predictor can align them spatially
        predictions = self.predictor(context_latents, context_indices, target_indices)
        
        return predictions

    @torch.no_grad()
    def update_target_encoder(self):
        """Standard EMA update: target = m * target + (1 - m) * context"""
        for c_param, t_param in zip(self.context_encoder.parameters(), 
                                    self.target_encoder.parameters()):
            # Using .lerp_ (Linear Interpolation) is numerically cleaner for EMA
            t_param.data.lerp_(c_param.data, 1.0 - self.ema_momentum)