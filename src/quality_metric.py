import numpy as np
import torch
import logging


class QualityMetric:
    # Q_loss: Normalized loss improvement.
    # Q_consistency: Consistency of model updates.
    # Q_data: Normalized data size.
    # quality = α·Q_loss + β·Q_consistency + γ·Q_data
    # where α + β + γ = 1.0
    # Default weights are α=0.6, β=0.3, γ= 0.1
    
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        """Quality weights as specified in paper Section IV.B"""
        self.alpha = alpha  # Loss improvement weight
        self.beta = beta    # Consistency weight  
        self.gamma = gamma  # Data size weight
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1"
        self.logger = logging.getLogger('PUMB_Quality')

    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        """
        THEORY-ALIGNED: Implement q_i^t = α·Q_loss + β·Q_consistency + γ·Q_data
        """
        # Q_loss: Normalized loss improvement
        loss_improvement = max(0, loss_before - loss_after)  # ΔL_i^t
        
        if all_loss_improvements is not None and len(all_loss_improvements) > 0:
            max_improvement = max(all_loss_improvements) + 1e-8
            Q_loss = loss_improvement / max_improvement
        else:
            Q_loss = 1.0 if loss_improvement > 0 else 0.1
        
        # Q_consistency: As per paper equation
        param_values = torch.cat([p.flatten() for p in param_update.values()])
        param_np = param_values.detach().cpu().numpy()
        
        if len(param_np) > 0:
            mean_val = np.mean(param_np)
            var_val = np.var(param_np)
            # Q_consistency = exp(-Var[Δθ]/Mean[Δθ]^2 + ε)
            consistency_ratio = var_val / (mean_val**2 + 1e-8)
            Q_consistency = np.exp(-consistency_ratio)
        else:
            Q_consistency = 0.5
        
        # Q_data: Normalized data size
        max_data_size = max(data_sizes.values()) if data_sizes else 1
        Q_data = data_sizes.get(client_id, 1) / max_data_size
        
        # Combine according to paper formula
        quality = self.alpha * Q_loss + self.beta * Q_consistency + self.gamma * Q_data
        
        self.logger.info(f"Client {client_id}: Q_loss={Q_loss:.4f}, Q_consistency={Q_consistency:.4f}, Q_data={Q_data:.4f}, final_score={quality:.4f}")
        return max(0.01, min(1.0, quality))  # Clamp to reasonable range


    def _flatten_params(self, model_params):
        """Flatten PyTorch parameters into a single vector."""
        if isinstance(model_params, dict):
            # If it's a state dict
            return torch.cat([p.flatten() for p in model_params.values()])
        elif isinstance(model_params, list):
            # If it's a list of tensors
            return torch.cat([p.flatten() for p in model_params])
        else:
            # Assume it's already a tensor
            return model_params.flatten()
    
    def _flatten_params_numpy(self, model_params):
        """Flatten numpy parameters into a single vector."""
        if isinstance(model_params, dict):
            # If it's a dict
            return np.concatenate([p.flatten() for p in model_params.values()])
        elif isinstance(model_params, list):
            # If it's a list of arrays
            return np.concatenate([p.flatten() for p in model_params])
        else:
            # Assume it's already an array
            return model_params.flatten()