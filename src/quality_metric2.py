import numpy as np
import torch
import logging



class QualityMetric:
    """
    IMPROVED QUALITY METRIC with fixes for mathematical stability and better exploration
    
    PROBLEMS FIXED:
    1. Q_consistency formula is now mathematically stable
    2. More balanced weights (reduced loss bias)
    3. Proper handling of negative mean values in consistency calculation
    4. Robust division by small numbers with better fallbacks
    5. Cross-client normalization within each round
    """
    
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):  # FIXED: More balanced weights
        """
        IMPROVED: More balanced quality assessment
        - Reduced loss weight from 0.6 to 0.4
        - Increased data weight from 0.1 to 0.3
        - This gives more value to clients with substantial data
        """
        self.alpha = alpha  # Loss improvement weight (reduced)
        self.beta = beta    # Consistency weight  
        self.gamma = gamma  # Data size weight (increased)
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1"
        self.logger = logging.getLogger('PUMB_Quality')
        
        # ADDED: Track statistics for better normalization
        self.loss_improvements_history = []
        self.consistency_scores_history = []

    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        """
        THEORY-ALIGNED with FIXES: Implement q_i^t = α·Q_loss + β·Q_consistency + γ·Q_data
        """
        # FIX 1: Better Q_loss calculation with relative improvement
        loss_improvement = max(0, loss_before - loss_after)  # ΔL_i^t
        
        # IMPROVED: Better normalization using all clients in current round
        if all_loss_improvements is not None and len(all_loss_improvements) > 0:
            # Use percentile-based normalization instead of max
            if len(all_loss_improvements) > 1:
                improvement_75th = np.percentile(all_loss_improvements, 75)
                Q_loss = min(1.0, loss_improvement / (improvement_75th + 1e-8))
            else:
                Q_loss = 1.0 if loss_improvement > 0 else 0.1
        else:
            # For standalone calculation, use relative improvement
            relative_improvement = loss_improvement / (loss_before + 1e-8)
            Q_loss = min(1.0, relative_improvement * 10)  # Scale up relative improvement
        
        # FIX 2: Robust Q_consistency calculation
        Q_consistency = self._calculate_robust_consistency(param_update)
        
        # FIX 3: Improved Q_data with relative scaling
        Q_data = self._calculate_data_quality(data_sizes, client_id)
        
        # FIX 4: Add quality floor to prevent extremely low scores
        quality = self.alpha * Q_loss + self.beta * Q_consistency + self.gamma * Q_data
        quality = max(0.1, min(1.0, quality))  # More generous floor (0.1 instead of 0.01)
        
        # Enhanced logging
        # self.logger.info(f"Client {client_id} R{round_num}: "
        #                 f"Q_loss={Q_loss:.4f}, Q_consistency={Q_consistency:.4f}, "
        #                 f"Q_data={Q_data:.4f}, final={quality:.4f}")
        
        # Track for normalization
        self.consistency_scores_history.append(Q_consistency)
        if len(self.consistency_scores_history) > 100:
            self.consistency_scores_history.pop(0)
            
        return quality
    
    def _calculate_robust_consistency(self, param_update):
        """
        FIX: Robust consistency calculation that handles edge cases
        """
        try:
            # Extract parameter values
            if isinstance(param_update, dict):
                param_values = torch.cat([p.flatten() for p in param_update.values()])
            else:
                param_values = param_update.flatten()
                
            param_np = param_values.detach().cpu().numpy()
            
            if len(param_np) == 0:
                return 0.5
            
            # Remove outliers for more stable calculation
            param_np = param_np[np.abs(param_np) < np.percentile(np.abs(param_np), 95)]
            
            if len(param_np) < 10:  # Too few values
                return 0.5
                
            mean_val = np.mean(param_np)
            std_val = np.std(param_np)
            
            # IMPROVED: Multiple consistency measures
            # 1. Coefficient of variation (robust to scale)
            if abs(mean_val) > 1e-8:
                cv = std_val / abs(mean_val)
                consistency_1 = np.exp(-cv)
            else:
                consistency_1 = 0.5  # Neutral for near-zero updates
                
            # 2. Normalized standard deviation
            param_range = np.max(param_np) - np.min(param_np)
            if param_range > 1e-8:
                normalized_std = std_val / param_range
                consistency_2 = 1.0 - min(1.0, normalized_std)
            else:
                consistency_2 = 1.0  # Perfect consistency for constant values
                
            # 3. Sparsity-based consistency (penalize too many zeros)
            sparsity = np.sum(np.abs(param_np) < 1e-6) / len(param_np)
            sparsity_penalty = 1.0 - min(0.5, sparsity)  # Don't penalize more than 50%
            
            # Combine measures
            Q_consistency = (consistency_1 + consistency_2 + sparsity_penalty) / 3.0
            
        except Exception as e:
            self.logger.warning(f"Consistency calculation failed: {e}, using default")
            Q_consistency = 0.5
            
        return max(0.1, min(1.0, Q_consistency))
    
    def _calculate_data_quality(self, data_sizes, client_id):
        """
        IMPROVED: More nuanced data size quality
        """
        if not data_sizes or client_id not in data_sizes:
            return 0.5
            
        client_data_size = data_sizes[client_id]
        all_sizes = list(data_sizes.values())
        
        if len(all_sizes) <= 1:
            return 1.0
            
        # Use relative positioning instead of just max normalization
        sorted_sizes = sorted(all_sizes)
        client_percentile = (sorted_sizes.index(client_data_size) + 1) / len(sorted_sizes)
        
        # Clients in top 50% get higher data quality scores
        if client_percentile >= 0.5:
            Q_data = 0.5 + 0.5 * ((client_percentile - 0.5) / 0.5)
        else:
            Q_data = 0.3 + 0.2 * (client_percentile / 0.5)
            
        return Q_data

    # Keep legacy methods for compatibility
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


# ALTERNATIVE: Even more generous quality metric for better exploration
class GenerousQualityMetric(QualityMetric):
    """
    EXPLORATION-FRIENDLY: Version that gives higher baseline scores
    to encourage more client participation
    """
    
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):  # CIFAR-100 optimized weights
        super().__init__(alpha, beta, gamma)
        self.baseline_quality = 0.5  # Much higher baseline quality
        
        # Set up file logging to same directory as comprehensive analysis
        from datetime import datetime
        import os
        
        # Create logs directory if it doesn't exist
        log_dir = '../save/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file in same location as comprehensive analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{log_dir}/generous_quality_{timestamp}.log'
        
        # Check if file handler already exists to avoid duplicates
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in self.logger.handlers)
        
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
            
            # Test log to confirm it's working
            self.logger.info("=== GENEROUS QUALITY METRIC INITIALIZED ===")
            self.logger.info(f"Weights: alpha={alpha}, beta={beta}, gamma={gamma}")
            self.logger.info(f"Baseline quality: {self.baseline_quality}")
            self.logger.info("ADDRESSING CIFAR-100 PERFORMANCE ISSUES")
            
        print(f"📝 GenerousQualityMetric logging to: {log_file}")
        
        # Track quality statistics
        self.quality_history = []
        self.client_selections = {}
        
    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        """
        ULTRA-GENEROUS: Much higher baseline quality to fix catastrophic performance
        """
        base_quality = super().calculate_quality(
            loss_before, loss_after, data_sizes, param_update,
            round_num, client_id, all_loss_improvements
        )
        
        # AGGRESSIVE exploration bonus for CIFAR-100
        exploration_bonus = 0.0
        if round_num < 10:  # Extended exploration phase
            exploration_bonus = 0.2 * (10 - round_num) / 10  # Larger bonus
            
        # Client diversity bonus - reward less frequently selected clients
        if client_id not in self.client_selections:
            self.client_selections[client_id] = 0
        self.client_selections[client_id] += 1
        
        # Bonus for underused clients
        avg_selections = np.mean(list(self.client_selections.values())) if self.client_selections else 1
        client_usage = self.client_selections[client_id]
        diversity_bonus = max(0, 0.1 * (avg_selections - client_usage) / avg_selections)
        
        # MUCH more generous quality calculation
        generous_quality = max(
            self.baseline_quality,  # 0.4 baseline
            base_quality + exploration_bonus + diversity_bonus
        )
        
        # Track statistics
        self.quality_history.append(generous_quality)
        
        # # Enhanced logging with performance context
        # self.logger.info(f"R{round_num} Client {client_id}: base={base_quality:.4f}, "
        #                 f"explore_bonus={exploration_bonus:.4f}, "
        #                 f"diversity_bonus={diversity_bonus:.4f}, "
        #                 f"final={generous_quality:.4f}, "
        #                 f"selections={client_usage}")
        
        # Log statistics every 5 rounds for better monitoring
        if round_num % 5 == 0 and len(self.quality_history) > 10:
            recent_qualities = self.quality_history[-50:]
            mean_q = np.mean(recent_qualities)
            std_q = np.std(recent_qualities)
            min_q = np.min(recent_qualities)
            max_q = np.max(recent_qualities)
            
            self.logger.info(f"=== QUALITY STATS ROUND {round_num} ===")
            self.logger.info(f"Recent qualities: mean={mean_q:.4f}, std={std_q:.4f}, "
                           f"min={min_q:.4f}, max={max_q:.4f}")
            self.logger.info(f"Client selections: {len(self.client_selections)} unique clients")
            self.logger.info(f"Selection distribution: min={min(self.client_selections.values())}, "
                           f"max={max(self.client_selections.values())}, "
                           f"avg={np.mean(list(self.client_selections.values())):.1f}")
            
            # CRITICAL: Log if performance is still terrible
            if round_num > 10 and mean_q < 0.5:
                self.logger.warning(f"LOW QUALITY SCORES DETECTED! Mean quality {mean_q:.4f} may cause poor performance")
                
        return min(1.0, generous_quality)





"""
The key insight: Your current harsh quality assessment (mean 0.166) is causing 
over-exploitation. A more generous baseline will improve exploration and should 
boost your overall federated learning performance.
"""
