
import torch
import numpy as np
import logging
from memory_bank import MemoryBank
from intelligent_selector import IntelligentSelector
from quality_metric import QualityMetric
from embedding_generator import EmbeddingGenerator

class PUMBFederatedServer:
    def __init__(self, model, optimizer, loss_fn, args, embedding_dim=512):
        """Initialize enhanced federated server with PUMB."""
        """FIXED: Accept args to access pumb_alpha and other parameters."""
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.args = args  # Store args for later use

        # Initialize PUMB components
        self.memory_bank = MemoryBank(embedding_dim=embedding_dim)
        self.client_selector = IntelligentSelector(
            self.memory_bank,
            initial_rounds=getattr(args, 'pumb_initial_rounds', 3),
            exploration_ratio=getattr(args, 'pumb_exploration_ratio', 0.7)
        )
        self.quality_calc = QualityMetric()
        self.embedding_gen = EmbeddingGenerator(embedding_dim=embedding_dim)

        # Track global direction/state
        self.global_direction = None
        self.prev_model_state = None
        self.current_round = 0

        # Setup logging
        self.logger = logging.getLogger('PUMB_FederatedServer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def select_clients(self, available_clients, num_clients):
        """Select clients for the current round."""
        self.logger.info(f"Round {self.current_round}: Selecting {num_clients} clients from {len(available_clients)} available")
        
        selected = self.client_selector.select_clients(
            available_clients,
            num_clients,
            self.current_round,
            self.global_direction
        )
        
        self.logger.info(f"Selected clients: {selected}")
        return selected


    def update_global_model(self, client_states, client_losses, data_sizes, aggregation_weights=None):
        """FIXED: Update using full client states with provided aggregation weights."""
        selected_clients = list(client_states.keys())
        if not selected_clients:
            return False

        # Use provided weights if available, otherwise calculate them
        if aggregation_weights is None:
            aggregation_weights = self.client_selector.get_aggregation_weights(
                selected_clients,
                client_states,
                data_sizes,
                self.global_direction,
                embedding_gen=self.embedding_gen,
                quality_calc=self.quality_calc,
                current_round=self.current_round
            )

        print(f"Round {self.current_round}: Weights = {aggregation_weights}")
        print(f"Round {self.current_round}: Weight sum = {sum(aggregation_weights.values())}")
        print(f"Round {self.current_round}: Weight std = {np.std(list(aggregation_weights.values()))}")
        
        # Aggregate full parameter states
        new_state_dict = {}
        for name, param in self.model.named_parameters():
            weighted_param = torch.zeros_like(param.data)
            
            for client_id, weight in aggregation_weights.items():
                client_param = client_states[client_id][name]
                weighted_param += weight * client_param
            
            new_state_dict[name] = weighted_param
        
        # Load the aggregated state
        self.model.load_state_dict(new_state_dict)
        
        # Calculate parameter updates for memory bank
        current_state = self._get_model_state_copy()
        if self.prev_model_state is not None:
            param_updates = {name: current_state[name] - self.prev_model_state[name]
                            for name in current_state}
        
        self.prev_model_state = current_state
        self.current_round += 1
        return True

    def _aggregate_updates(self, client_models, aggregation_weights):
        """
        THEORY-ALIGNED: Aggregate full client model states as per Algorithm 1 line 13
        θ^t ← Σ(i∈S_t) w_i^t θ_i^t
        """
        new_state_dict = {}
        for name, param in self.model.named_parameters():
            weighted_param = torch.zeros_like(param.data)
            for client_id, weight in aggregation_weights.items():
                # Use full client model state, not parameter updates
                client_param = client_models[client_id][name]
                weighted_param += weight * client_param
            new_state_dict[name] = weighted_param
        self.model.load_state_dict(new_state_dict)
        self.logger.info("Aggregation complete: full model averaging performed.")


    def _update_memory_bank(self, client_updates, client_losses, data_sizes):
        """Update memory bank with client contributions from this round."""
        successful_updates = 0
        
        for client_id, update in client_updates.items():
            try:
                # Generate embedding for this update
                embedding = self.embedding_gen.generate_embedding(update)
                
                # Calculate quality score
                if client_id in client_losses:
                    loss_before, loss_after = client_losses[client_id]
                    quality = self.quality_calc.calculate_quality(
                        loss_before,
                        loss_after,
                        data_sizes,
                        update,
                        update_norm=None,
                        round_num=self.current_round,
                        client_id=client_id
                    )
                else:
                    quality = 1.0
                    self.logger.warning(f"No loss data for client {client_id}, using default quality=1.0")
                
                # Add to memory bank
                self.memory_bank.add_update(
                    client_id,
                    embedding,
                    quality,
                    self.current_round
                )
                successful_updates += 1
                
            except Exception as e:
                self.logger.error(f"Failed to update memory bank for client {client_id}: {e}")

        self.logger.info(f"Memory bank updated: {successful_updates}/{len(client_updates)} clients successful")

    def _log_update_statistics(self, client_updates, client_losses, data_sizes):
        """Log statistics about client updates."""
        self.logger.info("=== CLIENT UPDATE STATISTICS ===")
        
        # Data size statistics
        if data_sizes:
            sizes = list(data_sizes.values())
            self.logger.info(f"Data sizes: {dict(data_sizes)}")
            self.logger.info(f"Data size stats: total={sum(sizes)}, "
                           f"mean={np.mean(sizes):.1f}, "
                           f"std={np.std(sizes):.1f}")

        # Loss improvement statistics
        if client_losses:
            improvements = []
            for client_id, (loss_before, loss_after) in client_losses.items():
                improvement = loss_before - loss_after
                improvements.append(improvement)
                self.logger.info(f"Client {client_id}: loss {loss_before:.4f} → {loss_after:.4f} "
                               f"(improvement: {improvement:.4f})")
            
            if improvements:
                self.logger.info(f"Loss improvement stats: mean={np.mean(improvements):.4f}, "
                               f"std={np.std(improvements):.4f}, "
                               f"best={max(improvements):.4f}")

        # Update norm statistics
        update_norms = {}
        for client_id, update in client_updates.items():
            total_norm = 0.0
            for param_name, param_update in update.items():
                total_norm += torch.norm(param_update).item() ** 2
            update_norms[client_id] = np.sqrt(total_norm)

        if update_norms:
            norms = list(update_norms.values())
            self.logger.info(f"Update norms: {[(cid, f'{norm:.4f}') for cid, norm in update_norms.items()]}")
            self.logger.info(f"Update norm stats: mean={np.mean(norms):.4f}, "
                           f"std={np.std(norms):.4f}, "
                           f"max={max(norms):.4f}")

    def _log_global_direction_stats(self):
        """Log statistics about the global direction."""
        if self.global_direction is None:
            return
            
        total_norm = 0.0
        param_count = 0
        
        for name, direction in self.global_direction.items():
            norm = torch.norm(direction).item()
            total_norm += norm ** 2
            param_count += 1
            
        global_direction_norm = np.sqrt(total_norm)
        self.logger.info(f"Global direction norm: {global_direction_norm:.6f}")

    def _get_model_state_copy(self):
        """Get a deep copy of current model state."""
        return {name: param.clone().detach()
                for name, param in self.model.named_parameters()}

    def _calculate_direction(self, prev_state, current_state):
        """Calculate the global direction as current - previous parameters."""
        return {name: current_state[name] - prev_state[name]
                for name in prev_state}

    def get_server_stats(self):
        """Get comprehensive server statistics."""
        stats = {
            'current_round': self.current_round,
            'memory_bank_size': len(self.memory_bank.memories),
            'has_global_direction': self.global_direction is not None,
        }
        
        # Add memory bank statistics if available
        if hasattr(self.memory_bank, 'get_statistics'):
            stats.update(self.memory_bank.get_statistics())
            
        return stats

    def save_checkpoint(self, filepath):
        """Save server state checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_round': self.current_round,
            'global_direction': self.global_direction,
            'prev_model_state': self.prev_model_state,
        }
        
        # Add memory bank state if it supports serialization
        if hasattr(self.memory_bank, 'get_state'):
            checkpoint['memory_bank_state'] = self.memory_bank.get_state()
            
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """Load server state from checkpoint."""
        checkpoint = torch.load(filepath)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_round = checkpoint['current_round']
        self.global_direction = checkpoint['global_direction']
        self.prev_model_state = checkpoint['prev_model_state']
        
        # Restore memory bank state if available
        if 'memory_bank_state' in checkpoint and hasattr(self.memory_bank, 'load_state'):
            self.memory_bank.load_state(checkpoint['memory_bank_state'])
            
        self.logger.info(f"Checkpoint loaded from {filepath}, resuming from round {self.current_round}")
