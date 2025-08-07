# intelligent_selector.py
import numpy as np
from collections import Counter, defaultdict
from embedding_generator import EmbeddingGenerator
from quality_metric import QualityMetric
from memory_bank import MemoryBank
import logging
import torch


class IntelligentSelector:
    def __init__(self, memory_bank, initial_rounds=3, exploration_ratio=0.4):
        """Initialize the intelligent client selector.
        
        Args:
            memory_bank: Reference to the memory bank object
            initial_rounds: Number of initial rounds for cold start
            exploration_ratio: Ratio of random client selection (exploration)
        """
        self.memory_bank = memory_bank
        self.initial_rounds = initial_rounds
        self.exploration_ratio = exploration_ratio
        
        # Enhanced logging attributes
        self.selection_history = []
        self.client_selection_count = Counter()
        self.quality_score_history = defaultdict(list)
        self.selection_reasons = []
        self.logger = logging.getLogger('PUMB_Selection')
        self.round_count = 0  # Track rounds for logging

    def select_clients(self, available_clients, num_to_select, round_num, 
                       global_direction=None):
        """Select clients for the current round based on memory bank data.
        
        Args:
            available_clients: List of available client IDs
            num_to_select: Number of clients to select
            round_num: Current federated learning round
            global_direction: Current global update direction (optional)
            
        Returns:
            selected_clients: List of selected client IDs
        """
        self.round_count = round_num  # Update round count for logging

        # Cold start: Random selection for initial rounds
        if round_num < self.initial_rounds:
            selected = self._random_selection(available_clients, num_to_select)
            self.logger.info(f"Round {round_num}: Cold start - randomly selected {selected}")
            return selected
        
        # Hybrid selection strategy
        num_exploit = int(num_to_select * (1 - self.exploration_ratio))
        num_explore = num_to_select - num_exploit
        
        # Ensure at least one client for each strategy
        num_exploit = max(1, min(num_exploit, len(available_clients) - 1))
        num_explore = num_to_select - num_exploit
        
        # Exploitation: Select clients with best reliability scores
        exploit_candidates = []
        for client_id in available_clients:
            reliability = self.memory_bank.get_client_reliability(client_id)
            exploit_candidates.append((client_id, reliability))
        
        # Sort by reliability (descending)
        exploit_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get top reliable clients
        exploit_clients = [client_id for client_id, _ in exploit_candidates[:num_exploit]]
        
        # Remove selected clients from the pool for exploration
        remaining_clients = [c for c in available_clients if c not in exploit_clients]
        
        # Exploration: Random selection from remaining clients
        explore_clients = self._random_selection(remaining_clients, num_explore)
        
        # Combine selections
        selected_clients = exploit_clients + explore_clients

        # Enhanced logging
        self._log_selection_details(selected_clients, available_clients, num_to_select)

        return selected_clients
    
    def _random_selection(self, client_pool, num_to_select):
        """Randomly select clients from the pool."""
        if not client_pool:
            return []
            
        if num_to_select >= len(client_pool):
            return client_pool.copy()
            
        return np.random.choice(client_pool, num_to_select, replace=False).tolist()
    
    def _log_selection_details(self, selected_clients, available_clients, num_clients):
        # Update counters
        self.selection_history.append(selected_clients.copy())
        for client in selected_clients:
            self.client_selection_count[client] += 1

        # Get comprehensive client information
        all_scores = {}
        all_reliability = {}
        all_trends = {}
        
        for client_id in available_clients:
            client_stats = self.memory_bank.get_client_statistics(client_id)
            if client_stats is not None:
                score = client_stats['recent_quality']
                reliability = client_stats['reliability_score']
                trend = client_stats['quality_trend']
            else:
                score = 0.0
                reliability = 0.0
                trend = 0.0
            
            all_scores[client_id] = score
            all_reliability[client_id] = reliability
            all_trends[client_id] = trend
            self.quality_score_history[client_id].append(score)

        # Log selection details
        selected_scores = [all_scores[cid] for cid in selected_clients]
        selected_reliability = [all_reliability[cid] for cid in selected_clients]
        unselected_clients = [cid for cid in available_clients if cid not in selected_clients]
        unselected_scores = [all_scores[cid] for cid in unselected_clients] if unselected_clients else []

        self.logger.info(f"=== ROUND {self.round_count} CLIENT SELECTION ===")
        self.logger.info(f"Exploration ratio: {self.exploration_ratio:.3f}")
        self.logger.info(f"Available clients: {len(available_clients)}, Selecting: {num_clients}")
        self.logger.info(f"Selected clients: {selected_clients}")
        self.logger.info(f"Selected scores: {[f'{s:.3f}' for s in selected_scores]}")
        self.logger.info(f"Selected reliability: {[f'{r:.3f}' for r in selected_reliability]}")
        
        if selected_scores:
            self.logger.info(f"Selected score stats: mean={np.mean(selected_scores):.3f}, "
                            f"std={np.std(selected_scores):.3f}, "
                            f"min={np.min(selected_scores):.3f}, "
                            f"max={np.max(selected_scores):.3f}")

        if unselected_scores:
            self.logger.info(f"Unselected score stats: mean={np.mean(unselected_scores):.3f}, "
                            f"std={np.std(unselected_scores):.3f}")

        # Show top reliable clients
        top_reliable = self.memory_bank.get_top_reliable_clients(5)
        self.logger.info(f"Top 5 reliable clients: {top_reliable}")

        # Rest of the logging code remains the same...
        if len(self.selection_history) >= 5:
            recent_selections = self.selection_history[-5:]
            frequent_clients = Counter()
            for round_clients in recent_selections:
                for client in round_clients:
                    frequent_clients[client] += 1
            self.logger.info(f"Most frequent clients (last 5 rounds): {frequent_clients.most_common(5)}")

        if self.round_count > 3:
            trending_up = []
            trending_down = []
            for client_id in selected_clients:
                trend = all_trends[client_id]
                if trend > 0.05:
                    trending_up.append((client_id, trend))
                elif trend < -0.05:
                    trending_down.append((client_id, trend))
            if trending_up:
                self.logger.info(f"Clients trending UP: {trending_up}")
            if trending_down:
                self.logger.info(f"Clients trending DOWN: {trending_down}")

    def get_aggregation_weights(self, selected_clients, client_models, data_sizes, 
                            global_direction=None, embedding_gen=None, 
                            quality_calc=None, current_round=0):
        """
        FIXED: Theory-aligned weight calculation with proper round count and state storage
        w_i^t = (reliability_i^t · similarity_i^t · quality_i^t) / Σ_j(...)
        """
        if not selected_clients:
            return {}
        
        weights = {}
        weight_components = {}  # For debugging
        
        # FIX 1: Use memory bank's round count instead of checking conditions manually
        print(f"DEBUG: Current round = {current_round}, Memory bank round_count = {self.memory_bank.round_count}")
        
        for client_id in selected_clients:
            # 1. Reliability from memory bank
            reliability = max(0.1, self.memory_bank.get_client_reliability(client_id))
            
            # 2. Similarity computation - FIX: Use memory bank round count
            similarity = 0.5  # Default for early rounds or no history
            
            # FIX 2: Check memory bank round count properly and ensure global states exist
            if self.memory_bank.round_count > 5 and embedding_gen:
                try:
                    # FIX 3: Ensure global states are available before computing similarity
                    if (hasattr(self.memory_bank, 'global_states') and 
                        self.memory_bank.global_states and 
                        len(self.memory_bank.global_states) > 0):
                        
                        last_round = max(self.memory_bank.global_states.keys())
                        last_global_state = self.memory_bank.global_states[last_round]
                        
                        print(f"DEBUG: Computing similarity for client {client_id}, last_round={last_round}")
                        
                        # FIX: Compute parameter update with proper device handling
                        param_update = {}
                        for name in client_models[client_id]:
                            if name in last_global_state:
                                # Ensure both tensors are on the same device (CPU)
                                current_param = client_models[client_id][name].cpu()
                                previous_param = last_global_state[name].cpu()
                                param_update[name] = current_param - previous_param
                            else:
                                # If parameter doesn't exist in previous state, use current value
                                param_update[name] = client_models[client_id][name].cpu()
                        
                        # Generate embedding and compute similarity
                        current_embedding = embedding_gen.generate_embedding(param_update)
                        similarity = self.memory_bank.compute_similarity(client_id, current_embedding)
                        
                        print(f"DEBUG: Client {client_id} computed similarity: {similarity:.3f}")
                        
                    else:
                        print(f"DEBUG: No global states available for similarity computation")
                        similarity = 0.5  # Keep default
                        
                except Exception as e:
                    print(f"ERROR: Similarity computation failed for client {client_id}: {e}")
                    similarity = 0.5  # Fallback
            else:
                if self.memory_bank.round_count <= 5:
                    print(f"DEBUG: Round {self.memory_bank.round_count} <= 5, using default similarity")
                else:
                    print(f"DEBUG: No embedding generator provided")
            
            # 3. Current quality (use recent average)
            recent_qualities = self.memory_bank.get_recent_qualities(client_id, window=3)
            quality = np.mean(recent_qualities) if recent_qualities else 0.5
            
            # 4. Multiplicative combination as per theory
            weight = reliability * similarity * quality
            weights[client_id] = weight
            
            # Store components for debugging
            weight_components[client_id] = {
                'reliability': reliability,
                'similarity': similarity, 
                'quality': quality,
                'final_weight': weight
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {client_id: w / total_weight for client_id, w in weights.items()}
        else:
            # Fallback to uniform weights
            uniform_weight = 1.0 / len(selected_clients)
            weights = {client_id: uniform_weight for client_id in selected_clients}
        
        # Enhanced debugging
        print(f"\n=== ROUND {current_round} WEIGHT CALCULATION ===")
        for client_id in selected_clients:
            comp = weight_components[client_id]
            print(f"Client {client_id}: R={comp['reliability']:.3f}, S={comp['similarity']:.3f}, "
                f"Q={comp['quality']:.3f} → W={weights[client_id]:.3f}")
        
        weight_std = np.std(list(weights.values()))
        print(f"Weight std: {weight_std:.4f} (higher = more differentiation)")
        
        return weights