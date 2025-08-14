# memory_bank.py
import numpy as np
import torch.nn.functional as F
import faiss
import torch
from collections import defaultdict, deque

class MemoryBank:
    def __init__(self, embedding_dim=512, max_memories=1000):
        """Initialize the memory bank for storing client parameter update patterns."""
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories

        # Keep your existing FAISS-based storage
        self.memories = []  # Each entry: dict(client_id, embedding, quality, round)
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Keep your existing client statistics
        self.client_quality_history = defaultdict(lambda: deque(maxlen=50))
        self.client_reliability = defaultdict(float)
        self.client_participation = defaultdict(int)
        self.round_count = 0
        
        # ADD: New storage for theory-aligned methods
        self.client_embeddings = defaultdict(list)  # client_id -> list of embeddings
        self.client_qualities = defaultdict(list)   # client_id -> list of qualities
        self.client_rounds = defaultdict(list)      # client_id -> list of round numbers
        self.global_states = {}      # round -> global model state

    def add_update(self, client_id, update_embedding, quality_score, round_num):
        # Your existing FAISS logic (keep unchanged)
        if isinstance(update_embedding, torch.Tensor):
            update_embedding = update_embedding.cpu().numpy()
        
        if len(update_embedding.shape) == 1:
            update_embedding = update_embedding.reshape(1, -1)
            
        if update_embedding.shape[1] != self.embedding_dim:
            if update_embedding.shape[1] < self.embedding_dim:
                padding = np.zeros((1, self.embedding_dim - update_embedding.shape[1]))
                update_embedding = np.hstack([update_embedding, padding])
            else:
                update_embedding = update_embedding[:, :self.embedding_dim]
                
        update_embedding = update_embedding.reshape(1, self.embedding_dim).astype(np.float32)

        # Store in memory and FAISS (your existing logic)
        memory_entry = {
            'client_id': client_id,
            'embedding': update_embedding.flatten(),
            'quality': quality_score,
            'round': round_num
        }
        self.memories.append(memory_entry)
        self.index.add(update_embedding)

        # Memory size management (your existing logic)
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)
            self._rebuild_index()

        # Update your existing stats
        self.client_quality_history[client_id].append(quality_score)
        self.client_participation[client_id] += 1
        
        # ADD: Also store in theory-aligned format for new methods
        self.client_embeddings[client_id].append(update_embedding.flatten())
        self.client_qualities[client_id].append(quality_score)
        self.client_rounds[client_id].append(round_num)
        
        # Maintain max memories limit for new storage
        if len(self.client_embeddings[client_id]) > self.max_memories:
            self.client_embeddings[client_id].pop(0)
            self.client_qualities[client_id].pop(0)
            self.client_rounds[client_id].pop(0)
        
        # Update reliability using new method
        self.update_client_reliability_theory_aligned(client_id)
        # FIX: REMOVE this line - round count should only be updated once per round
        # self.round_count = max(self.round_count, round_num + 1)

    def update_round_count(self):
        """Increment the round counter - call this once per round."""
        self.round_count += 1
        #print(f"Memory Bank: Round count updated to {self.round_count}")

    def store_global_state(self, round_num, global_state):
        """Store global model state for a round."""
        # FIX: Ensure all tensors are moved to CPU for consistent device handling
        if isinstance(global_state, dict):
            stored_state = {}
            for name, param in global_state.items():
                if isinstance(param, torch.Tensor):
                    # CRITICAL FIX: Always move to CPU and detach
                    stored_state[name] = param.cpu().detach().clone()
                else:
                    stored_state[name] = param
            self.global_states[round_num] = stored_state
        else:
            if isinstance(global_state, torch.Tensor):
                self.global_states[round_num] = global_state.cpu().detach().clone()
            else:
                self.global_states[round_num] = global_state
        #print(f"Memory Bank: Stored global state for round {round_num}")
        
    # Keep your existing methods unchanged
    def _rebuild_index(self):
        """Rebuild FAISS index after popping old memories."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        for entry in self.memories:
            emb = entry['embedding'].reshape(1, self.embedding_dim).astype(np.float32)
            self.index.add(emb)

    def update_client_reliability(self, client_id):
        """Your existing reliability calculation (keep for compatibility)."""
        quality_scores = list(self.client_quality_history[client_id])
        if not quality_scores:
            self.client_reliability[client_id] = 0.0
            return 0.0
        participation_bonus = min(1.0, self.client_participation[client_id] / 10)
        weights = np.exp(np.linspace(0, 1, len(quality_scores)))
        weighted_quality = np.average(quality_scores, weights=weights)
        reliability = weighted_quality * (1 + participation_bonus)
        self.client_reliability[client_id] = reliability
        return reliability
    
    def update_client_reliability_theory_aligned(self, client_id):
        """
        THEORY-ALIGNED: reliability_i^t = q̄_recent_i · log(1 + participation_count_i)
        """
        if client_id not in self.client_qualities or not self.client_qualities[client_id]:
            self.client_reliability[client_id] = 0.1
            return 0.1
        
        # Recent average quality (last 5 rounds)
        recent_qualities = self.client_qualities[client_id][-5:]
        q_recent = np.mean(recent_qualities)
        
        # Participation count
        participation_count = len(self.client_qualities[client_id])
        
        # Reliability formula from paper
        reliability = q_recent * np.log(1 + participation_count)
        reliability = max(0.1, min(2.0, reliability))  # Reasonable bounds
        
        self.client_reliability[client_id] = reliability
        return reliability

    def get_client_reliability(self, client_id):
        return self.client_reliability.get(client_id, 0.0)


    def compute_similarity(self, client_id, current_embedding):
        """Compute similarity with historical embeddings using AVERAGE instead of MAX."""
        if client_id not in self.client_embeddings or len(self.client_embeddings[client_id]) == 0:
            #print(f"  No history for client {client_id}, using default similarity")
            return 0.5
        
        similarities = []
        for i, historical_embedding in enumerate(self.client_embeddings[client_id]):
            # Ensure both are torch tensors
            if isinstance(current_embedding, np.ndarray):
                curr_emb = torch.from_numpy(current_embedding).float()
            else:
                curr_emb = current_embedding.float()
            if isinstance(historical_embedding, np.ndarray):
                hist_emb = torch.from_numpy(historical_embedding).float()
            else:
                hist_emb = historical_embedding.float()
            sim = F.cosine_similarity(curr_emb.unsqueeze(0), hist_emb.unsqueeze(0)).item()
            similarities.append(sim)
            #print(f"  Client {client_id} vs history[{i}]: {sim:.4f}")

        # FIX: Use average similarity instead of maximum
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            #print(f"Client {client_id} average similarity: {avg_similarity:.4f}")

            # Optional: Apply slight penalty for very high similarities to encourage diversity
            if avg_similarity > 0.95:
                final_similarity = avg_similarity * 0.9  # 10% penalty for very high similarity
            else:
                final_similarity = avg_similarity

            #print(f"Client {client_id} final similarity: {final_similarity:.4f}")
            return final_similarity
        else:
            return 0.5
        
    def get_recent_qualities(self, client_id, window=3):
        """Get recent quality scores for a client."""
        if client_id not in self.client_qualities:
            return []
        return self.client_qualities[client_id][-window:]

    def get_last_global_state(self):
        """Get the last stored global state."""
        if not self.global_states:
            return {}
        last_round = max(self.global_states.keys())
        return self.global_states[last_round]

    def get_similar_updates(self, query_embedding, k=5):
        """Find similar parameter updates to the query embedding."""
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        query_embedding = query_embedding.reshape(1, self.embedding_dim).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memories):
                memory = self.memories[idx]
                results.append((distances[0][i], memory['client_id'], memory))
        return results

    def get_top_reliable_clients(self, n=10):
        sorted_clients = sorted(
            self.client_reliability.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [client_id for client_id, _ in sorted_clients[:n]]

    def get_client_statistics(self, client_id):
        if client_id not in self.client_participation:
            return None
        scores = list(self.client_quality_history[client_id])
        stats = {
            'participation_count': self.client_participation[client_id],
            'reliability_score': self.client_reliability[client_id],
            'avg_quality': np.mean(scores) if scores else 0,
            'quality_trend': self.calculate_trend(scores),
            'recent_quality': scores[-1] if scores else 0,
        }
        return stats

    def calculate_trend(self, values, window=5):
        if len(values) < 2:
            return 0.0
        recent = values[-min(window, len(values)):]
        if len(recent) < 2:
            return 0.0
        x = np.arange(len(recent))
        y = np.array(recent)
        return np.polyfit(x, y, 1)[0]
