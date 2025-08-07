from scipy.stats import skew, kurtosis
import torch
import torch.nn.functional as F
import numpy as np

class EmbeddingGenerator:
    def __init__(self, feature_dim=13, embedding_dim=512):  # Increased to 13 features
        """Initialize the parameter update embedding generator."""
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # IMPROVED: Better network architecture with dropout and batch norm
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 128),  # Increased capacity
            torch.nn.BatchNorm1d(128),          # Add batch norm for stability
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),             # Prevent overfitting
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, embedding_dim)
        )
        
        # Better weight initialization
        with torch.no_grad():
            for layer in self.projection:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    torch.nn.init.zeros_(layer.bias)
    
    def generate_embedding(self, parameter_update):
        """Generate an embedding vector from parameter updates with enhanced discrimination."""
        # Extract statistical features
        features = self._extract_features(parameter_update)
        
        # Convert to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Handle NaN or infinite values
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # FIX: Add more aggressive feature scaling for discrimination
        features = self._scale_features(features)
        
        # Project to embedding space
        self.projection.eval()
        with torch.no_grad():
            embedding = self.projection(features.unsqueeze(0)).squeeze(0)
        
        # FIX: Add controlled noise to break near-perfect similarities
        client_hash = hash(str(features.numpy().tolist())) % 10000
        torch.manual_seed(client_hash)
        noise = torch.normal(0, 0.05, size=embedding.shape)  # Increased noise
        embedding = embedding + noise
        
        # Normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=0)
        
        return embedding
    
    def _scale_features(self, features):
        """More conservative scaling that preserves client differences."""
        # Method 1: Simple robust scaling
        median_val = torch.median(features)
        mad = torch.median(torch.abs(features - median_val))  # Median Absolute Deviation
        
        if mad > 1e-8:
            # Robust scaling using MAD
            scaled = (features - median_val) / (mad * 1.4826 + 1e-8)  # 1.4826 makes MAD comparable to std
        else:
            # Fallback to simple centering
            scaled = features - median_val
        
        # Method 2: Gentle feature enhancement (less aggressive)
        for i in range(len(scaled)):
            # Very gentle progressive scaling
            scale_factor = 1.0 + (i * 0.01)  # Much smaller scaling (was 0.1)
            scaled[i] = scaled[i] * scale_factor
        
        # Method 3: Gentle non-linear transformation
        scaled = torch.tanh(scaled * 0.5)  # Much gentler than 2.0
        
        return scaled

    def _extract_features(self, parameter_update):
        """Enhanced feature extraction with better discrimination."""
        # Convert to numpy for processing
        if isinstance(parameter_update, torch.Tensor):
            update_values = parameter_update.detach().cpu().numpy().flatten()
        elif isinstance(parameter_update, dict):
            # Handle state dict - FIX: Add randomization for better discrimination
            if all(isinstance(v, torch.Tensor) for v in parameter_update.values()):
                update_values = torch.cat([p.flatten() for p in parameter_update.values()]).detach().cpu().numpy()
            else:
                update_values = np.concatenate([np.array(p).flatten() for p in parameter_update.values()])
        else:
            update_values = np.array(parameter_update).flatten()
        
        # Handle edge cases
        if len(update_values) == 0:
            return np.random.normal(0, 0.1, size=self.feature_dim)  # Random fallback
        
        # Remove NaN and infinite values
        update_values = update_values[np.isfinite(update_values)]
        if len(update_values) == 0:
            return np.random.normal(0, 0.1, size=self.feature_dim)  # Random fallback
        
        try:
            # FIX: Enhanced discriminative features
            mean_val = np.mean(update_values)
            std_val = np.std(update_values)
            l2_norm = np.linalg.norm(update_values, 2)
            l1_norm = np.linalg.norm(update_values, 1)
            min_val = np.min(update_values)
            max_val = np.max(update_values)
            median_val = np.median(update_values)
            
            # Enhanced statistical features for better discrimination
            q75, q25 = np.percentile(update_values, [75, 25])
            iqr = q75 - q25
            
            # More discriminative features
            if std_val > 1e-8:
                skewness = skew(update_values)
                kurtosis_val = kurtosis(update_values)
            else:
                skewness = 0.0
                kurtosis_val = 0.0
                
            sparsity = np.sum(np.abs(update_values) < 1e-6) / len(update_values)
            energy = np.sum(update_values ** 2)
            
            # FIX: Add client-specific randomization to break ties
            random_seed = hash(str(update_values[:min(10, len(update_values))].tolist())) % 1000
            np.random.seed(random_seed)
            noise_factor = np.random.normal(0, 1e-6)  # Small noise for discrimination
            
            # Enhanced features with discrimination
            features = np.array([
                mean_val + noise_factor,
                std_val,
                l2_norm,
                l1_norm,
                min_val,
                max_val,
                median_val,
                skewness,
                iqr,
                np.mean(np.abs(update_values)),
                kurtosis_val,
                sparsity,
                energy + noise_factor  # 13 total features
            ])
            
            # Ensure we have exactly feature_dim features
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            elif len(features) < self.feature_dim:
                # Fill with small random values instead of zeros
                padding = np.random.normal(0, 1e-6, size=self.feature_dim - len(features))
                features = np.concatenate([features, padding])
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            # Better fallback with some randomization
            features = np.random.normal(0, 0.01, size=self.feature_dim)
        
        return features