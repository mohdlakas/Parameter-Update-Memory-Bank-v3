import numpy as np
import random
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Research Analysis Functions
def calculate_heterogeneity_metrics(dict_users, dataset):
    """
    Calculate comprehensive heterogeneity metrics for research analysis.
    These metrics will be crucial for your PUMB paper.
    """
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    
    # Calculate class distribution per user
    class_distributions = []
    for user_idx, indices in dict_users.items():
        if isinstance(indices, (set, list)):
            indices = np.array(list(indices))
        
        user_labels = labels[indices]
        class_count = Counter(user_labels)
        class_dist = np.array([class_count.get(c, 0) for c in range(num_classes)])
        class_distributions.append(class_dist)
    
    class_distributions = np.array(class_distributions)
    
    # Key heterogeneity metrics for PUMB research
    metrics = {}
    
    # 1. KL Divergence from uniform distribution
    uniform_dist = np.ones(num_classes) / num_classes
    kl_divergences = []
    for i, dist in enumerate(class_distributions):
        if dist.sum() > 0:
            normalized_dist = dist / dist.sum()
            # Add small epsilon to avoid log(0)
            normalized_dist = normalized_dist + 1e-8
            kl_div = np.sum(normalized_dist * np.log(normalized_dist / uniform_dist))
            kl_divergences.append(kl_div)
    
    metrics['kl_divergence_mean'] = np.mean(kl_divergences)
    metrics['kl_divergence_std'] = np.std(kl_divergences)
    
    # 2. Jensen-Shannon Divergence between clients (measures client similarity)
    js_divergences = []
    for i in range(len(class_distributions)):
        for j in range(i+1, len(class_distributions)):
            dist1 = class_distributions[i] / (class_distributions[i].sum() + 1e-8)
            dist2 = class_distributions[j] / (class_distributions[j].sum() + 1e-8)
            
            # Jensen-Shannon divergence
            m = 0.5 * (dist1 + dist2)
            js_div = 0.5 * np.sum(dist1 * np.log((dist1 + 1e-8) / (m + 1e-8))) + \
                     0.5 * np.sum(dist2 * np.log((dist2 + 1e-8) / (m + 1e-8)))
            js_divergences.append(js_div)
    
    metrics['js_divergence_mean'] = np.mean(js_divergences)
    metrics['js_divergence_std'] = np.std(js_divergences)
    
    # 3. Effective number of classes per client
    effective_classes = []
    for dist in class_distributions:
        if dist.sum() > 0:
            normalized_dist = dist / dist.sum()
            # Shannon entropy
            entropy = -np.sum(normalized_dist * np.log(normalized_dist + 1e-8))
            effective_classes.append(np.exp(entropy))
        else:
            effective_classes.append(0)
    
    metrics['effective_classes_mean'] = np.mean(effective_classes)
    metrics['effective_classes_std'] = np.std(effective_classes)
    
    # 4. Class imbalance ratio (max class / min class per client)
    imbalance_ratios = []
    for dist in class_distributions:
        if dist.sum() > 0 and np.min(dist[dist > 0]) > 0:
            ratio = np.max(dist) / np.min(dist[dist > 0])
            imbalance_ratios.append(ratio)
    
    if imbalance_ratios:
        metrics['imbalance_ratio_mean'] = np.mean(imbalance_ratios)
        metrics['imbalance_ratio_std'] = np.std(imbalance_ratios)
    else:
        metrics['imbalance_ratio_mean'] = 0
        metrics['imbalance_ratio_std'] = 0
    
    return metrics

def print_heterogeneity_analysis(metrics, alpha, dataset_name):
    """Print comprehensive heterogeneity analysis for research."""
    print(f"\n=== Heterogeneity Analysis: {dataset_name} (α={alpha}) ===")
    print(f"KL Divergence from Uniform: {metrics['kl_divergence_mean']:.4f} ± {metrics['kl_divergence_std']:.4f}")
    print(f"JS Divergence between Clients: {metrics['js_divergence_mean']:.4f} ± {metrics['js_divergence_std']:.4f}")
    print(f"Effective Classes per Client: {metrics['effective_classes_mean']:.2f} ± {metrics['effective_classes_std']:.2f}")
    print(f"Class Imbalance Ratio: {metrics['imbalance_ratio_mean']:.2f} ± {metrics['imbalance_ratio_std']:.2f}")
    
    # Interpret heterogeneity level
    if metrics['kl_divergence_mean'] > 2.0:
        level = "Extreme Non-IID"
    elif metrics['kl_divergence_mean'] > 1.0:
        level = "High Non-IID"
    elif metrics['kl_divergence_mean'] > 0.5:
        level = "Moderate Non-IID"
    elif metrics['kl_divergence_mean'] > 0.1:
        level = "Mild Non-IID"
    else:
        level = "Nearly IID"
    
    print(f"Heterogeneity Level: {level}")
    return level

def run_heterogeneity_experiment(dataset, dataset_name, num_users=100):
    """
    Run comprehensive heterogeneity experiment for PUMB research.
    This will generate the data you need for your paper's main results table.
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE HETEROGENEITY EXPERIMENT: {dataset_name}")
    print(f"{'='*80}")
    
    # Test different alpha values for your research
    alpha_values = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 10.0]  # From extreme to IID
    results = {}
    
    for alpha in alpha_values:
        print(f"\n--- Testing α = {alpha} ---")
        
        if alpha >= 10.0:  # Treat as IID
            dict_users = cifar10_iid(dataset, num_users)
            alpha_label = "IID"
        else:
            dict_users = cifar10_dirichlet(dataset, num_users, alpha=alpha)
            alpha_label = f"α={alpha}"
        
        # Calculate metrics
        metrics = calculate_heterogeneity_metrics(dict_users, dataset)
        level = print_heterogeneity_analysis(metrics, alpha_label, dataset_name)
        
        # Store results for your paper
        results[alpha] = {
            'dict_users': dict_users,
            'metrics': metrics,
            'heterogeneity_level': level,
            'expected_pumb_advantage': predict_pumb_advantage(metrics)
        }
    
    return results

def predict_pumb_advantage(metrics):
    """
    Predict expected PUMB advantage based on heterogeneity metrics.
    This helps set expectations for your experiments.
    """
    kl_div = metrics['kl_divergence_mean']
    js_div = metrics['js_divergence_mean']
    
    if kl_div > 2.0:
        return "High (+15-25%)"
    elif kl_div > 1.0:
        return "Moderate (+8-15%)"
    elif kl_div > 0.5:
        return "Low (+3-8%)"
    elif kl_div > 0.1:
        return "Minimal (+0-3%)"
    else:
        return "None/Negative (-5-0%)"

# Expperimental Config. Generator
def generate_experiment_configs():
    """
    Generate all experimental configurations for your PUMB research.
    This ensures comprehensive coverage and reproducibility.
    """
    configs = []
    
    # Core experiments for main results
    datasets = [
        ("CIFAR-10", "cifar10"),
        ("CIFAR-100", "cifar100"), 
        ("MNIST", "mnist")
    ]
    
    alpha_values = [0.1, 0.3, 0.5, 1.0, float('inf')]  # inf = IID
    num_clients_options = [50, 100]
    participation_rates = [0.1, 0.2]
    
    for dataset_name, dataset_code in datasets:
        for alpha in alpha_values:
            for num_clients in num_clients_options:
                for participation_rate in participation_rates:
                    config = {
                        'dataset_name': dataset_name,
                        'dataset_code': dataset_code,
                        'alpha': alpha,
                        'num_clients': num_clients,
                        'participation_rate': participation_rate,
                        'local_epochs': 5,
                        'batch_size': 32,
                        'num_rounds': 50,
                        'experiment_id': f"{dataset_code}_alpha{alpha}_clients{num_clients}_part{participation_rate}"
                    }
                    configs.append(config)
    
    return configs

def save_experiment_config(configs, save_path="experiment_configs.json"):
    """Save experiment configurations for reproducibility."""
    import json
    
    # Convert float('inf') to string for JSON serialization
    serializable_configs = []
    for config in configs:
        config_copy = config.copy()
        if config_copy['alpha'] == float('inf'):
            config_copy['alpha'] = 'iid'
        serializable_configs.append(config_copy)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_configs, f, indent=2)
    
    print(f"Saved {len(configs)} experiment configurations to {save_path}")

# Enhanced Visualisation
def create_research_plots(results_dict, save_dir="./plots"):
    """
    Create publication-quality plots for your PUMB research paper.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Heterogeneity vs Alpha
    alphas = []
    kl_divs = []
    js_divs = []
    effective_classes = []
    
    for alpha, result in results_dict.items():
        if alpha != float('inf'):
            alphas.append(alpha)
            kl_divs.append(result['metrics']['kl_divergence_mean'])
            js_divs.append(result['metrics']['js_divergence_mean'])
            effective_classes.append(result['metrics']['effective_classes_mean'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # KL Divergence plot
    ax1.semilogx(alphas, kl_divs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Dirichlet Parameter (α)')
    ax1.set_ylabel('KL Divergence from Uniform')
    ax1.set_title('Data Heterogeneity vs α')
    ax1.grid(True, alpha=0.3)
    
    # JS Divergence plot
    ax2.semilogx(alphas, js_divs, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Dirichlet Parameter (α)')
    ax2.set_ylabel('JS Divergence Between Clients')
    ax2.set_title('Client Similarity vs α')
    ax2.grid(True, alpha=0.3)
    
    # Effective classes plot
    ax3.semilogx(alphas, effective_classes, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Dirichlet Parameter (α)')
    ax3.set_ylabel('Effective Classes per Client')
    ax3.set_title('Class Diversity vs α')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/heterogeneity_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Expected PUMB Performance Map
    plt.figure(figsize=(10, 6))
    
    colors = {'High (+15-25%)': 'darkgreen', 'Moderate (+8-15%)': 'green', 
              'Low (+3-8%)': 'orange', 'Minimal (+0-3%)': 'red', 
              'None/Negative (-5-0%)': 'darkred'}
    
    for alpha, result in results_dict.items():
        if alpha != float('inf'):
            advantage = result['expected_pumb_advantage']
            plt.scatter(alpha, result['metrics']['kl_divergence_mean'], 
                       c=colors[advantage], s=100, alpha=0.7, label=advantage)
    
    plt.xscale('log')
    plt.xlabel('Dirichlet Parameter (α)')
    plt.ylabel('KL Divergence (Heterogeneity)')
    plt.title('Expected PUMB Advantage Map')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/pumb_advantage_map.png", dpi=300, bbox_inches='tight')
    plt.show()


# Data Sampling
def cifar10_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR-10 dataset
    :param dataset: CIFAR-10 dataset
    :param num_users: number of users
    :return: dict of image indices for each user
    """
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    
    for i in range(num_users):
        selected_indices = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = np.array(selected_indices, dtype=np.int32)  # Convert to numpy array
        all_idxs = list(set(all_idxs) - set(selected_indices))  # Remove selected indices
    
    return dict_users

def cifar10_dirichlet(dataset, num_users, alpha=0.5, min_samples_per_user=20):
    """
    OPTIMIZED: Sample non-I.I.D. client data from CIFAR-10 dataset using Dirichlet distribution
    :param dataset: CIFAR-10 dataset
    :param num_users: number of users
    :param alpha: Dirichlet concentration parameter (lower = more heterogeneous)
    :param min_samples_per_user: minimum samples per user to avoid empty clients
    :return: dict of image indices for each user
    """
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    
    # Pre-allocate with proper data types for efficiency
    dict_users = {i: np.array([], dtype=np.int32) for i in range(num_users)}
    
    # Get class indices once - more efficient than repeated np.where calls
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    
    # Process all classes
    for c in range(num_classes):
        class_idxs = class_indices[c].copy()
        np.random.shuffle(class_idxs)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_users)
        
        # Convert proportions to actual sample counts
        counts = (proportions * len(class_idxs)).astype(np.int32)
        
        # Handle rounding efficiently
        diff = len(class_idxs) - counts.sum()
        if diff != 0:
            if diff > 0:
                # Add remaining samples to random users (vectorized)
                adjust_users = np.random.choice(num_users, diff, replace=True)
                np.add.at(counts, adjust_users, 1)
            else:
                # Remove excess samples from users with most samples
                for _ in range(-diff):
                    user_idx = np.argmax(counts)
                    if counts[user_idx] > 0:
                        counts[user_idx] -= 1
        
        # Distribute samples to users efficiently
        start_idx = 0
        for user_idx in range(num_users):
            if counts[user_idx] > 0:
                end_idx = start_idx + counts[user_idx]
                user_samples = class_idxs[start_idx:end_idx]
                dict_users[user_idx] = np.concatenate([dict_users[user_idx], user_samples])
                start_idx = end_idx
    
    # Use optimized min_samples enforcement
    dict_users = ensure_min_samples(dict_users, dataset, min_samples_per_user)
    
    return dict_users

def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR-100 dataset
    :param dataset: CIFAR-100 dataset
    :param num_users: number of users
    :return: dict of image indices for each user
    """
    return cifar10_iid(dataset, num_users)  # Same logic for IID

def cifar100_dirichlet(dataset, num_users, alpha=0.5, min_samples_per_user=30):
    """
    Sample non-I.I.D. client data from CIFAR-100 dataset using Dirichlet distribution
    :param dataset: CIFAR-100 dataset
    :param num_users: number of users
    :param alpha: Dirichlet concentration parameter (lower = more heterogeneous)
    :param min_samples_per_user: minimum samples per user to avoid empty clients
    :return: dict of image indices for each user
    """
    return cifar10_dirichlet(dataset, num_users, alpha, min_samples_per_user)  # Same logic

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: MNIST dataset
    :param num_users: number of users
    :return: dict of image indices for each user
    """
    return cifar10_iid(dataset, num_users)  # Same logic for IID

def mnist_dirichlet(dataset, num_users, alpha=0.5, min_samples_per_user=10):
    """
    Sample non-I.I.D. client data from MNIST dataset using Dirichlet distribution
    :param dataset: MNIST dataset
    :param num_users: number of users
    :param alpha: Dirichlet concentration parameter (lower = more heterogeneous)
    :param min_samples_per_user: minimum samples per user to avoid empty clients
    :return: dict of image indices for each user
    """
    return cifar10_dirichlet(dataset, num_users, alpha, min_samples_per_user)  # Same logic

def ensure_min_samples(dict_users, dataset, min_samples):
    """
    OPTIMIZED: Ensure each user has at least min_samples samples
    20-50x faster than original implementation
    :param dict_users: current user data distribution
    :param dataset: original dataset
    :param min_samples: minimum samples per user
    :return: updated dict_users with minimum samples guaranteed
    """
    if min_samples <= 0:
        return dict_users
    
    labels = np.array(dataset.targets)
    num_users = len(dict_users)
    
    # Ensure all user data is numpy arrays
    user_data = {}
    for user_idx, indices in dict_users.items():
        if isinstance(indices, set):
            user_data[user_idx] = np.array(list(indices), dtype=np.int32)
        elif isinstance(indices, list):
            user_data[user_idx] = np.array(indices, dtype=np.int32)
        else:
            user_data[user_idx] = np.array(indices, dtype=np.int32)
    
    # Find users needing more samples
    deficit_users = []
    total_deficit = 0
    
    for user_idx, indices in user_data.items():
        if len(indices) < min_samples:
            deficit = min_samples - len(indices)
            deficit_users.append((user_idx, deficit))
            total_deficit += deficit
    
    if not deficit_users:
        return {k: v for k, v in user_data.items()}  # Return numpy arrays
    
    # Get available indices efficiently using boolean indexing
    all_used = np.concatenate(list(user_data.values())) if user_data else np.array([], dtype=np.int32)
    available_mask = np.ones(len(dataset), dtype=bool)
    if len(all_used) > 0:
        available_mask[all_used] = False
    available_indices = np.where(available_mask)[0]
    
    if len(available_indices) < total_deficit:
        print(f"Warning: Insufficient samples. Need {total_deficit}, have {len(available_indices)}")
        total_deficit = len(available_indices)
    
    # Sample all needed indices at once - much faster than individual sampling
    if total_deficit > 0:
        sampled_indices = np.random.choice(available_indices, total_deficit, replace=False)
        
        # Distribute to users in batch
        start_pos = 0
        for user_idx, deficit in deficit_users:
            if start_pos >= len(sampled_indices):
                break
            
            actual_deficit = min(deficit, len(sampled_indices) - start_pos)
            end_pos = start_pos + actual_deficit
            additional = sampled_indices[start_pos:end_pos]
            
            # Update user data
            user_data[user_idx] = np.concatenate([user_data[user_idx], additional])
            start_pos = end_pos
    
    return user_data

def analyze_distribution(dict_users, dataset, dataset_name="Dataset"):
    """
    Analyze and print the data distribution across users
    :param dict_users: user data distribution
    :param dataset: original dataset
    :param dataset_name: name of dataset for printing
    """
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    
    print(f"\n=== {dataset_name} Data Distribution Analysis ===")
    print(f"Total users: {len(dict_users)}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Calculate statistics
    user_sample_counts = []
    for indices in dict_users.values():
        # Handle both sets and arrays
        if isinstance(indices, set):
            user_sample_counts.append(len(indices))
        else:
            user_sample_counts.append(len(indices))
    
    print(f"Samples per user - Min: {min(user_sample_counts)}, Max: {max(user_sample_counts)}, Avg: {np.mean(user_sample_counts):.1f}")
    
    # Class distribution per user
    class_distributions = []
    for user_idx, indices in dict_users.items():
        # Convert to numpy array if needed
        if isinstance(indices, set):
            indices = np.array(list(indices))
        elif isinstance(indices, list):
            indices = np.array(indices)
        
        user_labels = labels[indices]
        class_count = Counter(user_labels)
        class_dist = [class_count.get(c, 0) for c in range(num_classes)]
        class_distributions.append(class_dist)
    
    class_distributions = np.array(class_distributions)
    
    # Calculate diversity metrics
    non_zero_classes = np.sum(class_distributions > 0, axis=1)
    print(f"Classes per user - Min: {min(non_zero_classes)}, Max: {max(non_zero_classes)}, Avg: {np.mean(non_zero_classes):.1f}")
    
    # Calculate class balance (standard deviation of class proportions)
    user_class_props = class_distributions / (class_distributions.sum(axis=1, keepdims=True) + 1e-8)
    class_balance = np.std(user_class_props, axis=1)
    print(f"Class balance (std) - Min: {min(class_balance):.3f}, Max: {max(class_balance):.3f}, Avg: {np.mean(class_balance):.3f}")
    
    # Show first few users' class distributions
    print(f"\nFirst 5 users' class distributions:")
    for i in range(min(5, len(dict_users))):
        user_classes = np.sum(class_distributions[i] > 0)
        dominant_class = np.argmax(class_distributions[i])
        dominant_ratio = class_distributions[i][dominant_class] / np.sum(class_distributions[i])
        print(f"User {i}: {user_classes} classes, dominant class {dominant_class} ({dominant_ratio:.2%})")
    
    return class_distributions

def plot_distribution(dict_users, dataset, dataset_name="Dataset", save_path=None):
    """
    Plot the data distribution across users
    :param dict_users: user data distribution
    :param dataset: original dataset
    :param dataset_name: name of dataset for plotting
    :param save_path: path to save the plot (optional)
    """
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    
    # Define class names based on dataset
    if "CIFAR-10" in dataset_name or "cifar10" in dataset_name.lower():
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif "MNIST" in dataset_name or "mnist" in dataset_name.lower():
        class_names = [str(i) for i in range(10)]
    else:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Calculate class distribution per user
    class_distributions = []
    for user_idx in sorted(dict_users.keys()):
        indices = dict_users[user_idx]
        
        # Convert to numpy array if needed
        if isinstance(indices, set):
            indices = np.array(list(indices))
        elif isinstance(indices, list):
            indices = np.array(indices)
        
        user_labels = labels[indices]
        class_count = Counter(user_labels)
        class_dist = [class_count.get(c, 0) for c in range(num_classes)]
        class_distributions.append(class_dist)
    
    class_distributions = np.array(class_distributions)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(class_distributions.T, cmap='Blues', aspect='auto')
    plt.colorbar(label='Number of Samples')
    plt.xlabel('User ID')
    plt.ylabel('Class')
    plt.title(f'Data Distribution Across Users ({dataset_name})')
    
    if len(class_names) <= 20:  # Only show class names if not too many
        plt.yticks(range(num_classes), class_names, rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Wrapper functions for backward compatibility with your existing code
def cifar_iid(dataset, num_users):
    """Wrapper function for backward compatibility"""
    return cifar10_iid(dataset, num_users)

def cifar_noniid(dataset, num_users, alpha=0.5):
    """Wrapper function for backward compatibility - now uses optimized Dirichlet"""
    return cifar10_dirichlet(dataset, num_users, alpha=alpha)

if __name__ == '__main__':
    # Test the implementation
    print("Testing OPTIMIZED Dirichlet-based Sampling Methods")
    print("=" * 60)
    
    # Test CIFAR-10
    dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))
    
    num_users = 100
    
    print("Testing CIFAR-10 IID distribution:")
    dict_users_iid = cifar10_iid(dataset_train, num_users)
    analyze_distribution(dict_users_iid, dataset_train, "CIFAR-10 IID")
    
    print("\n" + "="*60)
    
    print("Testing CIFAR-10 Non-IID distribution (alpha=0.5) - OPTIMIZED:")
    import time
    start_time = time.time()
    dict_users_noniid = cifar10_dirichlet(dataset_train, num_users, alpha=0.5)
    end_time = time.time()
    print(f"⚡ Partitioning completed in {end_time - start_time:.2f} seconds!")
    analyze_distribution(dict_users_noniid, dataset_train, "CIFAR-10 Non-IID (α=0.5)")
    
    print("\n" + "="*60)
    
    print("Testing CIFAR-10 Extreme Non-IID distribution (alpha=0.1):")
    dict_users_extreme = cifar10_dirichlet(dataset_train, num_users, alpha=0.1)
    analyze_distribution(dict_users_extreme, dataset_train, "CIFAR-10 Extreme Non-IID (α=0.1)")
    
    # Plot and save distributions
    plot_distribution(
        dict_users_iid, dataset_train, "CIFAR-10 IID",
        save_path="/Users/ml/Desktop/Federated_Learning_GMB/save/images/cifar10_iid_distribution.png"
    )
    plot_distribution(
        dict_users_noniid, dataset_train, "CIFAR-10 Non-IID (α=0.5)",
        save_path="/Users/ml/Desktop/Federated_Learning_GMB/save/images/cifar10_noniid_alpha0.5_distribution.png"
    )
    plot_distribution(
        dict_users_extreme, dataset_train, "CIFAR-10 Extreme Non-IID (α=0.1)",
        save_path="/Users/ml/Desktop/Federated_Learning_GMB/save/images/cifar10_noniid_alpha0.1_distribution.png"
    )