import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import os

def run_experiment(script_name, method_name, args_dict):
    """Run a single experiment and return test accuracy + training curves"""
    print(f"\n{'='*50}")
    print(f"Running {method_name}...")
    print(f"{'='*50}")
    
    # Build command
    cmd = ['python', script_name]
    for key, value in args_dict.items():
        cmd.extend([f'--{key}', str(value)])
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            # Extract training curves AND final accuracy
            train_accuracies, train_losses, test_accuracies = parse_training_curves(result.stdout, method_name)
            
            # Extract final test accuracy
            final_test_acc = None
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Final Test Accuracy" in line:
                    acc_str = line.split(':')[1].strip().replace('%', '').replace(')', '')
                    final_test_acc = float(acc_str) / 100.0
                    break
            
            print(f"✅ {method_name} completed: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
            return {
                'final_accuracy': final_test_acc,
                'runtime': runtime,
                'train_accuracies': train_accuracies,
                'train_losses': train_losses,
                'test_accuracies': test_accuracies
            }
        else:
            print(f"❌ {method_name} failed:")
            print(result.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"⏰ {method_name} timed out")
        return None
    except Exception as e:
        print(f"❌ {method_name} error: {e}")
        return None

def parse_training_curves(output_text, method_name):
    """Extract training curves from console output"""
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    
    lines = output_text.split('\n')
    
    for line in lines:
        # Look for different output patterns from different algorithms
        
        # Pattern 1: "Round X: Train Acc = Y%, Test Acc = Z%"
        round_match = re.search(r'Round\s+(\d+).*?Train.*?Acc.*?(\d+\.?\d*)%.*?Test.*?Acc.*?(\d+\.?\d*)%', line, re.IGNORECASE)
        if round_match:
            train_acc = float(round_match.group(2)) / 100.0
            test_acc = float(round_match.group(3)) / 100.0
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            continue
            
        # Pattern 2: "Training accuracy: X%, Test accuracy: Y%"
        acc_match = re.search(r'Training accuracy:\s*(\d+\.?\d*)%.*?Test accuracy:\s*(\d+\.?\d*)%', line, re.IGNORECASE)
        if acc_match:
            train_acc = float(acc_match.group(1)) / 100.0
            test_acc = float(acc_match.group(2)) / 100.0
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            continue
            
        # Pattern 3: Loss values
        loss_match = re.search(r'(?:Loss|loss).*?(\d+\.?\d+)', line)
        if loss_match and 'Round' in line:
            loss_val = float(loss_match.group(1))
            train_losses.append(loss_val)
            continue
    
    print(f"📊 {method_name}: Extracted {len(train_accuracies)} train accuracies, "
          f"{len(test_accuracies)} test accuracies, {len(train_losses)} losses")
    
    return train_accuracies, train_losses, test_accuracies

def create_comparison_plots(all_results, timestamp):
    """Create comprehensive comparison plots"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    # 1. Final Accuracy Bar Chart
    methods = []
    final_accs = []
    for method, data in all_results.items():
        if data and data['final_accuracy']:
            methods.append(method)
            final_accs.append(data['final_accuracy'] * 100)
    
    bars = ax1.bar(methods, final_accs, color=colors[:len(methods)])
    ax1.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_ylim([0, max(final_accs) * 1.1])
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Training Accuracy Curves
    max_rounds = 0
    for i, (method, data) in enumerate(all_results.items()):
        if data and data['train_accuracies']:
            rounds = range(1, len(data['train_accuracies']) + 1)
            ax2.plot(rounds, [acc * 100 for acc in data['train_accuracies']], 
                    label=method, color=colors[i], linewidth=2, marker='o', markersize=3)
            max_rounds = max(max_rounds, len(data['train_accuracies']))
    
    ax2.set_title('Training Accuracy Over Rounds', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Training Accuracy (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Test Accuracy Curves
    for i, (method, data) in enumerate(all_results.items()):
        if data and data['test_accuracies']:
            rounds = range(1, len(data['test_accuracies']) + 1)
            ax3.plot(rounds, [acc * 100 for acc in data['test_accuracies']], 
                    label=method, color=colors[i], linewidth=2, marker='s', markersize=3)
    
    ax3.set_title('Test Accuracy Over Rounds', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Loss Curves
    for i, (method, data) in enumerate(all_results.items()):
        if data and data['train_losses']:
            rounds = range(1, len(data['train_losses']) + 1)
            ax4.plot(rounds, data['train_losses'], 
                    label=method, color=colors[i], linewidth=2, marker='^', markersize=3)
    
    ax4.set_title('Training Loss Over Rounds', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Training Loss')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')  # Log scale for better loss visualization
    
    plt.tight_layout()
    
    # Save plots
    os.makedirs('../save/images', exist_ok=True)
    plot_path = f'../save/images/comprehensive_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_detailed_results(all_results, timestamp):
    """Save detailed results including training curves"""
    
    # Create comprehensive CSV with training curves
    detailed_data = []
    
    for method, data in all_results.items():
        if data:
            # Add final results
            detailed_data.append({
                'Method': method,
                'Final_Test_Accuracy': data['final_accuracy'],
                'Runtime_Seconds': data['runtime'],
                'Max_Train_Accuracy': max(data['train_accuracies']) if data['train_accuracies'] else None,
                'Min_Train_Loss': min(data['train_losses']) if data['train_losses'] else None,
                'Max_Test_Accuracy': max(data['test_accuracies']) if data['test_accuracies'] else None,
                'Convergence_Round': len(data['test_accuracies']) if data['test_accuracies'] else None
            })
    
    df_summary = pd.DataFrame(detailed_data)
    
    # Save summary CSV
    os.makedirs('../save/logs', exist_ok=True)
    summary_path = f'../save/logs/detailed_comparison_{timestamp}.csv'
    df_summary.to_csv(summary_path, index=False)
    
    # Save individual training curves
    curves_path = f'../save/logs/training_curves_{timestamp}.csv'
    
    curves_data = []
    max_rounds = max([len(data['train_accuracies']) if data and data['train_accuracies'] else 0 
                     for data in all_results.values()])
    
    for round_num in range(1, max_rounds + 1):
        row = {'Round': round_num}
        for method, data in all_results.items():
            if data:
                # Training accuracy
                if data['train_accuracies'] and round_num <= len(data['train_accuracies']):
                    row[f'{method}_Train_Acc'] = data['train_accuracies'][round_num-1]
                
                # Test accuracy
                if data['test_accuracies'] and round_num <= len(data['test_accuracies']):
                    row[f'{method}_Test_Acc'] = data['test_accuracies'][round_num-1]
                
                # Training loss
                if data['train_losses'] and round_num <= len(data['train_losses']):
                    row[f'{method}_Train_Loss'] = data['train_losses'][round_num-1]
        
        curves_data.append(row)
    
    df_curves = pd.DataFrame(curves_data)
    df_curves.to_csv(curves_path, index=False)
    
    return summary_path, curves_path

def main():
    # Experiment configuration
    base_args = {
        'dataset': 'cifar10',
        'model': 'cnn',
        'epochs': 50,  # Reduced for faster testing
        'num_users': 100,
        'frac': 0.1,
        'local_ep': 5,
        'local_bs': 10,
        'lr': 0.01,
        'iid': 0,
        'alpha': 0.5,
        'seed': 42
    }
    
    # Define experiments
    experiments = [
        ('../federated_main.py', 'FedAvg', base_args),
        ('federated_fedprox_main.py', 'FedProx', {**base_args, 'mu': 0.01}),
        ('federated_scaffold_main.py', 'SCAFFOLD', base_args),
        ('federated_power_of_choice_main.py', 'Power-of-Choice', {**base_args, 'd': 10}),
        ('federated_fednova_main.py', 'FedNova', {**base_args, 'gm': 1.0, 'tau': 5}),
        ('../federated_pumb_main.py', 'PUMB', {**base_args, 'pumb_exploration_ratio': 0.4, 'pumb_initial_rounds': 10}),
    ]
    
    # Run experiments
    all_results = {}
    
    for script, method, args in experiments:
        result = run_experiment(script, method, args)
        all_results[method] = result
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate comprehensive plots
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE COMPARISON PLOTS...")
    print(f"{'='*60}")
    
    plot_path = create_comparison_plots(all_results, timestamp)
    summary_path, curves_path = save_detailed_results(all_results, timestamp)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ENHANCED COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Sort by final accuracy
    sorted_results = sorted([(k, v) for k, v in all_results.items() if v and v['final_accuracy']], 
                           key=lambda x: x[1]['final_accuracy'], reverse=True)
    
    print(f"{'Method':<20} {'Final Acc':<12} {'Runtime':<12} {'Curves':<10}")
    print("-" * 60)
    for method, data in sorted_results:
        curves_info = f"{len(data['test_accuracies'])} rounds" if data['test_accuracies'] else "No data"
        print(f"{method:<20} {data['final_accuracy']*100:.2f}%{'':<7} {data['runtime']:.1f}s{'':<6} {curves_info}")
    
    print(f"\n✅ Enhanced results saved to:")
    print(f"   📊 Comprehensive plots: {plot_path}")
    print(f"   📋 Summary CSV: {summary_path}")
    print(f"   📈 Training curves CSV: {curves_path}")

if __name__ == '__main__':
    main()