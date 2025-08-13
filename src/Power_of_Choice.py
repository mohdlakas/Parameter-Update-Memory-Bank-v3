import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNCifar
from utils_dir import (get_dataset, exp_details, plot_data_distribution,
                      ComprehensiveAnalyzer, write_comprehensive_analysis)
from datetime import datetime

def power_of_choice_selection(user_groups, num_users, fraction, d=10):
    """
    Power-of-Choice client selection
    d: number of candidates to sample for each selection
    """
    m = max(int(fraction * num_users), 1)
    selected_clients = []
    
    for _ in range(m):
        # Sample d candidates
        candidates = np.random.choice(range(num_users), min(d, num_users), replace=False)
        
        # Select client with largest dataset
        best_client = max(candidates, key=lambda x: len(user_groups[x]))
        selected_clients.append(best_client)
        
        # Remove selected client from future selections this round
        num_users_remaining = num_users
    
    return list(set(selected_clients))  # Remove duplicates

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    
    # Add Power-of-Choice specific parameter
    args.d = getattr(args, 'd', 10)  # Number of candidates
    
    exp_details(args)
    
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'
    
    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Build model
    if args.model == 'cnn':
        if args.dataset == 'cifar100':
            args.num_classes = 100
        elif args.dataset == 'cifar' or args.dataset == 'cifar10':
            args.num_classes = 10
        else:
            exit(f'Error: unsupported dataset {args.dataset}')
            
        global_model = CNNCifar(args)
        print(f"Power-of-Choice Model created with {global_model.fc2.out_features} output classes")
    else:
        exit('Error: only CNN model is supported')
    
    global_model.to(device)
    global_model.train()
    
    analyzer = ComprehensiveAnalyzer()
    train_loss, train_accuracy = [], []
    print_every = 2
    
    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()
        print(f'\n | Power-of-Choice Global Training Round : {epoch+1} |\n')
        
        local_weights, local_losses = [], []
        
        # Power-of-Choice client selection
        selected_clients = power_of_choice_selection(user_groups, args.num_users, args.frac, args.d)
        print(f"Selected {len(selected_clients)} clients using Power-of-Choice")
        
        for idx in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=None)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # FedAvg aggregation
        global_weights = {}
        for key in local_weights[0].keys():
            global_weights[key] = torch.stack([local_weights[i][key] for i in range(len(local_weights))], 0).mean(0)
        
        global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Calculate training accuracy
        list_acc = []
        for c in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[c], logger=None)
            correct, total = 0, 0
            criterion = torch.nn.NLLLoss().to(device)
            
            with torch.no_grad():
                for images, labels in local_model.trainloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = global_model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total if total > 0 else 0
            list_acc.append(acc)
        
        train_accuracy.append(np.mean(list_acc))
        
        # Track for analysis
        round_time = time.time() - round_start_time
        test_acc_current = None
        if epoch % print_every == 0 or epoch == args.epochs - 1:
            test_acc_current, _ = test_inference(args, global_model, test_dataset)
            print(f"Power-of-Choice Round {epoch}: Test Accuracy = {test_acc_current:.4f}")
        
        # Log data to analyzer
        analyzer.log_round_data(
            round_num=epoch,
            train_acc=train_accuracy[-1],
            train_loss=train_loss[-1],
            test_acc=test_acc_current,
            selected_clients=selected_clients,
            aggregation_weights={i: 1.0/len(selected_clients) for i in selected_clients},
            client_reliabilities={i: len(user_groups[i]) for i in selected_clients},
            client_qualities={i: 1.0 for i in selected_clients},
            memory_bank_size=None,
            avg_similarity=None,
            round_time=round_time
        )
    
    # Final evaluation
    test_acc, _ = test_inference(args, global_model, test_dataset)
    total_time = time.time() - start_time
    
    print(f' \n Power-of-Choice Results after {args.epochs} global rounds:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Final Test Accuracy: {:.2f}%".format(100*test_acc))
    
    # Save and analyze results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comprehensive_filename = f"../save/logs/power_of_choice_comprehensive_analysis_{timestamp}.txt"
    write_comprehensive_analysis(analyzer, args, test_acc, total_time, comprehensive_filename, getattr(args, 'seed', None))
    
    print(f"\n✅ Power-of-Choice analysis saved to: {comprehensive_filename}")
    print(f"🔍 Power-of-Choice Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")