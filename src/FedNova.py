# filepath: c:\Users\Mohamed\Desktop\UpdatedLogic\src\Algorithms\federated_fednova_main.py
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

class FedNovaLocalUpdate(LocalUpdate):
    def __init__(self, args, dataset, idxs, logger):
        super().__init__(args, dataset, idxs, logger)
        self.tau = args.tau if args.tau else args.local_ep
        
    def update_weights_fednova(self, model, global_round):
        model.train()
        epoch_loss = []
        
        # Store initial weights
        w_old = copy.deepcopy(model.state_dict())
        
        # Track gradient norm for normalization
        gradient_sqnorm = 0.0
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                
                # Accumulate gradient squared norm
                for param in model.parameters():
                    if param.grad is not None:
                        gradient_sqnorm += param.grad.data.norm(2) ** 2
                
                self.optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # Calculate effective local steps (tau)
        effective_tau = self.tau
        
        # Calculate covariance shift
        w_new = model.state_dict()
        coeff = effective_tau - self.args.gm * gradient_sqnorm
        
        # Normalize the update
        for key in w_new.keys():
            w_new[key] = w_old[key] + coeff * (w_new[key] - w_old[key])
        
        return w_new, sum(epoch_loss) / len(epoch_loss), effective_tau

def fednova_aggregate(local_weights, taus, gm=1.0):
    """FedNova aggregation with normalized averaging"""
    # Calculate total tau
    total_tau = sum(taus)
    
    # Weighted average based on effective local steps
    global_weights = {}
    for key in local_weights[0].keys():
        weighted_sum = torch.zeros_like(local_weights[0][key])
        for i, weights in enumerate(local_weights):
            weight = taus[i] / total_tau
            weighted_sum += weight * weights[key]
        global_weights[key] = weighted_sum
    
    return global_weights

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    
    # Set FedNova defaults
    args.gm = getattr(args, 'gm', 1.0)
    args.tau = getattr(args, 'tau', None)
    
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
        print(f"FedNova Model created with {global_model.fc2.out_features} output classes")
    else:
        exit('Error: only CNN model is supported')
    
    global_model.to(device)
    global_model.train()
    
    analyzer = ComprehensiveAnalyzer()
    train_loss, train_accuracy = [], []
    print_every = 2
    
    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()
        print(f'\n | FedNova Global Training Round : {epoch+1} |\n')
        
        local_weights, local_losses = [], []
        taus = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            local_model = FedNovaLocalUpdate(args=args, dataset=train_dataset,
                                           idxs=user_groups[idx], logger=None)
            
            w, loss, tau = local_model.update_weights_fednova(
                model=copy.deepcopy(global_model), 
                global_round=epoch
            )
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            taus.append(tau)
        
        # FedNova aggregation
        global_weights = fednova_aggregate(local_weights, taus, args.gm)
        global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Calculate training accuracy
        list_acc = []
        for c in idxs_users:
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
            print(f"FedNova Round {epoch}: Test Accuracy = {test_acc_current:.4f}")
        
        # Log data to analyzer
        analyzer.log_round_data(
            round_num=epoch,
            train_acc=train_accuracy[-1],
            train_loss=train_loss[-1],
            test_acc=test_acc_current,
            selected_clients=list(idxs_users),
            aggregation_weights={i: taus[j]/sum(taus) for j, i in enumerate(idxs_users)},
            client_reliabilities={i: 1.0 for i in idxs_users},
            client_qualities={i: 1.0 for i in idxs_users},
            memory_bank_size=None,
            avg_similarity=None,
            round_time=round_time
        )
    
    # Final evaluation
    test_acc, _ = test_inference(args, global_model, test_dataset)
    total_time = time.time() - start_time
    
    print(f' \n FedNova Results after {args.epochs} global rounds:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Final Test Accuracy: {:.2f}%".format(100*test_acc))
    
    # Save and analyze results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comprehensive_filename = f"../save/logs/fednova_comprehensive_analysis_{timestamp}.txt"
    write_comprehensive_analysis(analyzer, args, test_acc, total_time, comprehensive_filename, getattr(args, 'seed', None))
    
    print(f"\n✅ FedNova analysis saved to: {comprehensive_filename}")
    print(f"🔍 FedNova Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")