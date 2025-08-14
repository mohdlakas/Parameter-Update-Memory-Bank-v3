import os
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
import sys

sys.path.append('../')
from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNCifar
from utils_dir import get_dataset, exp_details, check_gpu_pytorch


class SCAFFOLDLocalUpdate(LocalUpdate):
    def __init__(self, args, dataset, idxs, logger):
        super().__init__(args, dataset, idxs, logger)
        
    def update_weights_scaffold(self, model, global_round, c_global, c_local):
        # Set model to train mode
        model.train()
        
        # Initialize device and criterion if not already done
        if self.device is None:
            self.device = next(model.parameters()).device
            self.criterion = self.criterion.to(self.device)
        
        # Initialize optimizer for the current model
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                      momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                       weight_decay=1e-4)
        
        epoch_loss = []
        w_old = copy.deepcopy(model.state_dict())
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                
                # SCAFFOLD correction with safety checks
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        correction = c_global[name] - c_local[name]
                        # Add safety check for NaN/inf
                        if torch.isnan(correction).any() or torch.isinf(correction).any():
                            print(f"WARNING: Invalid correction for {name}, skipping")
                            continue
                        param.grad.data += correction
                
                optimizer.step()
                
                # Safety check for loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: Invalid loss {loss.item()}, using 2.0")
                    batch_loss.append(2.0)
                else:
                    batch_loss.append(loss.item())
                
            if batch_loss:  # Ensure we have valid losses
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            else:
                epoch_loss.append(2.0)  # Fallback loss
        
        # Calculate delta for control variate update
        w_new = model.state_dict()
        c_delta = {}
        for name in w_new.keys():
            c_delta[name] = (w_old[name] - w_new[name]) / (self.args.local_ep * self.args.lr) - c_global[name]
        
        # Safety check for final loss calculation
        if epoch_loss and all(not (np.isnan(l) or np.isinf(l)) for l in epoch_loss):
            final_loss = sum(epoch_loss) / len(epoch_loss)
        else:
            print("WARNING: All epoch losses invalid, using fallback")
            final_loss = 2.0
            
        return w_new, final_loss, c_delta


def initialize_control_variates(model):
    c = {}
    for name, param in model.named_parameters():
        c[name] = torch.zeros_like(param.data)
    return c


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    
    device = check_gpu_pytorch()

    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    if args.model == 'cnn':
        if args.dataset == 'cifar100':
            args.num_classes = 100
        elif args.dataset == 'cifar' or args.dataset == 'cifar10':
            args.num_classes = 10
        else:
            exit(f'Error: unsupported dataset {args.dataset}')
            
        global_model = CNNCifar(args)
    else:
        exit('Error: only CNN model is supported')
    
    global_model.to(device)
    global_model.train()
    
    # Initialize control variates
    c_global = initialize_control_variates(global_model)
    c_local = {i: initialize_control_variates(global_model) for i in range(args.num_users)}
    
    train_loss, train_accuracy = [], []
    
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        c_deltas = []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            local_model = SCAFFOLDLocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=None)
            
            w, loss, c_delta = local_model.update_weights_scaffold(
                model=copy.deepcopy(global_model), 
                global_round=epoch,
                c_global=c_global,
                c_local=c_local[idx]
            )
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            c_deltas.append(c_delta)
            
            # Update local control variate
            for name in c_local[idx].keys():
                c_local[idx][name] += c_delta[name]
        
        # Aggregate weights
        global_weights = {}
        for key in local_weights[0].keys():
            global_weights[key] = torch.stack([local_weights[i][key] for i in range(len(local_weights))], 0).mean(0)
        
        # Update global control variate
        for name in c_global.keys():
            c_global[name] += (1.0 / args.num_users) * sum([c_delta[name] for c_delta in c_deltas])
        
        global_model.load_state_dict(global_weights)
        
        # MOVED: These lines now belong in the main training loop
        loss_avg = sum(local_losses) / len(local_losses)
        
        # Add safety check
        if np.isnan(loss_avg) or np.isinf(loss_avg):
            print(f"WARNING: Invalid loss_avg = {loss_avg}, using fallback")
            loss_avg = 2.0
        
        train_loss.append(loss_avg)
        
        # Calculate training accuracy
        list_acc = []
        for c in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[c], logger=None)
            correct, total = 0, 0
            
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
        
        # Print every round like other algorithms
        test_acc, _ = test_inference(args, global_model, test_dataset)
        print(f"Round {epoch+1}: Train Accuracy = {train_accuracy[-1]*100:.2f}%, Test Accuracy = {test_acc*100:.2f}%, Loss = {loss_avg:.4f}")
    
    # Final test accuracy
    test_acc, _ = test_inference(args, global_model, test_dataset)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")