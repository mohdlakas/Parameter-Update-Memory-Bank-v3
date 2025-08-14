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

class FedNovaLocalUpdate(LocalUpdate):
    def __init__(self, args, dataset, idxs, logger):
        super().__init__(args, dataset, idxs, logger)
        self.tau = args.tau if hasattr(args, 'tau') and args.tau else args.local_ep
        
    def update_weights_fednova(self, model, global_round):
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
                
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # Calculate effective local steps
        effective_tau = self.tau
        w_new = model.state_dict()
        coeff = effective_tau - self.args.gm * gradient_sqnorm
        
        # Normalize the update
        for key in w_new.keys():
            w_new[key] = w_old[key] + coeff * (w_new[key] - w_old[key])
        
        return w_new, sum(epoch_loss) / len(epoch_loss), effective_tau

def fednova_aggregate(local_weights, taus, gm=1.0):
    total_tau = sum(taus)
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
    args.gm = getattr(args, 'gm', 1.0)
    args.tau = getattr(args, 'tau', None)
    
    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'
    
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
    
    train_loss, train_accuracy = [], []
    
    for epoch in tqdm(range(args.epochs)):
        print(f'Round {epoch+1}/{args.epochs}')
        
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
        
        if epoch % 1 == 0:
            test_acc, _ = test_inference(args, global_model, test_dataset)
            print(f"Round {epoch+1}: Train Accuracy = {train_accuracy[-1]*100:.2f}%, Test Accuracy = {test_acc*100:.2f}%, Loss = {loss_avg:.4f}")
    test_acc, _ = test_inference(args, global_model, test_dataset)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
