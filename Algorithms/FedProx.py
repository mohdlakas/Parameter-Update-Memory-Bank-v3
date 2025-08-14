import os
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
import sys

sys.path.append('../')
from options import args_parser
from models import CNNCifar
from utils_dir import get_dataset, exp_details , check_gpu_pytorch
from update import LocalUpdate, test_inference


class FedProxLocalUpdate(LocalUpdate):
    def __init__(self, args, dataset, idxs, logger, mu=0.01):
        super().__init__(args, dataset, idxs, logger)
        self.mu = mu
        
    def update_weights(self, model, global_round):
        # Set model to train mode
        model.train()
        
        # Initialize device and criterion if not already done
        if self.device is None:
            self.device = next(model.parameters()).device
            self.criterion = self.criterion.to(self.device)
        
        # Initialize optimizer for the current model
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                           momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                            weight_decay=1e-4)
        
        epoch_loss = []
        global_weights = copy.deepcopy(model.state_dict())
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                
                # Add proximal term
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += (param - global_weights[name]).norm(2) ** 2
                loss += (self.mu / 2) * proximal_term
                
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    args.mu = getattr(args, 'mu', 0.01)
    

    device = check_gpu_pytorch()

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
    else:
        exit('Error: only CNN model is supported')
    
    global_model.to(device)
    global_model.train()
    
    train_loss, train_accuracy = [], []
    
    for epoch in tqdm(range(args.epochs)):
        print(f'Round {epoch+1}/{args.epochs}')
        
        local_weights, local_losses = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            local_model = FedProxLocalUpdate(args=args, dataset=train_dataset,
                                           idxs=user_groups[idx], logger=None, mu=args.mu)
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
        
        # Print progress every round
        if epoch % 1 == 0:
            test_acc, _ = test_inference(args, global_model, test_dataset)
            print(f"Round {epoch+1}: Train Accuracy = {train_accuracy[-1]*100:.2f}%, Test Accuracy = {test_acc*100:.2f}%, Loss = {loss_avg:.4f}")
    test_acc, _ = test_inference(args, global_model, test_dataset)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")