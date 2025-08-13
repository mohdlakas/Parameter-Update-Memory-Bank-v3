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

class FedProxLocalUpdate(LocalUpdate):
    def __init__(self, args, dataset, idxs, logger, mu=0.01):
        super().__init__(args, dataset, idxs, logger)
        self.mu = mu  # Proximal term coefficient
        
    def update_weights(self, model, global_round):
        model.train()
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
    
    # Add FedProx specific parameter
    args.mu = getattr(args, 'mu', 0.01)  # Proximal term coefficient
    
    exp_details(args)
    
    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'
    
    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Plot client data distribution
    plot_data_distribution(
        user_groups, train_dataset,
        save_path='../save/images/data_distribution_fedprox_{}_iid[{}]_alpha[{}].png'.format(
            args.dataset, args.iid, getattr(args, 'alpha', 'NA')
        ),
        title="FedProx Client Data Distribution (IID={})".format(args.iid)
    )
    
    # Build model
    if args.model == 'cnn':
        if args.dataset == 'cifar100':
            args.num_classes = 100
        elif args.dataset == 'cifar' or args.dataset == 'cifar10':
            args.num_classes = 10
        else:
            exit(f'Error: unsupported dataset {args.dataset}')
            
        global_model = CNNCifar(args)
        print(f"FedProx Model created with {global_model.fc2.out_features} output classes")
    else:
        exit('Error: only CNN model is supported')
    
    global_model.to(device)
    global_model.train()
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer()
    
    train_loss, train_accuracy = [], []
    print_every = 2
    
    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()
        print(f'\n | FedProx Global Training Round : {epoch+1} |\n')
        
        local_weights, local_losses = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            local_model = FedProxLocalUpdate(args=args, dataset=train_dataset,
                                           idxs=user_groups[idx], logger=None, mu=args.mu)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # FedAvg aggregation (same as original)
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
            print(f"FedProx Round {epoch}: Test Accuracy = {test_acc_current:.4f}")
        
        # Log data to analyzer
        analyzer.log_round_data(
            round_num=epoch,
            train_acc=train_accuracy[-1],
            train_loss=train_loss[-1],
            test_acc=test_acc_current,
            selected_clients=list(idxs_users),
            aggregation_weights={i: 1.0/len(idxs_users) for i in idxs_users},
            client_reliabilities={i: 1.0 for i in idxs_users},
            client_qualities={i: 1.0 for i in idxs_users},
            memory_bank_size=None,
            avg_similarity=None,
            round_time=round_time
        )
    
    # Final evaluation
    test_acc, _ = test_inference(args, global_model, test_dataset)
    total_time = time.time() - start_time
    
    print(f' \n FedProx Results after {args.epochs} global rounds:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Final Test Accuracy: {:.2f}%".format(100*test_acc))
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'../save/objects/FedProx_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)
    
    # Generate comprehensive analysis
    comprehensive_filename = f"../save/logs/fedprox_comprehensive_analysis_{timestamp}.txt"
    write_comprehensive_analysis(analyzer, args, test_acc, total_time, comprehensive_filename, getattr(args, 'seed', None))
    
    print(f"\n✅ FedProx analysis saved to: {comprehensive_filename}")
    print(f"🔍 FedProx Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")