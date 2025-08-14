#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random
import torch

from tensorboardX import SummaryWriter
from datetime import datetime

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils_dir import (get_dataset, exp_details, average_weights, plot_data_distribution,
                      ComprehensiveAnalyzer, write_fedavg_comprehensive_analysis, check_gpu_pytorch)


import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
   

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #if args.gpu_id is not None:
    #print(f"Using GPU {args.gpu_id}")

    # FIX: Create save directories based on current working directory
    # Check if we're in src/Algorithms (when run from auto_compare.py)
    current_dir = os.getcwd()
    if 'Algorithms' in current_dir or 'algorithms' in current_dir:
        save_base = '../../save'  # From src/Algorithms to project root
    else:
        save_base = '../save'     # From src to project root
    
    # Create all necessary directories
    os.makedirs(f'{save_base}/objects', exist_ok=True)
    os.makedirs(f'{save_base}/images', exist_ok=True)
    os.makedirs(f'{save_base}/logs', exist_ok=True)

    # Check and set device
    device = check_gpu_pytorch()
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    plot_data_distribution(
        user_groups, train_dataset,
        save_path=f'{save_base}/images/data_distribution_{args.dataset}_iid[{args.iid}]_alpha[{getattr(args, "alpha", "NA")}].png',
        title="Client Data Distribution (IID={})".format(args.iid)
    )

# Build model
    if args.model == 'cnn':
        # Force correct number of classes before model creation
        if args.dataset == 'cifar100':
            args.num_classes = 100
            print(f"FORCED: Setting num_classes to {args.num_classes} for CIFAR-100")
        elif args.dataset == 'cifar' or args.dataset == 'cifar10':
            args.num_classes = 10
            print(f"FORCED: Setting num_classes to {args.num_classes} for CIFAR-10")
        else:
            exit(f'Error: unsupported dataset {args.dataset}. Only cifar10 and cifar100 are supported.')
            
        # Create model with correct output dimensions
        global_model = CNNCifar(args)
        print(f"Model created with {global_model.fc2.out_features} output classes")
        
    else:
        exit('Error: only CNN model is supported. Use --model=cnn')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Initialize comprehensive analyzer for detailed metrics
    analyzer = ComprehensiveAnalyzer()
    
    # Set random seed for reproducibility if provided
    experiment_seed = getattr(args, 'seed', None)
    if experiment_seed is not None:
        torch.manual_seed(experiment_seed)
        np.random.seed(experiment_seed)
        random.seed(experiment_seed)

    # Training
    train_loss, train_accuracy = [], []    
    test_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()  # Track round time
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        selected_clients = list(idxs_users)  # For analysis tracking

        # Track client data for analysis
        client_losses = {}  # Store (before, after) loss tuples for quality analysis
        data_sizes = {}     # Store client data sizes
        
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            
            # Evaluate loss before local training for quality analysis
            loss_before = local_model.inference(model=global_model)[1]
            
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
            # Evaluate loss after local training
            temp_model = copy.deepcopy(global_model)
            temp_model.load_state_dict(w)
            loss_after = local_model.inference(model=temp_model)[1]
            
            # Store for analysis
            client_losses[idx] = (loss_before, loss_after)
            data_sizes[idx] = len(user_groups[idx])

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over SELECTED users only (like PUMB does)
        list_acc, list_loss = [], []

        global_model.eval()
        for c in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[c], logger=logger)
            
            # FIXED: Evaluate on training data instead of test data
            correct, total = 0, 0
            loss_sum = 0
            criterion = torch.nn.NLLLoss().to(device)
            
            with torch.no_grad():
                for images, labels in local_model.trainloader:  # ← Use trainloader!
                    images, labels = images.to(device), labels.to(device)
                    outputs = global_model(images)
                    loss_sum += criterion(outputs, labels).item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total if total > 0 else 0
            loss = loss_sum / len(local_model.trainloader) if len(local_model.trainloader) > 0 else 0
            
            list_acc.append(acc)
            list_loss.append(loss)

        train_accuracy.append(sum(list_acc)/len(list_acc))
        # FIX: Calculate test accuracy for printing
        test_acc_current, _ = test_inference(args, global_model, test_dataset)
        print(f"Round {epoch+1}: Train Accuracy = {train_accuracy[-1]*100:.2f}%, Test Accuracy = {test_acc_current*100:.2f}%, Loss = {loss_avg:.4f}")
        # Collect data for comprehensive analysis
        round_time = time.time() - round_start_time
        
        # For FedAvg, we'll track basic client selection patterns
        # Calculate simple aggregation weights (uniform for FedAvg)
        aggregation_weights = {client_id: 1.0/len(selected_clients) for client_id in selected_clients}
        
        # For FedAvg, client reliability is based on data size (no learning)
        client_reliabilities = {}
        for client_id in selected_clients:
            # Simple reliability based on data size and loss improvement
            loss_before, loss_after = client_losses[client_id]
            loss_improvement = max(0, loss_before - loss_after)
            # Normalize by data size for basic reliability metric
            reliability = (loss_improvement + 1e-6) * data_sizes[client_id] / 1000.0
            client_reliabilities[client_id] = min(1.0, reliability)  # Cap at 1.0
        
  

        # # Test accuracy every 5 rounds for detailed tracking
        # test_acc_current = None
        # if (epoch + 1) % 5 == 0:
        #     test_acc_current, _ = test_inference(args, global_model, test_dataset)
        
        # Log all data to analyzer (adapted for FedAvg)
        analyzer.log_round_data(
            round_num=epoch + 1,
            train_acc=train_accuracy[-1],
            train_loss=loss_avg,
            test_acc=test_acc_current,
            selected_clients=selected_clients,
            aggregation_weights=aggregation_weights,
            client_reliabilities=client_reliabilities,
            client_qualities=None,  # FedAvg doesn't have quality computation
            memory_bank_size=None,  # FedAvg doesn't have memory bank
            avg_similarity=None,    # FedAvg doesn't track similarities
            round_time=round_time
        )

        # # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        #     print(f'Selected clients: {len(selected_clients)}')
        #     print(f'Avg client reliability: {np.mean(list(client_reliabilities.values())):.4f}')

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = f'{save_base}/objects/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


    # Plot Loss and Accuracy in one chart
    plt.figure()
    plt.title(f'Test Accuracy: {test_acc*100:.2f}%\nTraining Loss and Accuracy vs Communication Rounds')
    plt.xlabel('Communication Rounds')

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(range(len(train_loss)), train_loss, color='r', label='Training Loss')
    ax2.plot(range(len(train_accuracy)), train_accuracy, color='k', label='Training Accuracy')

    ax1.set_ylabel('Training Loss', color='r')
    ax2.set_ylabel('Training Accuracy', color='k')

    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='k')


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'{save_base}/images/fed_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_opt[{getattr(args, "optimizer", "NA")}]_lr[{getattr(args, "lr", "NA")}]_alpha[{getattr(args, "alpha", "NA")}]_{timestamp}.png'
 
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Generate comprehensive analysis report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Write basic experiment summary (original)
    # exp_details_to_file(args, f"../save/logs/fedavg_experiment_summary_{timestamp}.txt")
    # print(f"📊 Basic experiment summary saved to: fedavg_experiment_summary_{timestamp}.txt")
    # Generate FedAvg-specific comprehensive analysis report

    fedavg_filename = f"{save_base}/logs/fedavg_comprehensive_analysis_{timestamp}.txt"
    write_fedavg_comprehensive_analysis(analyzer, args, test_acc, total_time, fedavg_filename, experiment_seed)

    print(f"\n✅ FedAvg comprehensive analysis saved to: {fedavg_filename}")

    # Print key metrics to console for immediate feedback
    convergence_metrics = analyzer.calculate_convergence_metrics()
    client_analysis = analyzer.analyze_client_selection_quality()
    
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    
    print(f"\n🔍 FEDAVG RESULTS SUMMARY:")
    print(f"   Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Convergence Speed: {convergence_metrics.get('convergence_round', 'N/A')} rounds")
    print(f"   Training Stability: {convergence_metrics.get('training_stability', 0):.6f}")
    print(f"   Unique Clients Selected: {client_analysis.get('total_unique_clients', 0)}")
    print(f"   Avg Participation Rate: {client_analysis.get('avg_participation_rate', 0):.4f}")
    print(f"   Total Runtime: {total_time:.2f} seconds")
    print(f"\n📁 All results saved in ../save/logs/ directory")
