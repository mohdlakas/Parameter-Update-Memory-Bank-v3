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
                      ComprehensiveAnalyzer, write_fedavg_comprehensive_analysis)


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

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    plot_data_distribution(
    user_groups, train_dataset,
    save_path='../save/images/data_distribution_{}_iid[{}]_alpha[{}].png'.format(
        args.dataset, args.iid, getattr(args, 'alpha', 'NA')
    ),
    title="Client Data Distribution (IID={})".format(args.iid)
    )

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar' or args.dataset == 'cifar100':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

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
        
        # Test accuracy every 5 rounds for detailed tracking
        test_acc_current = None
        if (epoch + 1) % 5 == 0:
            test_acc_current, _ = test_inference(args, global_model, test_dataset)
        
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

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print(f'Selected clients: {len(selected_clients)}')
            print(f'Avg client reliability: {np.mean(list(client_reliabilities.values())):.4f}')

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                args.local_ep, args.local_bs)

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
    plot_filename = (
        '../save/images/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_opt[{}]_lr[{}]_alpha[{}]_{}.png'
        .format(
            args.dataset,
            args.model,
            args.epochs,
            args.frac,
            args.iid,
            args.local_ep,
            args.local_bs,
            getattr(args, 'optimizer', 'NA'),
            getattr(args, 'lr', 'NA'),
            getattr(args, 'alpha', 'NA'),
            timestamp
        )
    )
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

    fedavg_filename = f"../save/logs/fedavg_comprehensive_analysis_{timestamp}.txt"
    write_fedavg_comprehensive_analysis(analyzer, args, test_acc, total_time, fedavg_filename, experiment_seed)

    print(f"\n✅ FedAvg comprehensive analysis saved to: {fedavg_filename}")

    # Print key metrics to console for immediate feedback
    convergence_metrics = analyzer.calculate_convergence_metrics()
    client_analysis = analyzer.analyze_client_selection_quality()
    
    print(f"\n🔍 FEDAVG RESULTS SUMMARY:")
    print(f"   Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Convergence Speed: {convergence_metrics.get('convergence_round', 'N/A')} rounds")
    print(f"   Training Stability: {convergence_metrics.get('training_stability', 0):.6f}")
    print(f"   Unique Clients Selected: {client_analysis.get('total_unique_clients', 0)}")
    print(f"   Avg Participation Rate: {client_analysis.get('avg_participation_rate', 0):.4f}")
    print(f"   Total Runtime: {total_time:.2f} seconds")
    print(f"\n📁 All results saved in ../save/logs/ directory")