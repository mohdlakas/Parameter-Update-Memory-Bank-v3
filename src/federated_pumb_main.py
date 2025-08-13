import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
import logging
matplotlib.use('Agg')

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar

from federated_PUMB import PUMBFederatedServer
from utils_dir import (get_dataset, exp_details, plot_data_distribution,
                      ComprehensiveAnalyzer, write_comprehensive_analysis)

from datetime import datetime

# Create directories for saving results
os.makedirs('../save/objects', exist_ok=True)
os.makedirs('../save/images', exist_ok=True)
os.makedirs('../save/logs', exist_ok=True)

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    exp_details(args)

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Plot client data distribution (IID/non-IID)
    plot_data_distribution(
        user_groups, train_dataset,
        save_path='../save/images/data_distribution_{}_iid[{}]_alpha[{}].png'.format(
            args.dataset, args.iid, getattr(args, 'alpha', 'NA')
        ),
        title="Client Data Distribution (IID={})".format(args.iid)
    )
    
    # Build model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar' or args.dataset == 'cifar100':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()

    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize PUMB server
    server = PUMBFederatedServer(global_model, optimizer, loss_fn, args, embedding_dim=512)

    # Initialize comprehensive analyzer for detailed metrics
    analyzer = ComprehensiveAnalyzer()
    
    # FIX: Initialize tracking lists for analyzer
    analyzer.similarity_scores = []
    analyzer.client_ranking_history = []
    analyzer.quality_scores_history = []
    
    # Set random seed for reproducibility if provided
    experiment_seed = getattr(args, 'seed', None)
    if experiment_seed is not None:
        torch.manual_seed(experiment_seed)
        np.random.seed(experiment_seed)

    train_loss, train_accuracy = [], []
    test_accuracy_history = []
    print_every = 2

    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()  # Track round time
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # Store previous model state BEFORE updates (important!)
        if epoch > 0:
            server.prev_model_state = server._get_model_state_copy()

        # Intelligent Client Selection
        m = max(int(args.frac * args.num_users), 1)
        available_clients = list(range(args.num_users))
        selected_clients = server.select_clients(available_clients, m)

        client_models = {}
        client_losses = {}
        data_sizes = {}
        param_updates = {}
        all_loss_improvements = []

        # Collect all client data
        for idx in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=None)
            
            initial_state = copy.deepcopy(global_model.state_dict())
            loss_before = local_model.inference(model=global_model)[1]
            
            updated_weights, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            temp_model = copy.deepcopy(global_model)
            temp_model.load_state_dict(updated_weights)
            loss_after = local_model.inference(model=temp_model)[1]

            param_update = {name: updated_weights[name] - initial_state[name]
                            for name in updated_weights}

            client_models[idx] = updated_weights
            client_losses[idx] = (loss_before, loss_after)
            data_sizes[idx] = len(user_groups[idx])
            param_updates[idx] = param_update

            loss_improvement = max(0, loss_before - loss_after)
            all_loss_improvements.append(loss_improvement)

        # FIX: Track quality scores for analysis
        client_qualities = {}
        
        # Update memory bank with all client data
        for idx in selected_clients:
            loss_before, loss_after = client_losses[idx]
            
            quality = server.quality_calc.calculate_quality(
                loss_before, loss_after, data_sizes, param_updates[idx],
                epoch, idx, all_loss_improvements
            )
            
            # FIX: Store quality scores for analysis
            client_qualities[idx] = quality
            
            embedding = server.embedding_gen.generate_embedding(param_updates[idx])
            server.memory_bank.add_update(idx, embedding, quality, epoch)

        # Store global state BEFORE similarity computation
        current_state = server._get_model_state_copy()
        server.memory_bank.store_global_state(epoch, current_state)
        
        # Update memory bank round count properly - ONLY ONCE per round
        server.memory_bank.update_round_count()
        print(f"Round {epoch}: Memory bank round_count updated to {server.memory_bank.round_count}")

        # Improved embedding diversity check
        if epoch >= 5 and server.memory_bank.round_count > 5:
            all_embeddings = []
            for client_id in selected_clients:
                if client_id in server.memory_bank.client_embeddings:
                    recent_emb = server.memory_bank.client_embeddings[client_id][-1]
                    all_embeddings.append(recent_emb.flatten())
            
            if len(all_embeddings) > 1:
                all_embeddings = np.array(all_embeddings)
                pairwise_similarities = []
                for i in range(len(all_embeddings)):
                    for j in range(i+1, len(all_embeddings)):
                        sim = np.dot(all_embeddings[i], all_embeddings[j]) / (
                            np.linalg.norm(all_embeddings[i]) * np.linalg.norm(all_embeddings[j]) + 1e-8
                        )
                        pairwise_similarities.append(sim)
                
                print(f"Round {epoch}: Embedding diversity - mean similarity: {np.mean(pairwise_similarities):.4f}, "
                      f"std: {np.std(pairwise_similarities):.4f}")

        # Get aggregation weights
        aggregation_weights = server.client_selector.get_aggregation_weights(
            selected_clients, client_models, data_sizes,
            server.global_direction, server.embedding_gen,
            server.quality_calc, epoch
        )

        # FIX: Calculate similarity scores for analysis
        similarities = []
        client_reliabilities = {}
        
        for client_id in selected_clients:
            # Get reliability
            reliability = server.memory_bank.get_client_reliability(client_id)
            client_reliabilities[client_id] = reliability
            
            # Calculate similarity if we have embeddings
            if client_id in server.memory_bank.client_embeddings and len(server.memory_bank.client_embeddings[client_id]) > 0:
                current_embedding = server.embedding_gen.generate_embedding(param_updates[client_id])
                sim = server.memory_bank.compute_similarity(client_id, current_embedding)
                similarities.append(sim)

        # FIX: Store average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        analyzer.similarity_scores.append(avg_similarity)

        # FIX: Track client rankings for stability analysis
        if client_reliabilities:
            rankings = sorted(client_reliabilities.items(), key=lambda x: x[1], reverse=True)
            analyzer.client_ranking_history.append(rankings)

        # FIX: Track quality scores
        analyzer.quality_scores_history.append(client_qualities)

        # Store aggregation weights for analysis
        analyzer.track_aggregation_weights(epoch, aggregation_weights)
        
        # Debug output for weights
        print(f"Round {epoch}: Weights = {aggregation_weights}")
        print(f"Round {epoch}: Weight sum = {sum(aggregation_weights.values())}")
        print(f"Round {epoch}: Weight std = {np.std(list(aggregation_weights.values()))}")

        # Update global model with weighted aggregation
        server.update_global_model(client_models, client_losses, data_sizes, aggregation_weights)
        
        # Update global direction AFTER model update
        updated_state = server._get_model_state_copy()
        if server.prev_model_state is not None:
            server.global_direction = {
                name: updated_state[name] - server.prev_model_state[name]
                for name in updated_state
            }
        server.prev_model_state = updated_state

        # Evaluation and logging
        loss_avg = np.mean([loss_after for _, loss_after in client_losses.values()])
        train_loss.append(loss_avg)

        # ADD THIS: Print test accuracy every round for auto_compare parsing
        test_acc_current, _ = test_inference(args, global_model, test_dataset)
        print(f"Round {epoch+1}: Test Accuracy = {test_acc_current*100:.2f}%")
      
        # FIX: Calculate round time properly
        round_time = time.time() - round_start_time
        
        # FIX: Comprehensive tracking for analysis
        analyzer.track_round_time(round_time)
        analyzer.track_client_selection(selected_clients)
        analyzer.track_memory_bank_size(len(server.memory_bank.memories))
        
        # Track client reliability improvements
        for client_id in selected_clients:
            reliability = server.memory_bank.get_client_reliability(client_id)
            analyzer.track_client_reliability(client_id, reliability)

        # Compute training accuracy (FIX: Only evaluate selected clients for fair comparison)
        list_acc = []
        # global_model.eval()
        # for c in selected_clients:  # Only selected clients, not all clients
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                             idxs=user_groups[c], logger=None)
        #     acc, _ = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        
        
        for c in selected_clients:  # Only selected clients
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[c], logger=None)
            
            # FIXED: Evaluate on training data instead of test data
            correct, total = 0, 0
            loss_sum = 0
            criterion = torch.nn.CrossEntropyLoss().to(device)
            
            with torch.no_grad():
                for images, labels in local_model.trainloader:  # ← Use trainloader!
                    images, labels = images.to(device), labels.to(device)
                    outputs = global_model(images)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total if total > 0 else 0
            list_acc.append(acc)



        train_accuracy.append(np.mean(list_acc))
        # Track accuracy for analysis
        analyzer.track_training_accuracy(train_accuracy[-1])
        
        # FIX: Comprehensive round data logging
        test_acc_for_round = None
        if epoch % print_every == 0 or epoch == args.epochs - 1:
            test_acc_for_round, _ = test_inference(args, global_model, test_dataset)
            test_accuracy_history.append(test_acc_for_round)
            analyzer.track_test_accuracy(test_acc_for_round)
            print(f"Round {epoch}: Test Accuracy = {test_acc_for_round:.4f}")

        # FIX: Log comprehensive round data
        analyzer.log_round_data(
            round_num=epoch,
            train_acc=train_accuracy[-1],
            train_loss=train_loss[-1],
            test_acc=test_acc_for_round,
            selected_clients=selected_clients,
            aggregation_weights=aggregation_weights,
            client_reliabilities=client_reliabilities,
            client_qualities=client_qualities,
            memory_bank_size=len(server.memory_bank.memories),
            avg_similarity=avg_similarity,
            round_time=round_time
        )

    # Final evaluation
    test_acc, _ = test_inference(args, global_model, test_dataset)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("Final Test Accuracy: {:.2f}%".format(100*test_acc))

    total_time = time.time() - start_time

    # Save train_loss and train_accuracy
    file_name = '../save/objects/PUMB_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(total_time))

    # Plot Loss and Accuracy
    plt.figure(figsize=(12, 8))
    plt.title(f'PUMB Results - Test Accuracy: {test_acc*100:.2f}%\nTraining Loss and Accuracy vs Communication Rounds')
    plt.xlabel('Communication Rounds')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(range(len(train_loss)), train_loss, color='r', label='Training Loss', linewidth=2)
    ax2.plot(range(len(train_accuracy)), train_accuracy, color='k', label='Training Accuracy', linewidth=2)
    ax1.set_ylabel('Training Loss', color='r')
    ax2.set_ylabel('Training Accuracy', color='k')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='k')

    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = (
        '../save/images/PUMB_{}_{}_{}_C[{}]_iid[{}]_alpha[{}]_E[{}]_B[{}]_explr[{}]_lr[{}]_initR[{}]_loss_acc_{}.png'
        .format(
            args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs,
            getattr(args, 'alpha', 'NA'),
            getattr(args, 'pumb_exploration_ratio', 'NA'),
            getattr(args, 'lr', 'NA'),
            getattr(args, 'pumb_initial_rounds', 'NA'),
            timestamp
        )
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Generate comprehensive analysis report
    comprehensive_filename = f"../save/logs/comprehensive_analysis_{timestamp}.txt"
    write_comprehensive_analysis(analyzer, args, test_acc, total_time, comprehensive_filename, experiment_seed)
    
    print(f"\n✅ Comprehensive analysis saved to: {comprehensive_filename}")
    print(f"📊 Basic experiment summary saved to: experiment_summary_{timestamp}.txt")
    
    # Print key metrics to console for immediate feedback
    convergence_metrics = analyzer.calculate_convergence_metrics()
    client_analysis = analyzer.analyze_client_selection_quality()
    memory_analysis = analyzer.analyze_memory_bank_effectiveness()
    
    print(f"\n🔍 QUICK RESULTS SUMMARY:")
    print(f"   Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Convergence Speed: {convergence_metrics.get('convergence_round', 'N/A')} rounds")
    print(f"   Training Stability: {convergence_metrics.get('training_stability', 0):.6f}")
    print(f"   Unique Clients Selected: {client_analysis.get('total_unique_clients', 0)}")
    print(f"   Memory Bank Final Size: {memory_analysis.get('final_memory_size', 0)}")
    print(f"   Avg Client Reliability Improvement: {client_analysis.get('avg_reliability_improvement', 0):.4f}")
    print(f"\n📁 All results saved in ../save/logs/ directory")
