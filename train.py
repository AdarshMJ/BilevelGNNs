import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def proximal_operator(h, gamma):
    """
    Implements the proximal operator:
    prox_(eta,gamma)(h) := arg min(eta(z) + 1/2gamma ||z−h||^2)
    where eta(z) = I_infty[z <0] (indicator function)
    """
    return F.relu(h)  # This effectively projects onto the non-negative orthant

def proximal_gradient_step(param, grad, lr, gamma):
    """
    Performs a proximal gradient step
    """
    # Standard gradient step
    param_new = param - lr * grad
    # Proximal operator step
    return proximal_operator(param_new, gamma)

##Taken from https://github.com/roth-andreas/rank_collapse###
def dirichlet_energy(x, edge_index, edge_weights):
    edge_difference = edge_weights * (torch.norm(x[edge_index[0]] - x[edge_index[1]], dim=1) ** 2)
    return edge_difference.sum() / 2

def plot_dirichlet_energy(epochs, dirichlet_energies, title, num_layers):
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, dirichlet_energies)  # Use semilogy for log scale on y-axis
    plt.title(f"{title} (Layers: {num_layers})")
    plt.xlabel('Epoch')
    plt.ylabel('Dirichlet Energy (log scale)')
    plt.savefig(f"{title.replace(' ', '_')}_layers_{num_layers}.png")
    plt.close()

def train_and_get_results(data, model, mlpmodel, p, lr, seed, splits, weight_decay, inner_lr, inner_iterations, epochs, gamma):
    
    dirichlet_energies = []

    def visualize_embeddings(embeddings, labels, title):
        tsne = TSNE(n_components=2, random_state=seed)
        embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels.cpu().numpy(), cmap='viridis')
        plt.colorbar(scatter)
        plt.title(title)
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()

    def inner_loop_loss(node_reps, transformed_reps, gcn_out, mlp_out):
        rep_loss = torch.norm(transformed_reps - node_reps, p=2, dim=1).mean()
        huber_loss_val = F.smooth_l1_loss(gcn_out, mlp_out)
        return rep_loss + huber_loss_val

    def train_inner_loop(node_reps, gcn_out, mlp_out):
        transformed_reps = torch.matmul(node_reps, model.C)
        losses = []
        for i in range(inner_iterations):
            # Forward pass with current C
            transformed_reps = torch.matmul(node_reps, model.C)
            
            # Compute the inner loss
            loss = inner_loop_loss(node_reps, transformed_reps, gcn_out, mlp_out)
            losses.append(loss.item())
            
            # Clear gradients before backward pass
            model.zero_grad()
            
            # Backward pass
            loss.backward(retain_graph=True)

            # Apply proximal gradient update to model.C
            with torch.no_grad():
                grad = model.C.grad
                model.C.data = proximal_gradient_step(model.C, grad, inner_lr, gamma)

        return transformed_reps, losses

    def train(epoch):
        model.train()
        mlpmodel.train()
        optimizer.zero_grad()
        
        # Forward pass to get initial representations
        initial_node_reps = model.get_representations(data.x, data.edge_index)
        initial_gcn_out = model.convs[-1](initial_node_reps, data.edge_index)
        mlp_out = mlpmodel(data.x)
        
        # Visualize initial embeddings
        #if epoch == 1:
        #    visualize_embeddings(initial_node_reps, data.y, f"Initial Embeddings (Epoch {epoch})")
        
        # Inner loop optimization using proximal gradient
        optimized_reps, inner_losses = train_inner_loop(initial_node_reps, initial_gcn_out, mlp_out)
        
        # Visualize optimized embeddings
        #if epoch % 100 == 0:
        #    visualize_embeddings(optimized_reps, data.y, f"Optimized Embeddings (Epoch {epoch})")
        
        # Use optimized representations for final GCN layer
        final_gcn_out = model.convs[-1](optimized_reps, data.edge_index)
        
        # Outer loop loss (cross-entropy) using optimized representations
        outer_loss = F.cross_entropy(final_gcn_out[train_mask], data.y[train_mask])
        
        # Calculate Dirichlet energy
        dirichlet_energy_val = dirichlet_energy(optimized_reps.detach().cpu(), data.edge_index.detach().cpu(), torch.ones(data.edge_index.size(1)).detach().cpu())
        
        # Backward pass for outer optimization
        outer_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            pred = final_gcn_out.argmax(dim=1)
            train_correct = pred[train_mask] == data.y[train_mask]
            train_acc = int(train_correct.sum()) / int(train_mask.sum())
        
        return outer_loss, train_acc, inner_losses, dirichlet_energy_val

    def val():
        model.eval()
        with torch.no_grad():
            node_reps = model.get_representations(data.x, data.edge_index)
            optimized_reps = torch.matmul(node_reps, model.C)
            out = model.convs[-1](optimized_reps, data.edge_index)
            pred = out.argmax(dim=1)
            val_correct = pred[val_mask] == data.y[val_mask]
            val_acc = int(val_correct.sum()) / int(val_mask.sum())
        return val_acc

    def test():
        model.eval()
        with torch.no_grad():
            node_reps = model.get_representations(data.x, data.edge_index)
            optimized_reps = torch.matmul(node_reps, model.C)
            out = model.convs[-1](optimized_reps, data.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[test_mask] == data.y[test_mask]
            test_acc = int(test_correct.sum()) / int(test_mask.sum())
        return test_acc

    # Rest of the function remains the same
    test_acc_allsplits = []
    val_acc_allsplits = []
    for split_idx in range(splits):
        print(f"\n{'='*20} Starting Split {split_idx + 1}/{splits} {'='*20}")
        model.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(mlpmodel.parameters()), lr=lr, weight_decay=weight_decay)
        
        train_mask = data.train_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        
        set_seed(seed)
        print("Starting training...")
        
        outer_pbar = tqdm(range(1, epochs + 1), desc="Outer Loop")
        dirichlet_energies = []
        for epoch in outer_pbar:
            loss, train_acc, inner_losses, dirichlet_energy_val = train(epoch)
            dirichlet_energies.append(dirichlet_energy_val.item())
            if epoch % 10 == 0:
                val_acc = val()
                outer_pbar.set_postfix({
                    "Epoch": epoch,
                    "Outer Loss": f"{loss.item():.4f}",
                    "Inner Loss": f"{inner_losses[-1]:.4f}",
                    "Train Acc": f"{train_acc:.4f}",
                    "Val Acc": f"{val_acc:.4f}",
                    "Dirichlet Energy": f"{dirichlet_energy_val.item():.4f}"
                })
                
                # Plot inner loop losses
                #plt.figure(figsize=(10, 5))
                #plt.plot(inner_losses)
                #plt.title(f"Inner Loop Losses (Epoch {epoch})")
               # plt.xlabel("Inner Iteration")
               # plt.ylabel("Loss")
                #plt.savefig(f"inner_losses_epoch_{epoch}.png")
                #plt.close()
        
        # Plot Dirichlet energy
        #plot_dirichlet_energy(range(1, epochs + 1), dirichlet_energies, f"Dirichlet Energy (Split {split_idx + 1})", model.num_layers)
        
        val_acc = val()            
        test_acc = test()
        final_test_acc = test_acc * 100
        final_val_acc = val_acc * 100
        test_acc_allsplits.append(final_test_acc)
        val_acc_allsplits.append(final_val_acc)

        print(f"Split {split_idx + 1} Results:")
        print(f"  Test Accuracy: {final_test_acc:.2f}%")
        print(f"  Validation Accuracy: {final_val_acc:.2f}%")
    
    print("\n" + "="*50)
    print("Final Results:")
    print(f"Number of splits: {len(test_acc_allsplits)}")
    print(f"Average Test Accuracy: {np.mean(test_acc_allsplits):.2f}% ± {2 * np.std(test_acc_allsplits) / np.sqrt(len(test_acc_allsplits)):.2f}%")
    print(f"Average Validation Accuracy: {np.mean(val_acc_allsplits):.2f}% ± {2 * np.std(val_acc_allsplits) / np.sqrt(len(val_acc_allsplits)):.2f}%")

    return final_test_acc, 2 * np.std(test_acc_allsplits) / np.sqrt(len(test_acc_allsplits)), final_val_acc, 2 * np.std(val_acc_allsplits) / np.sqrt(len(val_acc_allsplits)), dirichlet_energies
