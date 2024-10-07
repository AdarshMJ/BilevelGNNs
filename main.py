import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import os
import torch
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx,homophily
from torch_geometric.transforms import RandomNodeSplit
from model import GCN,GATv2, SimpleGCN,MLP
from dataloader import *
from nodeli import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import time
import csv
from train import train_and_get_results
from arguments import parse_args
import matplotlib.pyplot as plt
import pandas as pd

args = parse_args()


print("=== Script Arguments ===")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
print("========================")
print()


device = torch.device(args.device)
filename = args.out
p = args.dropout
lr = args.LR
hidden_dimension = args.hidden_dimension
splits = args.splits
seed = args.seed

print(f"Loading the dataset...")


if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo']:
    data, num_classes,num_features = load_data(args.dataset,args.num_train,args.num_val)
    #print(data)

elif args.dataset in ['cornell.npz','texas.npz','wisconsin.npz']:
    path = '/home/adarshjamadandi/Bilevel/bilevel/data'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print()
    print("Splitting datasets train/val/test...")
    transform2 = RandomNodeSplit(split="train_rest",num_splits=100,num_test=0.2,num_val=0.2)
    data  = transform2(data)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

elif args.dataset in ['chameleon_filtered.npz','squirrel_filtered.npz','actor.npz','roman_empire.npz','minesweeper.npz']:
    path = '/home/adarshjamadandi/Bilevel/bilevel/data'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print("Splitting datasets train/val/test...")
    transform2 = RandomNodeSplit(split="train_rest", num_splits=100, num_test=0.2, num_val=0.2)
    data = transform2(data)
    print()
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")


num_train_nodes = data.train_mask.sum().item()
num_val_nodes = data.val_mask.sum().item()
num_test_nodes = data.test_mask.sum().item()
print()
print(f"Number of training nodes: {num_train_nodes/splits}")
print(f"Number of validation nodes: {num_val_nodes/splits}")
print(f"Number of test nodes: {num_test_nodes/splits}")
datasetname, _ = os.path.splitext(args.dataset)



## For Community+SimilarityBased Rewiring ####
# algo_stime = time.time()
# data,nmiscoremod_before = modify_graph(data,args.dataset,budget_edges_add,budget_edges_delete,seed)
# algo_etime = time.time()
# rewire_time =  algo_etime - algo_stime
# print(f"Time Taken for Rewiring : {rewire_time}")


# newG = to_networkx(data, to_undirected=True)
# nxcg_G = nxcg.from_networkx(newG) 
# communities_after = list(nx.community.louvain_communities(nxcg_G, seed=seed))
# cluster_dict_after = {node: i for i, cluster in enumerate(communities_after) for node in cluster}
# cluster_list_after = [cluster_dict_after[node] for node in range(len(data.y))]
# nmiscoremod_after = NMI(cluster_list_after, data.y.cpu().numpy())






# print("Calculating Edge Label Informativeness...")
# graphaf, labelsaf = get_graph_and_labels_from_pyg_dataset(data)
# edgeliaf = li_edge(graphaf, labelsaf)
# print(f'Edge label informativeness: {edgeliaf:.4f}')

print("=============================================================")

print()


# print("Calculating Full Graph Adjusted Homophily...")
# hadjfull = h_adj(graphaf, labelsaf)
# print(f'Full Graph Adjusted Homophily: {hadjfull:.4f}')
# print()


data = data.to(device)
print()
print("Start Training...")
##=========================##=========================##=========================##=========================
model1 = GCN(num_features, hidden_dimension, num_classes, num_layers=args.num_layers)
model2 = MLP(num_features,num_classes,hidden_dimension)

model1.to(device)
model2.to(device)

print(model1)
print(model2)

gcn_start = time.time()
finaltestacc, teststd, finalvalacc, valstd, dirichlet_energies = train_and_get_results(data, model1, model2, lr, args.seed, args.splits, weight_decay=args.weight_decay, inner_lr=args.inner_lr, inner_iterations=args.inner_iterations,epochs=args.epochs, gamma=args.gamma)
gcn_end = time.time()

# Write Dirichlet energies to a separate CSV file
dirichlet_csv_filename = f"{args.dataset}_DE_layers_{args.num_layers}.csv"
df = pd.DataFrame({
    'Epoch': range(1, len(dirichlet_energies) + 1),
    'Dirichlet Energy': dirichlet_energies
})
df.to_csv(dirichlet_csv_filename, index=False)
print(f"Dirichlet energies saved to {dirichlet_csv_filename}")

if args.dataset.endswith('.npz'):
    dataset_name = args.dataset.replace('.npz', '').replace('_filtered', '').capitalize()
else:
    dataset_name = args.dataset



headers = ['Dataset','AvgValAcc','DeviationVal','AvgTestAcc', 'Deviation','HiddenDim','LR','Dropout','GCNTime','InnerIterations','InnerLR','Gamma']

with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                    writer.writerow(headers)
            writer.writerow([args.dataset,f"{(finalvalacc):.2f}", f"{(valstd):.2f}",f"{(finaltestacc):.2f}", f"{(teststd):.2f}",
            hidden_dimension,lr,p,gcn_end-gcn_start,args.inner_iterations,args.inner_lr,args.gamma])


