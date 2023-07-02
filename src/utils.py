import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

import time
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from datetime import datetime
from models.agcrn.lib.metrics import All_Metrics
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as pyg_DataLoader

# Similarity function between two timestamps
def sim(t1, t2, index, decay=0.0, hour_weight=0.5):
    hour_weight = np.clip(hour_weight, 0, 1)

    f1 = encode_time(t1)
    f2 = encode_time(t2)

    day_similarity = np.dot(f1[:2], f2[:2]) / (np.linalg.norm(f1[:2]) * np.linalg.norm(f2[:2]))
    hour_similarity = np.dot(f1[2:], f2[2:]) / (np.linalg.norm(f1[2:]) * np.linalg.norm(f2[2:]))

    return  (hour_similarity * hour_weight + day_similarity * (1 - hour_weight)) * ((1 - decay) ** index)

# Encode a timestamp into a vector
def encode_time(timestamp):
    week_day = datetime.fromisoformat(timestamp).weekday()
    week_day_sin = np.sin(2 * np.pi * week_day / 7.0)
    week_day_cos = np.cos(2 * np.pi * week_day / 7.0)

    hour = datetime.fromisoformat(timestamp).hour
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    return week_day_sin, week_day_cos, hour_sin, hour_cos


def create_spatiotemporal_data(adjacency, horizon, days_back=np.array([1, 1, 0, 0, 0, 0, 0, 1]), time_weight=1, prev_horizon=6):
    n_nodes = len(adjacency)
    active_replicas = horizon + prev_horizon * np.count_nonzero(days_back[1:])

    # offsets for node indices
    node_offsets = np.arange(active_replicas) * n_nodes

    day_timesteps = 24 * 60 / 5
    time_offsets = [] # offsets for time indices

    for i, day in enumerate(days_back):
        if i == 0:
            time_offsets.append(np.arange(horizon))

        elif day == 0:
            continue

        else:
            # shift time over half of the horizon
            time_offsets.append((np.arange(prev_horizon) - int(prev_horizon/2) + i * day_timesteps).astype(int))

    time_offsets = np.concatenate(time_offsets, dtype=int).reshape(-1)
    
    # calculate edge indices of the original adjacency
    edge_index_orig = []
    features_orig = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue

            if adjacency[i, j] > 0:
                edge_index_orig.append([i, j])
                features_orig.append(adjacency[i, j])

    edge_index_orig = np.array(edge_index_orig)
    features_orig = list(features_orig)

    edge_index = []
    # duplicate edge weights for each replica
    features = features_orig * active_replicas

    # duplicate edge indices for each replica
    for offset in node_offsets:
        edge_index = [*edge_index, *(edge_index_orig + offset)]

    # connect nodes over timesteps in the same segment
    for i in range(n_nodes):
        for offset in node_offsets[:-1]:
            from_node = offset + i + n_nodes
            to_node = offset + i
            edge_index.append([from_node, to_node])
            edge_index.append([to_node, from_node])
            features.append(sim(data[data.columns[0]][from_node], data[data.columns[0]][to_node], offset))
            features.append(sim(data[data.columns[0]][from_node], data[data.columns[0]][to_node], offset))
            
    # connect different time segments
    for i in range(n_nodes):
        for offset in node_offsets[int(horizon/2)-1::horizon][1:]:
            from_node = offset + i + n_nodes
            edge_index.append([from_node, i])
            edge_index.append([i, from_node])
            features.append(sim(data[data.columns[0]][from_node], data[data.columns[0]][i], offset))
            features.append(sim(data[data.columns[0]][from_node], data[data.columns[0]][i], offset))

    edge_index = np.array(edge_index)
    features = np.array(features).reshape(-1, 1)

    print("Edges:", len(features))
    return edge_index, features, time_offsets


def train_epoch(model, loader, optimizer, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    Trains a neural network model for one epoch using the specified data loader and optimizer.

    Args:
        model (nn.Module): The neural network model to be trained.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
        device (str): The device used for training the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """

    # opt = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for batch in loader:
        batch = batch.to(device) # GPU

        optimizer.zero_grad()
        y_hat = model(batch)
        loss = nn.L1Loss()
        loss = loss(batch.y, y_hat)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())

    return np.mean(losses)


def evaluate_epoch(model, loader, n_nodes, out_name, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    Evaluates the performance of a trained neural network model on a dataset using the specified data loader.

    Args:
        model (nn.Module): The trained neural network model to be evaluated.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the evaluation data.
        device (str): The device used for evaluating the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """

    logs = []

    losses = []
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device) # GPU

            y_hat = model(batch)
            loss = nn.L1Loss()
            loss = loss(batch.y, y_hat)
            y_pred.append(y_hat)
            y_true.append(batch.y)
            losses.append(loss.cpu().numpy())
        
    y_pred = torch.cat(y_pred[:n_nodes], dim=0).cpu().numpy()    
    y_true = torch.cat(y_true[:n_nodes], dim=0).cpu().numpy()

    for t in range(y_true.shape[1]):
        mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], None, 0.)
        out = "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            t + 1, mae, rmse, mape*100)
        print(out)
        logs.append(out)
    mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, None, 0.)
    out = "Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                mae, rmse, mape*100)
    print(out)
    logs.append(out)

    with open(out_name, "w") as f:
        f.write(str(logs))

    return np.mean(losses)


global data

def get_datasets():
    global data
    data = pd.read_csv('data/metr-la.csv')
    ids = np.loadtxt('data/graph_sensor_ids_small.csv', delimiter=",", dtype=int)
    str_ids = [str(x) for x in ids] # for pandas


    with open('data/adj_mx.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    id_map = p[1]
    adj = p[2]

    adj_ids = [id_map[str(i)] for i in ids]

    reduced_adj = adj[adj_ids].T[adj_ids].T

    # edge_index = []
    # features = []

    # for i, id_a in enumerate(adj_ids):
    #     for j, id_b in enumerate(adj_ids):
    #         if i == j:
    #             continue
            
    #         if reduced_adj[i, j] > 0:
    #             edge_index.append([i, j])
    #             features.append(reduced_adj[i, j])
                
    # edge_index = np.array(edge_index).T
    # features = np.array(features).reshape(-1, 1)
    n_nodes = len(reduced_adj)
    horizon = 12
    edge_index, features, time_offsets = create_spatiotemporal_data(reduced_adj, horizon=horizon)

    train_test_val_split = [0.8, 0.1, 0.1]
    n = 10000 # make sure this number is smaller than len(data) - 2 * horizon and higher than 2018

    tra_dataset_pyg = []
    tst_dataset_pyg = []
    val_dataset_pyg = []

    np_data = data[str_ids].to_numpy()

    n_features = 1
    use_time_encoding = False

    if use_time_encoding:
        n_features += 4

    for i in range(np.max(time_offsets)+1, n):
        data_offsetted_x = []
        data_offsetted_y = []
        
        for offset in -time_offsets:
            new_x = []
            if use_time_encoding:
                time_encoding = encode_time(data[data.columns[0]][i + offset])
                for measurement in np_data[i + offset].T:
                    new_x.append([*time_encoding, measurement])
            else:
                new_x = np_data[i + offset].T
            data_offsetted_x.append(new_x)
            data_offsetted_y.append(np_data[i + offset + 1 : i + offset + horizon + 1].T)
            
        data_offsetted_x = np.array(data_offsetted_x).reshape(-1, n_features)
        data_offsetted_y = np.array(data_offsetted_y).reshape(-1, horizon)
        
        pyg_data = Data(x=torch.Tensor(data_offsetted_x),
                y=torch.Tensor(data_offsetted_y),
                edge_index=torch.Tensor(edge_index.T),
                edge_attr=torch.Tensor(features))


        
        if (i / n < train_test_val_split[0]):
            tra_dataset_pyg.append(pyg_data)
        elif (i / n < train_test_val_split[0] + train_test_val_split[1]):
            tst_dataset_pyg.append(pyg_data)
        else:
            val_dataset_pyg.append(pyg_data)
        print(f"Sample {i+1} / {n}", end='\r')

    print('Number of training samples:',   len(tra_dataset_pyg))
    print('Number of validation samples:', len(val_dataset_pyg))
    print('Number of test samples:',       len(tst_dataset_pyg))

    return tra_dataset_pyg, tst_dataset_pyg, val_dataset_pyg, n_nodes


def train_model(
        model, 
        tra_dataset_pyg, 
        val_dataset_pyg, 
        n_nodes,
        batch_size=64, 
        num_epochs=30, 
        patience=15, 
        learning_rate=0.001,
        save_model=False, 
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
        model_name=None
    ):

    # Create the optimizer to train the neural network via back-propagation
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # Create the training and validation dataloaders to "feed" data to the GNN in batches
    tra_loader = pyg_DataLoader(tra_dataset_pyg, batch_size=batch_size, shuffle=True)
    val_loader = pyg_DataLoader(val_dataset_pyg, batch_size=batch_size, shuffle=False)
    #create vectors for the training and validation loss
    train_losses = []
    val_losses = []

    #start measuring time
    start_time = time.time()

    for epoch in range(1, num_epochs+1):
        model_name = model_name
        
        # Model training
        model.train()
        train_loss = train_epoch(model, tra_loader, optimizer, device=device)

        # Model validation
        val_loss = evaluate_epoch(model=model, loader=val_loader, n_nodes=n_nodes, out_name=f"{model_name}_{epoch}.log", device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping
        try:  
            if val_losses[-1]>=val_losses[-2]:
                
                if epoch % 10 == 0:
                    print(f"saving {model_name}")
                    torch.save(model, model_name)
                    
                early_stop += 1
                if early_stop == patience:
                    print("Early stopping! Epoch:", epoch)
                    break
            else:
                print(f"saving {model_name}")
                torch.save(model, model_name)
                early_stop = 0
        except:
            early_stop = 0

        print("epoch:",epoch, "\t training loss:", np.round(train_loss,4),
                            "\t validation loss:", np.round(val_loss,4))

    elapsed_time = time.time() - start_time
    print(f'Model training took {elapsed_time:.3f} seconds')

    if save_model:
        torch.save(model, f'model_{start_time}_final.pth')



#from torch_geometric.explain import Explainer, GNNExplainer

"""def explain_model(model, val_dataset_pyg, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )"""