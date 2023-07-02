import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

import time
import torch
from utils import get_datasets, train_model
from src.hyperparams_optimization import hyperparams_optimization
from src.models.CHEB import ChebNet
from models.GCN import GCN
from src.models.TAG import TAGNet


#### GET DATASETS ####
tra_dataset_pyg, tes_dataset_pyg, val_dataset_pyg, n_nodes = get_datasets()
#######################

### COMMON PARAMETERS ###
node_dim =   tra_dataset_pyg[0].x.shape[1]
edge_dim =   tra_dataset_pyg[0].edge_attr.shape[1]
output_dim = tra_dataset_pyg[0].y.shape[1]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_time = time.time()
model_name = f'model_{start_time}.pth'
########################

model_to_train = "CHEB" # "CHEB", "GCN", "TAG"

### CHEBNET TRAINING ###
if model_to_train == "CHEB":
    # Set model parameters 
    num_epochs = 50
    patience = 15
    learning_rate = 0.01
    batch_size = 64
    hidden_dim = 32
    n_gnn_layers = 2
    K = 10
    dropout_rate = 0.5
    save_model = False

    model = ChebNet(node_dim=node_dim, edge_dim=edge_dim, output_dim=output_dim, hidden_dim=hidden_dim, n_gnn_layers=n_gnn_layers, K=K, dropout_rate=dropout_rate).to(device)
    """train_model(
        model, 
        tra_dataset_pyg, 
        val_dataset_pyg,
        n_nodes=n_nodes,
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        patience=patience, 
        learning_rate=learning_rate, 
        save_model=save_model, 
        device=device, 
        model_name=model_name
    )
    """

    # Run hyperparameter optimization
    n_trials = 100
    hyperparams_optimization(
        model_type="CHEB", 
        node_dim=node_dim,
        edge_dim=edge_dim, 
        output_dim=output_dim, 
        tra_dataset_pyg=tra_dataset_pyg, 
        val_dataset_pyg=val_dataset_pyg, 
        n_nodes=n_nodes,
        n_trials=n_trials, 
        learning_rate_range=(0.001, 0.01),
        batch_size_values=[32, 64], 
        hidden_dim_range=(4, 24),
        n_gnn_layers_range = (2, 8),
        K_range = (2, 25),
        dropout_rate_range = (0.25, 0.6),
        num_epochs=30, 
        patience=15, 
        device=device, 
        model_name=model_name
    )
    ########################


### GCN TRAINING ###
if model_to_train == "GCN": 
    # Set model parameters
    hidden_dim=50
    n_gnn_layers=1
    dropout_rate=0.5
    K=1
    batch_size = 64
    num_epochs = 20
    patience = 15
    learning_rate = 0.001
    save_model = False
    
    model = GCN(node_dim=node_dim, edge_dim=edge_dim, output_dim=output_dim, hidden_dim=hidden_dim, n_gnn_layers=n_gnn_layers, K=K, dropout_rate=dropout_rate).to(device)
    train_model(
        model, 
        tra_dataset_pyg, 
        val_dataset_pyg, 
        n_nodes=n_nodes,
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        patience=patience, 
        learning_rate=learning_rate, 
        save_model=save_model, 
        device=device, 
        model_name=model_name
    )


    # Run hyperparameter optimization
    n_trials = 100
    hyperparams_optimization(
        model_type="GCN", 
        node_dim=node_dim,
        edge_dim=edge_dim, 
        output_dim=output_dim, 
        tra_dataset_pyg=tra_dataset_pyg, 
        val_dataset_pyg=val_dataset_pyg, 
        n_nodes=n_nodes,
        n_trials=n_trials, 
        learning_rate_range=(0.001, 0.01),
        batch_size_values=[32, 64], 
        hidden_dim_range=(4, 24),
        n_gnn_layers_range = (2, 8),
        K_range = (2, 25),
        dropout_rate_range = (0.25, 0.6),
        num_epochs=30, 
        patience=15, 
        device=device, 
        model_name=model_name
    )    
########################


### TAGNet TRAINING ###
if model_to_train == "TAG":
    # Set model parameters
    hidden_dim=50
    n_gnn_layers=1
    dropout_rate=0.5
    K=1
    batch_size = 64
    num_epochs = 20
    patience = 15
    learning_rate = 0.001
    save_model = False

    model = TAGNet(node_dim=node_dim, edge_dim=edge_dim, output_dim=output_dim, hidden_dim=hidden_dim, n_gnn_layers=n_gnn_layers, K=K, dropout_rate=dropout_rate).to(device)
    train_model(
        model, 
        tra_dataset_pyg, 
        val_dataset_pyg, 
        n_nodes=n_nodes,
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        patience=patience, 
        learning_rate=learning_rate, 
        save_model=save_model, 
        device=device, 
        model_name=model_name
    )


    # Run hyperparameter optimization
    n_trials = 100
    hyperparams_optimization(
        model_type="TAG", 
        node_dim=node_dim,
        edge_dim=edge_dim, 
        output_dim=output_dim, 
        tra_dataset_pyg=tra_dataset_pyg, 
        val_dataset_pyg=val_dataset_pyg, 
        n_nodes=n_nodes,
        n_trials=n_trials, 
        learning_rate_range=(0.001, 0.01),
        batch_size_values=[32, 64], 
        hidden_dim_range=(4, 24),
        n_gnn_layers_range = (2, 8),
        K_range = (2, 25),
        dropout_rate_range = (0.25, 0.6),
        num_epochs=30, 
        patience=15, 
        device=device, 
        model_name=model_name
    )    
########################