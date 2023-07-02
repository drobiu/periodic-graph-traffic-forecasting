import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

import optuna
import torch
import numpy as np
from torch_geometric.loader import DataLoader as pyg_DataLoader

from src.GLOBAL import model_dict
from src.utils import train_epoch, evaluate_epoch

# Hyperparameter optimization
def hyperparams_optimization(
        model_type, 
        node_dim, 
        edge_dim, 
        output_dim, 
        tra_dataset_pyg, 
        val_dataset_pyg, 
        n_nodes,
        n_trials=100, 
        learning_rate_range=(0.001, 0.01),
        batch_size_values=[32, 64], 
        hidden_dim_range=(4, 24),
        n_gnn_layers_range = (2, 8),
        K_range = (2, 25),
        dropout_rate_range = (0.25, 0.6),
        num_epochs=30, 
        patience=15, 
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
        model_name=None
    ):

    def objective(trial):
        if model_name:
            log_name = model_name.split(".")
            log_name = f"logs/{log_name[0]}_{trial.number}.log"
        else:
            log_name = f"logs/model_{trial.number}.log"
        # Set training parameters from Optuna suggestion
        learning_rate = trial.suggest_float('lr', *learning_rate_range)
        batch_size = trial.suggest_categorical('batch_size', batch_size_values)
        hidden_dim = trial.suggest_int('hidden_dim', *hidden_dim_range)
        n_gnn_layers = trial.suggest_int('n_gnn_layers',*n_gnn_layers_range)
        K = trial.suggest_int('K', *K_range)
        dropout_rate = trial.suggest_float('dropout_rate', *dropout_rate_range)
        
        with open(log_name, "a") as f:
            f.write(f"lr: {learning_rate}\nbatch size: {batch_size}\nhidden dim: {hidden_dim}\nn layers: {n_gnn_layers}\nK {K}\ndropout: {dropout_rate}\n")
          
        train_losses = []
        val_losses = []

        # Dictionary mapping the model type to the corresponding class
        Model = model_dict[model_type]

        # in GAT: k = num_head
        model = Model(node_dim, edge_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate).to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

        tra_loader = pyg_DataLoader(tra_dataset_pyg, batch_size=batch_size, shuffle=True)
        val_loader = pyg_DataLoader(val_dataset_pyg, batch_size=batch_size, shuffle=False)
        
        for epoch in range(1, num_epochs+1):
            print("epoch:", epoch)
            model.train()
            train_loss = train_epoch(model, tra_loader, optimizer, device=device)
            val_loss = evaluate_epoch(model=model, loader=val_loader, n_nodes=n_nodes, out_name=log_name, device=device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Early stopping
            try:  
                if val_losses[-1]>=val_losses[-2]:
                    early_stop += 1
                    if early_stop == patience:
                        print("Early stopping! Epoch:", epoch)
                        break
                else:
                    early_stop = 0
            except:
                early_stop = 0
                
            epoch_info = f"epoch: {epoch} \t training loss: {np.round(train_loss,4)} \t validation loss: {np.round(val_loss,4)}"
            print(epoch_info)
            with open(log_name, "a") as f:
                f.write(epoch_info)
            
            
        return val_loss

    # Optimize hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    print(f'Best hyperparameters: {best_params}\nBest validation loss: {best_value}')