import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

import torch.nn as nn
from torch_geometric.nn import TAGConv


class TAGNet(nn.Module):
    """
    This class defines a PyTorch module that takes in a graph represented in the PyTorch Geometric Data format,
    and outputs a tensor of predictions for each node in the graph. The model consists of one or more TAGConv layers,
    which are a type of graph convolutional layer.

    Args:
        node_dim (int): The number of node inputs.
        edge_dim (int): The number of edge inputs.
        output_dim (int, optional): The number of outputs (default: 1).
        hidden_dim (int, optional): The number of hidden units in each GNN layer (default: 50).
        n_gnn_layers (int, optional): The number of GNN layers in the model (default: 1).
        K (int, optional): The number of hops in the neighbourhood for each GNN layer (default: 2).
        dropout_rate (float, optional): The dropout rate to be applied to the output of each GNN layer (default: 0).

    """
    def __init__(self, node_dim, edge_dim, output_dim=1, hidden_dim=50, n_gnn_layers=1, K=2, bias=False, dropout_rate=0.5):
        super().__init__()
        self.node_dim = node_dim          
        self.edge_dim = edge_dim          
        self.output_dim = output_dim      
        self.hidden_dim = hidden_dim      
        self.n_gnn_layers = n_gnn_layers  
        self.K = K                        
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        self.prelus = nn.ModuleList()
        
        normalization="rw"

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(node_dim, output_dim, K, bias=bias))
        else:
            self.convs.append(TAGConv(node_dim, hidden_dim, K, bias=bias))
            self.prelus.append(nn.PReLU())

            for l in range(n_gnn_layers-2):
                self.convs.append(TAGConv(hidden_dim, hidden_dim, K, bias=bias))
                self.prelus.append(nn.PReLU())

            self.convs.append(TAGConv(hidden_dim, output_dim, K, bias=bias))

    def forward(self, data):
        """Applies the GNN to the input graph.

          Args:
              data (Data): A PyTorch Geometric Data object representing the input graph.

          Returns:
              torch.Tensor: The output tensor of the GNN.

          """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

      # print(x.shape)
      
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)

            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = self.prelus[i](x)
      
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
      
        return x
