import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

from src.models.CHEB import ChebNet
from models.GCN import GCN
from src.models.TAG import TAGNet

global model_dict
model_dict = {'CHEB': ChebNet, 'GCN': GCN, 'TAG': TAGNet}