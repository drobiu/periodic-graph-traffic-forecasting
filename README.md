
<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <br><br>
</p>

# Group 27: Machine Learning for Graph Data (CS4350) project 

## Authors
Aleksander Buszydlik

Karol Dobiczek

Francesco Piccoli

Edmundo Sanz-Gadea López

## User Manual
- Clone repo:
```
git clone https://github.com/drobiu/periodic-graph-traffic-forecasting.git
```
- Setup conda env:
```
conda env create -f environment.yml
```

## Repository structure
```
.
├── README.md
├── data
│   ├── ...
└── src
    ├── GLOBAL.py
    ├── __init__.py
    ├── models
    │   ├── CHEB.py
    │   ├── GCN.py
    │   ├── TAG.py
    │   ├── __init__.py
    │   ├── agcrn
    │   │   ├── ...
    │   ├── arima.py
    │   ├── dcrnn
    │   │   ├── ...
    ├── notebooks
    │   ├── basic_GNN.ipynb
    │   └── data_loading.ipynb
    ├── hyperparams_optimization.py
    ├── train.py
    └── utils.py
```

`models` directory contains the our model and the baselines utilized.

`train.py` file is used to train our model defined in the `models` folder.

`utils.py` file contains all the functions used to train and all the functions to create the product graph.

`hyperparams_optimization.py` file contains the code we used to perform hyperparams optimization.

