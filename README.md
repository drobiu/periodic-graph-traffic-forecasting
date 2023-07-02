
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
git clone https://github.com/francescoopiccoli/periodic-graph-traffic-forecasting.git
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

`models` directory contains the different graph neural networks utilized.

`train.py` file is used to train the models defined in the `models` folder.

`utils.py` file contains the functions used to train and the functions to create the product graph.

`hyperparams_optimization.py` file contains the code we used to perform hyperparams optimization.

`GLOBAL.py` defines a dictionary mapping the models' names to the models' class objects.

