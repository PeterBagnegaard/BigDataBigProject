# -*- coding: utf-8 -*-
"""
Get Ice Cube data
"""
import sqlite3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

db_file = Path().home().joinpath('Desktop').joinpath('Big Data Analysis').joinpath('Final_Project').joinpath('IceCube').joinpath('Data').joinpath('120000_00.db')
with sqlite3.connect(db_file) as con:
    query = 'select * from sequential'
    sequential = pd.read_sql(query, con)
    query = 'select * from scalar'
    scalar = pd.read_sql(query, con)

# Get smaller subset of data
sq = sequential.loc[:1000, :]
sc = scalar.loc[:1000, :]

#%%
"""
Convert Ice Cube data to torch geometry format
"""
from torch_geometric.data import Data
import torchvision.transforms as transforms

# Get numpy values
X_tilde = sq.values
y_tilde = y_tilde.values

X_trans = transforms.Normalize(X_tilde)
#y_trans = transforms.Normalize(y_tilde)

X = torch.tensor(X_trans, dtype=torch.float)
y = torch.tensor(y_tilde, dtype=torch.float)

data = Data(x=x, edge_index=edge_index)




#%%
"""
Create class that initializes 
graph convolutional neural network
models with a forward function
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    

#%%
"""
Create and train model
"""
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
#%%
    
model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))