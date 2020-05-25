import sys
sys.path.insert(0, 'C:\\applied_ML\\final_project\\tools')

from Create_Graph import Create_Graph

import sqlite3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import time
import matplotlib.cm as cm

import torch
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.nn as gnn
from torch_geometric.nn import SAGPooling

from torch.nn import MSELoss


#%%
db_file =  'C:\\applied_ML\\final_project\\data\\160000_00.db'
with sqlite3.connect(db_file) as con:
    query = 'select * from sequential'
    sequential = pd.read_sql(query, con)
    query = 'select * from scalar'
    scalar = pd.read_sql(query, con)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print('AVAILABLE TABLES:')
    print(cursor.fetchall())
#%%
scalar_small = scalar.loc[0:1000,:]
sequential_small = pd.DataFrame()

for k in range(0,len(scalar_small)):
    index =  sequential['event_no'] == scalar_small['event_no'][k]
    sequential_small = sequential_small.append(sequential.loc[index,:])

event_no= np.array(sequential_small['event_no'])[0]
index = sequential_small['event_no'] == event_no

first_event = pd.DataFrame(sequential_small.loc[index,:])

#%%
data = pd.DataFrame(sequential_small.loc[index,:])
index = data['event_no'] == event_no
event = pd.DataFrame(data.loc[index,:])
upper = list([0])
lower = list([1])
for k in range(1,len(event)):
    upper.append(k)
    if k != (len(event) - 1):
        upper.append(k)
for j in range(1,len(upper)-2,2):
    lower.append(upper[j-1])
    lower.append(upper[j+2])
lower.append(upper[len(upper) - 2])

edge_index = torch.tensor([upper,
                           lower], dtype = torch.long)

x = list()
data_index  = np.array(data['index'])
for i in range(0,len(data)):
    x.append(list(data.loc[data_index[i],['dom_x','dom_y','dom_z','dom_charge','dom_time']]))
x = torch.tensor(x,dtype = torch.float)
y = torch.tensor(scalar_small.loc[scalar_small['event_no'] == event_no,['true_primary_energy','true_primary_time',
                                                            'true_primary_position_x','true_primary_position_y',
                                                            'true_primary_position_z', 'true_primary_direction_x',
                                                            'true_primary_direction_y','true_primary_direction_z']].values,
                 dtype = torch.float)
data = Data(x = x, edge_index = edge_index,y=y)


#%%

test = DataLoader(data,batch_size = 1)



dataset = data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.x.shape[1], 8)
        self.pool  = SAGPooling(8,ratio = 0.02) 
        self.conv2 = GCNConv(8, dataset.y.shape[1])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        pooling = self.pool(x,edge_index)
        x, edge_index = pooling[0], pooling[1]
        
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_func = MSELoss()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_func(out, data.y)
    loss.backward()
    optimizer.step()
    
model.eval()
pred = model(data)
correct = data.y
acc = (abs(pred - data.y)/abs(data.y)).detach().cpu().numpy()

print('ACCURACY BY FEATURE IN %:')
count = 0
for feature in scalar_small.columns[2:9]:    
    print('%s : %s' %(feature,acc[0][count]))
    count = count + 1

