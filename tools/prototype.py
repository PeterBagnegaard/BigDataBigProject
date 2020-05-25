import torch
import sys
sys.path.insert(0, 'C:\\applied_ML\\final_project\\tools')                      # PATH TIL Create_Graph
from Create_Graph import Create_Graph                                           # Create_Graph Funktion
import pandas as pd
import numpy as np
import sqlite3
import time
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGPooling
from torch.nn import MSELoss
##############################################################################

start = time.time()                                                             ## STARTER EN COUNTER FOR AT MÅLE TID

db_file =  'C:\\applied_ML\\final_project\\data\\160000_00.db'                  #
with sqlite3.connect(db_file) as con:                                           #
    query = 'select * from sequential'                                          #
    sequential = pd.read_sql(query, con)                                        #  LÆSER 160000_00.db FILEN  
    query = 'select * from scalar'                                              #
    scalar = pd.read_sql(query, con)                                            #
    cursor = con.cursor()                                                       #
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")        #

scalar_small = scalar.loc[0:1000,:]                                             #
sequential_small = pd.DataFrame()                                               #    
for k in range(0,len(scalar_small)):                                            # LAVER scalar_small, sequential_small
    index =  sequential['event_no'] == scalar_small['event_no'][k]              # (De udgør de første 1000 events)
    sequential_small = sequential_small.append(sequential.loc[index,:])         #
                                                                                    
data_list = Create_Graph(sequential_small,scalar_small,scalar_small['event_no'])# LAVER GRAFER FRA scalar_small, sequential_small

class Net(torch.nn.Module):                                                     
    def __init__(self):                                                         #    
        super(Net, self).__init__()                                             #
        self.conv1 = GCNConv(5, 20)                                             #
        self.pool  = SAGPooling(20,ratio = 0.8)                                 # LAG I MODELLEN
        self.conv2 = GCNConv(20, 15)                                            #
        self.pool2 = torch.nn.AdaptiveMaxPool2d((1,8))                          #
        self.nn1   = torch.nn.Linear(8, 8)                                      #
                                                                                 
    def forward(self, data):                                                    #
        x, edge_index = data.x, data.edge_index                                 #
                                                                                #    
        x = self.conv1(x, edge_index)                                           #    
        x = F.relu(x)                                                           #
        x = F.dropout(x, training=self.training)                                #    
                                                                                #
        pooling = self.pool(x,edge_index)                                       # HVORDAN INPUT LEVERES MELLEM LAG
        x, edge_index = pooling[0], pooling[1]                                  #
                                                                                # 
        x = self.conv2(x, edge_index)                                           #
                                                                                # 
        x = self.pool2(x.unsqueeze(0)).squeeze(0)                               #
                                                                                # 
        x = F.relu(self.nn1(x))                                                 #
                                                                                #
        return F.log_softmax(x, dim=1)                                          #    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # VÆLGER GPU HVIS MULIGT
model = Net().to(device)                                                        # MOUNTER MODEL I GPU/CPU
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)    # OPTIMIZER-FUNCTION TIL TRÆNING
loss_func = MSELoss()                                                           # LOSS-FUNCTION TIL TRÆNING

n_epochs = 200
for data in data_list:                                                          # LOOP OVER GRAFER
    data = data.to(device)                                                      #
    model.train()                                                               #
    for epoch in range(n_epochs):                                               # LOOP OVER EPOCHS    
        print('GRAPH: %s / %s EPOCH / %s : %s' %(data_list.index(data)          #    
                                                 , len(data_list),              #
                                                 n_epochs,epoch))               # SELVE TRÆNINGEN   
        optimizer.zero_grad()                                                   #
        out = model(data)                                                       #
        loss = loss_func(out, data.y)                                           #
        loss.backward()                                                         #
        optimizer.step()                                                        #
    
print('TOTAL TRAINING TIME ON %s GRAPHS: %s' %(len(data_list),
                                               (time.time() - start)/60))

start = time.time()                                                             #
acc = 0                                                                         #
for data in data_list:                                                          #    
    model.eval()                                                                # PREDICTION OG UDREGNING AF NMAE-SCORE   
    data = data.to(device)                                                      #    ( BØR SKRIVES OM SENERE )
    pred = model(data)                                                          #
    correct = data.y                                                            #
    acc = acc + (abs(pred - data.y)/abs(data.y)).detach().cpu().numpy()         #
                                                                        
acc = acc/len(data_list)
print('ACCURACY BY FEATURE IN %:')
count = 0
for feature in scalar_small.columns[2:9]:    
    print('%s : %s' %(feature,acc[0][count]))
    count = count + 1
print((time.time() - start)/60)