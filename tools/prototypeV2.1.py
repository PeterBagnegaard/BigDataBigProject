import torch                                      
import pandas as pd
import numpy as np
import time
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import MSELoss
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn.models.graph_unet import GraphUNet
from torch.nn import Dropout
import matplotlib.pyplot as plt
##############################################################################

start = time.time()                                                             ## STARTER EN COUNTER FOR AT MÅLE TID

graphs = list()
for k in range(0,15):
    graphs.extend(['graphs_%s.pkl' %k])                                         ## DETTE ER EN LISTE OVER ALLE FIL-NAVNENE
                                                                                #   ( KAN BRUGES TIL AT LOOPE OVER SENERE)


class Net(torch.nn.Module):                                                     
    def __init__(self):                                                         #    
        super(Net, self).__init__()                                             #                                            #
        self.pool1  = TopKPooling(5,ratio = 0.01)
        self.nn1   = torch.nn.Linear(5,64)                               # LAG I MODELLEN#                          
        self.pool2  = TopKPooling(64,ratio = 0.1)                               #
        self.pool3  = TopKPooling(64,ratio = 0.1)                               #                                                                                   #
        
        self.nn2   = torch.nn.Linear(64,3)                                             #
                                                                                # 
    def forward(self, data):                                                    #
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch              #                   
                                   #    
                                                                                #
                                                                                # HVORDAN INPUT LEVERES MELLEM LAG
        x, edge_index,_,batch,_,_ = self.pool1(x,edge_index,None,batch)          #                     
          
        x = self.nn1(x)                                                                       # 
                                          #
                                                                                #
        x, edge_index,_,batch,_,_ = self.pool2(x,edge_index,None,batch)         #                                                               #                               #
                                                                                #
                                         #
                                                                                # 
        x, edge_index,_,batch,_,_ = self.pool3(x,edge_index,None,batch)         #    
                                                                                #    
                                           #
                                                                                # #
        x = self.nn2(x)                                                                         #
        return x                                          #    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # VÆLGER GPU HVIS MULIGT
model = Net().to(device)
batch_size = 32
                                                                                # MOUNTER MODEL I GPU/CPU
lr = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)    # OPTIMIZER-FUNCTION TIL TRÆNING
loss_func = MSELoss()                                                           # LOSS-FUNCTION TIL TRÆNING

data_list = torch.load('E:\\final_project\\data\\graphs\\%s' %graphs[0])        # HENTER 1 GRAF-FIL ( VI KAN LOOPE OVER DETTE SENERE)
n_epochs = 20

loss_list =list()

scaler = StandardScaler()
loader = DataLoader(data_list, batch_size = batch_size)                         # LOADER DATA
for i in range(0,len(loader)):                                                         # LOOP OVER BATCHES
    data_train = next(iter(loader))                                             # HIVER DATA UD AF DataLoader-FORMATET
    data_train = data_train.to(device)                                          #
    ## 'PRE'-PROCESSING
    if i == 0:
        scaler.fit(data_train.x.cpu().numpy()[:,0:5])                           
        data_train.x = torch.tensor(scaler.transform(data_train.x.cpu().numpy()[:,0:5]), dtype = torch.float).float().cuda()
    else:
        data_train.x = torch.tensor(scaler.transform(data_train.x.cpu().numpy()[:,0:5]), dtype = torch.float).float().cuda()
    data_train.y = torch.tensor(data_train.y.cpu().numpy()[:,5:8],dtype=torch.float).float().cuda()
    model.train()                                                               #
    for epoch in range(n_epochs):                                               # LOOP OVER EPOCHS    # SELVE TRÆNINGEN   
        optimizer.zero_grad()                                                   #
        out = model(data_train)                                                 #
        loss = loss_func(out, data_train.y.float())                                     #
        loss.backward()                                                         #
        optimizer.step()
        print('BATCH: %s / %s || EPOCH / %s : %s || MSE: %s' %(i, 
                                                               len(loader),
                                                               epoch, 
                                                               n_epochs
                                                               , loss.data.item()))                                                        #
        loss_list.append(loss.item())
print('TOTAL TRAINING TIME ON %s GRAPHS: %s' %(len(data_list),
                                               (time.time() - start)/60))

## PLOT LOSS
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')




data_list = torch.load('E:\\final_project\\data\\graphs\\%s' %graphs[1])
loader = DataLoader(data_list, batch_size = batch_size)
start = time.time()                                                             #
acc = 0
for i in range(0,len(loader)):                                                                         #
    data_pred = next(iter(loader))
    data_pred.x = torch.tensor(scaler.transform(data_pred.x.cpu().numpy()[:,0:5]),dtype=torch.float).float().cuda()
    data_pred.y = torch.tensor(data_pred.y.cpu().numpy()[:,5:8],dtype=torch.float).float().cuda()
    
    model.eval()                                                                # PREDICTION OG UDREGNING AF NMAE-SCORE   
    data = data_pred.to(device)                                                      #    ( BØR SKRIVES OM SENERE )
    pred = model(data)                                                          #
    correct = data.y                                                            #
    acc = acc + (abs(pred - data.y)/abs(data.y)).detach().cpu().numpy()         #
    print('PREDICTING: %s /  %s' %(i,len(loader)))                                                                        
res = acc.sum(0)/(batch_size*len(loader))
print(res)

torch.save(model.state_dict(), 'E:\\final_project\\models\\E_%s_NMAE%s.pt'%(str(scaler)[0:14], str(acc)[0:5]))