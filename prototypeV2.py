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
from sklearn.preprocessing import MinMaxScaler
##############################################################################

start = time.time()                                                             ## STARTER EN COUNTER FOR AT MÅLE TID

graphs = list()
for k in range(0,15):
    graphs.extend(['graphs_%s.pkl' %k])                                         ## DETTE ER EN LISTE OVER ALLE FIL-NAVNENE
                                                                                #   ( KAN BRUGES TIL AT LOOPE OVER SENERE)


class Net(torch.nn.Module):                                                     
    def __init__(self):                                                         #    
        super(Net, self).__init__()                                             #
        self.conv1 = GCNConv(5, 20)                                             #
        self.pool  = TopKPooling(20,ratio = 0.01)                               # LAG I MODELLEN
        self.conv2 = GCNConv(20, 15)                                            #                          
        self.pool2  = TopKPooling(15,ratio = 0.1)                               #
        self.conv3 = GCNConv(15, 12)                                            #
        self.pool3  = TopKPooling(12,ratio = 0.1)                               #    
        self.conv4 = GCNConv(12, 10)                                            #
        self.nn1   = torch.nn.Linear(10, 8)                                     #
                                                                                # 
    def forward(self, data):                                                    #
        x, edge_index, batch = data.x, data.edge_index, data.batch              #                   
                                                                                #    
        x = self.conv1(x, edge_index)                                           #    
        x = F.relu(x)                                                           #
        #x = F.dropout(x, training=self.training)                               #    
                                                                                #
                                                                                # HVORDAN INPUT LEVERES MELLEM LAG
        x, edge_index,_,batch,_,_ = self.pool(x,edge_index,None,batch)          #                     
                                                                                # 
        x = self.conv2(x, edge_index)                                           #
                                                                                #
        x, edge_index,_,batch,_,_ = self.pool2(x,edge_index,None,batch)         #                                                               #                               #
                                                                                #
        x = self.conv3(x, edge_index)                                           #
                                                                                # 
        x, edge_index,_,batch,_,_ = self.pool3(x,edge_index,None,batch)         #    
                                                                                #    
        x = self.conv4(x, edge_index)                                           #
                                                                                # 
        x = F.relu(self.nn1(x))                                                 #
                                                                                #
        return F.log_softmax(x, dim=1)                                          #    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # VÆLGER GPU HVIS MULIGT
model = Net().to(device)                                                        # MOUNTER MODEL I GPU/CPU
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)    # OPTIMIZER-FUNCTION TIL TRÆNING
loss_func = MSELoss()                                                           # LOSS-FUNCTION TIL TRÆNING

data_list = torch.load('E:\\final_project\\data\\graphs\\%s' %graphs[0])        # HENTER 1 GRAF-FIL ( VI KAN LOOPE OVER DETTE SENERE)
n_epochs = 200

batch_size = 32
scaler = MinMaxScaler()
loader = DataLoader(data_list, batch_size = batch_size)                         # LOADER DATA
for i in range(0,len(loader)):                                                  # LOOP OVER BATCHES
    data_train = next(iter(loader))                                             # HIVER DATA UD AF DataLoader-FORMATET
    data_train = data_train.to(device)                                          #
    ## 'PRE'-PROCESSING
    if i == 0:
        scaler.fit(data_train.x.cpu().numpy()[:,0:5])                           # 6. KOLONNE ER 'SRTInIcePulses'
        data_train.x = torch.tensor(scaler.transform(data_train.x.cpu().numpy()[:,0:5]), dtype = torch.float).cuda()
    else:
        data_train.x = torch.tensor(scaler.transform(data_train.x.cpu().numpy()[:,0:5]), dtype = torch.float).cuda()
    
    model.train()                                                               #
    for epoch in range(n_epochs):                                               # LOOP OVER EPOCHS    # SELVE TRÆNINGEN   
        optimizer.zero_grad()                                                   #
        out = model(data_train)                                                 #
        loss = loss_func(out, data_train.y)                                     #
        loss.backward()                                                         #
        optimizer.step()
        print('BATCH: %s / %s || EPOCH / %s : %s || MSE: %s' %(i, 
                                                               len(loader),
                                                               epoch, 
                                                               n_epochs
                                                               , loss.data.item()))                                                        #
    
print('TOTAL TRAINING TIME ON %s GRAPHS: %s' %(len(data_list),
                                               (time.time() - start)/60))
#%%
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