import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
def Create_Graph(data,data_y,event_no):
    ## INPUT : data      - dette er sequential-data
    ##         data.y    - dette er scalar-data
    ##         event_no  - liste el. array af event_no. Feks. scalar['event_no'][0:100]   
    ## OUTPUT: liste af grafer passende til hvert event i input (sorteret som input-listen)
    res = list()
    for a in range(0,len(event_no)):
        index = data['event_no'] == event_no[a]
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
        
        data_index  = data['event_no'] == event_no[a]
        x = data[data_index]
        x = x.loc[:,['dom_x','dom_y','dom_z','dom_charge','dom_time']].values.tolist()
        x = torch.tensor(x,dtype = torch.float)
        index_y = data_y['event_no'] == event_no[a]
        y = torch.tensor(data_y.loc[index_y,['true_primary_energy','true_primary_time',
                                                                    'true_primary_position_x','true_primary_position_y',
                                                                    'true_primary_position_z', 'true_primary_direction_x',
                                                                    'true_primary_direction_y','true_primary_direction_z']].values,dtype = torch.float)
        data_tensor = Data(x = x, edge_index = edge_index,y=y)
        res.append(data_tensor) 
    return res