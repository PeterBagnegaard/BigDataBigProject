import pandas as pd
import torch
def Create_Edges(data,event_no):
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
    return edge_index