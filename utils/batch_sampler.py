import random
import torch.utils.data as tordata
import numpy as np

class BatchSampler( tordata.sampler.Sampler):
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        while(True):
            sample_indices = []
            
            label_list = random.sample(list(self.dataset.set_label), self.batch_size)
            
            for label in label_list:
                
                s_idx = random.choices(self.dataset.index_dict[label], k=2)
                
                sample_indices += s_idx
            
            yield sample_indices
            
    def __len__(self):
        return self.dataset.data_size
    
    
def collate_fnn( batch ):
    
    batch_size = len(batch)
    
    peis = [batch[i][0] for i in range(batch_size)]
    view = [batch[i][1] for i in range(batch_size)]
    label = [batch[i][2] for i in range(batch_size)]
    
    data = [[0 for _ in range(batch_size // 2)] for _ in range(2)]
    channel = [[0 for _ in range(batch_size // 2)] for _ in range(2)]
    _view = [ list() for _ in range(2)]
    _label = [ list() for _ in range(2)]
    
    for i in range(batch_size // 2):
            
        ch_k = [[0 for _ in range(len(peis[0]))] for _ in range(2) ]
        
        k_1 = random.randint(0, len(peis[0])-1)
        k_2 = random.randint(0, len(peis[0])-1)
        
        ch_k[0][k_1] = 1
        ch_k[1][k_2] = 1
        
        channel[0][i] = ch_k[0]
        channel[1][i] = ch_k[1]
        
        data[0][i] = peis[i*2]
        data[1][i] = peis[i*2+1]
        
        _view[0].append(view[i*2])
        _view[1].append(view[i*2 + 1])
        
        _label[0].append(label[i*2])
        _label[1].append(label[i*2 + 1])
        
    return data, channel, _view, _label