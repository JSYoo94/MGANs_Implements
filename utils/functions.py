import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

def encode_view_transform(src_view, dst_view):

    view_encode = np.zeros([len(src_view), 14])
    
    for i in range(len(src_view)):
        if src_view[i]>dst_view[i]:
            if src_view[i]-dst_view[i]>5:
                view_encode[i,src_view[i]-1:] = 1
                view_encode[i,0:dst_view[i]-1] = 1
            else:
                view_encode[i,dst_view[i]-1:src_view[i]-1] = -1
        elif src_view[i]<dst_view[i]:
            if dst_view[i]-src_view[i]>5:
                view_encode[i,dst_view[i]-1:] = -1
                view_encode[i,0:src_view[i]-1] = -1
            else:
                view_encode[i,src_view[i]-1:dst_view[i]-1] = 1
            
    return Variable(torch.from_numpy(view_encode).float())
            
    
def encode_view_onehot(view):
       
    view_encode = np.zeros([len(view), 14])
    
    for i in range(len(view)):
        view_encode[i][view[i]-1] = 1
    
    return Variable(torch.from_numpy(view_encode).float())


def encode_label_onehot(label, set_labels):
    
    label_dict = {}
    for i in range(len(set_labels)):
        label_dict[set_labels[i]] = i
    
    label_encode = np.zeros([len(label), 1000])
    
    for i in range(len(label)):
        label_encode[i][label_dict[label[i]]] = 1
    
    return Variable(torch.from_numpy(label_encode).float())
        
def drawLoss(loss_dict):
    plt.style.use(['ggplot'])
    
    for key, value in loss_dict.items():
        x = np.arange(len(loss_dict[key]))
        plt.plot(x, loss_dict[key], label=key)
    
    plt.xlabel("train step")
    plt.ylabel("loss")
    
    plt.legend(loc="lower right")
    plt.show()