from torch import nn
import math
import torch
import numpy as np
from torch.nn import functional as F


def knn_cos_sim_2(x,k):
    #[batch,dim,channel] 2,256,128 
    x_norm =  F.normalize(x, p=2 , dim =1)
    #print(x_norm)
    x_norm_T = x_norm.transpose(2,1)
    cos_sim = torch.matmul(x_norm_T, x_norm)
    #print(cos_sim)
    idx = cos_sim.topk(k = k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=4):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn_cos_sim_2(x, k=k)   # (batch_size, num_points, k)
    #device = torch.device('cuda')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    idx_base = torch.arange(0, batch_size,device =device).view(-1, 1, 1)*num_points

    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    #feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return (feature-x).permute(0,3,1,2)    # (batch_size, 2*num_dims, num_points, k)

class edge_feature(nn.Module):
    def __init__(self,dim = 256):
        super(edge_feature,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1 , out_channels=dim , kernel_size = dim, stride =1)
        self.conv2 = nn.Conv1d(in_channels=1 , out_channels=dim , kernel_size = dim, stride =1)

    def forward(self ,f_i_set, f_j_set_k):
        f_i_set = f_i_set[:,:,None,...]#torch.Size([32, 128, 1, 256])
        f_j_set_k = f_j_set_k[:,:,None,...]#torch.Size([32, 128, 1, 256, 4])
        B, num_nodes,_ ,_ ,k = f_j_set_k.size()
        for num_node in range(num_nodes):
            f_i = f_i_set[: , num_node,:]
            f_j_k = f_j_set_k[: , num_node,...]
            #print(f_i.size(),f_j_k.size())
            for j in range(k):
                f_j = f_j_k[...,j]
                v_f_j_i = self.conv2(f_j-f_i)
                v_f_i = self.conv1(f_i)
                feat = nn.ReLU(inplace=False)(v_f_i+v_f_j_i)
                if  j==0:
                    feats = feat
                else:
                    feats = torch.cat([feats, feat], dim=-1)
            aggre_feat = torch.max(feats,dim=-1)[0][:,None,:]
            
            if num_node ==0:
                edge_features = aggre_feat
            else:
                edge_features =  torch.cat([edge_features, aggre_feat], dim=1)
        #print(edge_features.size())
        return edge_features

if __name__ == '__main__':
    fi,fjk =torch.ones(32, 128, 256), torch.ones(32,128,256,4)
    model = edge_feature()
    y = model(fi,fjk)
    print(y.shape)