import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from os.path import join
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class HIPT_None_FC(nn.Module):
    def __init__(self, path_input_dim=384, size_arg = "small", dropout=0.25, n_classes=2):
        super(HIPT_None_FC, self).__init__()
        self.size_dict_path = {"small": [path_input_dim, 256, 256], "big": [path_input_dim, 512, 384]}
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_phi = nn.Sequential(
            nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25),
        )
        self.local_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
        
        ### Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25),
        )
        self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])
        self.classifier = nn.Linear(size[1], n_classes)


    def forward(self, h):
        x_256 = h

        ### Local
        h_256 = self.local_phi(x_256)
        A_256, h_256 = self.local_attn_pool(h_256)  
        A_256 = A_256.squeeze(dim=2) # A = torch.transpose(A, 1, 0)
        A_256 = F.softmax(A_256, dim=1) 
        h_4096 = torch.bmm(A_256.unsqueeze(dim=1), h_256).squeeze(dim=1)
        
        ### Global
        h_4096 = self.global_phi(h_4096)
        A_4096, h_4096 = self.global_attn_pool(h_4096)  
        A_4096 = torch.transpose(A_4096, 1, 0)
        A_4096 = F.softmax(A_4096, dim=1) 
        h_path = torch.mm(A_4096, h_4096)
        h_path = self.global_rho(h_path)
        logits = self.classifier(h_path)

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, None, None



######################################
# Deep Attention MISL Implementation #
######################################
class MIL_Cluster_FC(nn.Module):
    def __init__(self, path_input_dim=1024, num_clusters=10, size_arg = "small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Cluster_FC, self).__init__()
        self.size_dict_path = {"small": [path_input_dim, 512, 256], "big": [path_input_dim, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.num_clusters = num_clusters
        
        ### FC Cluster layers + Pooling
        size = self.size_dict_path[size_arg]
        if path_input_dim == 384:
            size = [path_input_dim, path_input_dim, 256]
            
        phis = []
        for phenotype_i in range(num_clusters):
            phi = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
            phis.append(nn.Sequential(*phi))
        self.phis = nn.ModuleList(phis)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        
        ### WSI Attention MIL Construction
        fc = [nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        self.phis = self.phis.to(device)
        self.pool1d = self.pool1d.to(device)
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, data, **kwargs):
        x_path = data
        cluster_id = kwargs['cluster_id'].detach().cpu().numpy()
        ### FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            h_cluster_i = self.phis[i](x_path[cluster_id==i])
            if h_cluster_i.shape[0] == 0:
                h_cluster_i = torch.zeros((1,384)).to(torch.device('cuda'))
            h_cluster.append(self.pool1d(h_cluster_i.T.unsqueeze(0)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0)

        ### Attention MIL
        A, h_path = self.attention_net(h_cluster)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h = self.rho(h_path).squeeze()

        logits  = self.classifier(h).unsqueeze(0)

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, None, None

import sys
sys.path.append('../Pretraining/')
from vision_transformer4k import vit4k_xs

class HIPT_LGP_FC(nn.Module):
    def __init__(self, path_input_dim=384,  size_arg = "small", dropout=0.25, n_classes=4,
     pretrain_4k='None', freeze_4k=False, pretrain_WSI='None', freeze_WSI=False):
        super(HIPT_LGP_FC, self).__init__()
        self.size_dict_path = {"small": [384, 192, 192], "big": [1024, 512, 384]}
        #self.fusion = fusion
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_vit = vit4k_xs()
        if pretrain_4k != 'None':
            print("Loading Pretrained Local VIT model...",)
            state_dict = torch.load('../ckpts/pretrain/%s.pth' % pretrain_4k, map_location='cpu')['teacher']
            state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
            state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.local_vit.load_state_dict(state_dict, strict=False)
            print("Done!")
        if freeze_4k:
            print("Freezing Pretrained Local VIT model")
            for param in self.local_vit.parameters():
                param.requires_grad = False
            print("Done")

        ### Global Aggregation
        self.pretrain_WSI = pretrain_WSI
        if pretrain_WSI != 'None':
            pass
        else:
            self.global_phi = nn.Sequential(nn.Linear(192, 192), nn.ReLU(), nn.Dropout(0.25))
            self.global_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=192, nhead=3, dim_feedforward=192, dropout=0.25, activation='relu'
                ), 
                num_layers=2
            )
            self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
            self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])

        self.classifier = nn.Linear(size[1], n_classes)
        

    def forward(self, x_256):
        ### Local
        h_4096 = self.local_vit(x_256.unfold(1, 16, 16).transpose(1,2))
        
        ### Global
        if self.pretrain_WSI != 'None':
            h_WSI = self.global_vit(h_4096.unsqueeze(dim=0))
        else:
            h_4096 = self.global_phi(h_4096)
            h_4096 = self.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
            A_4096, h_4096 = self.global_attn_pool(h_4096)  
            A_4096 = torch.transpose(A_4096, 1, 0)
            A_4096 = F.softmax(A_4096, dim=1) 
            h_path = torch.mm(A_4096, h_4096)
            h_WSI = self.global_rho(h_path)

        logits = self.classifier(h_WSI)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits, F.softmax(logits, dim=1), Y_hat, None, None

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.local_vit = nn.DataParallel(self.local_vit, device_ids=device_ids).to('cuda:0')
            if self.pretrain_WSI != 'None':
                self.global_vit = nn.DataParallel(self.global_vit, device_ids=device_ids).to('cuda:0')

        if self.pretrain_WSI == 'None':
            self.global_phi = self.global_phi.to(device)
            self.global_transformer = self.global_transformer.to(device)
            self.global_attn_pool = self.global_attn_pool.to(device)
            self.global_rho = self.global_rho.to(device)

        self.classifier = self.classifier.to(device)