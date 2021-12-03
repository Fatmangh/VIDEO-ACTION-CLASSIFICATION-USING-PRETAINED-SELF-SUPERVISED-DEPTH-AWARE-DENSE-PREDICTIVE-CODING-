import math
import numpy as np
import sys
sys.path.append('../backbone')
from select_backbone import select_resnet
from convrnn import ConvGRU

import torch
import torch.nn as nn
import torch.nn.functional as F

class LC(nn.Module):
    def __init__(self, sample_size, num_seq, seq_len, useBoth,
            network='resnet18', dropout=0.5, num_class=101, packnet=None):
        super(LC, self).__init__()
        torch.cuda.manual_seed(666)
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.num_class = num_class 
        print('=> Using RNN + FC model ')

        print('=> Use 2D-3D %s!' % network)
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        track_running_stats = True 

        self.backbone, self.param = select_resnet(network, track_running_stats=track_running_stats)
        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        print('=> using ConvRNN, kernel_size = 1')
        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
        self._initialize_weights(self.agg)
        insize = 256 if packnet == None else 768
        insize = 512 if packnet != None and useBoth == False else insize 
        self.final_bn = nn.BatchNorm1d(insize)
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.final_fc = nn.Sequential(nn.Dropout(dropout),
                nn.Linear(insize, self.num_class))
        self._initialize_weights(self.final_fc)
        self.packnet = packnet
        self.useBoth = useBoth

    def forward(self, block):
        # seq1: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        
       # get depth embeddings
        if self.packnet != None:
            print('Getting depth embd.........')
            packnet_input = block
            self.packnet.to(block.get_device())
            packnet_input = packnet_input.permute(0, 2, 1, 3, 4).reshape(B*N,SL, C, H, W)
            #print('packnet input: ,', packnet_input.shape)
            depth_emb = None
            for i in range(packnet_input.shape[0]):
                out = self.packnet.depth(torch.unsqueeze(packnet_input[i, 2,:, :, :], dim=0))['depth_embedding'][0] # [N, 4 , 4]
                out = torch.unsqueeze(out, dim=0)
                if depth_emb != None: 
                    depth_emb = torch.vstack([depth_emb, out])
                else:
                    depth_emb = out
                 
                del out

            _, D, W_, H_ = depth_emb.shape
            depth_emb = depth_emb.view(B, N, D, self.last_size, self.last_size)
            depth_emb = F.relu(depth_emb)
            #print('final latent space shape: ', depth_emb.shape)
            del packnet_input
        
        if (self.packnet != None and self.useBoth == True) or (self.packnet == None and self.useBoth == False): 
            print('Getting spatio-temporal embd.........')
            feature = self.backbone(block)
            feature = F.relu(feature)
            feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
            feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B*N,D,last_size,last_size]
            #print('dpc feaature shape', feature.shape)

        del block

        if self.packnet != None and self.useBoth == True:
            print('Combinning spatio-temporal and  depth embd.........')
            feature = torch.cat((feature, depth_emb), 2)
            del depth_emb
        elif self.packnet != None:
            feature = depth_emb
            del depth_emb
        
        #print('final feature shape: ',feature.shape)
    
        feature = F.avg_pool3d(feature, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
    
        feature = self.final_bn(feature.transpose(-1,-2)).transpose(-1,-2) # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
        #print('fc input shape', feature.shape)
        output = self.final_fc(feature).view(B, -1, self.num_class)

        #print('final output has shape: ', output.shape)
        return output,None

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)        
        # other resnet weights have been initialized in resnet_3d.py


