#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch 

from models.LDNet import * 
from models.LDNet_mobilenet import * 
from models.LDNet_resnet import * 




class LDNet_network():
    
        
    def build(backbone_network,modelpath,CONFIG):
                
        # if backbone_network.lower() == 'darknet':
        #     net = darknet.RT(n_classes=19, pretrained=is_train,PRETRAINED_WEIGHTS=CONFIG.PRETRAINED_DarkNET19)
        if backbone_network == 'LDNet':
            net = LDNet(n_classes=CONFIG.n_classes,img_ch=3,output_ch=1)
        elif backbone_network == 'mobilenet':
            net = LDNet_mobile(n_classes=CONFIG.n_classes,img_ch=3,output_ch=1)
        elif backbone_network == 'resnet':
            net = LDNet_resnet(n_classes=CONFIG.n_classes,img_ch=3,output_ch=1)
        else:
            raise NotImplementedError
            
        if modelpath is not None:
            net.load_state_dict(torch.load(modelpath))
            
        print("Using LDNet with",backbone_network)
        return net
        
            
    
