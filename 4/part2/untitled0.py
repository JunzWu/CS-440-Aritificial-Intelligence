# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:02:15 2019

@author: Junz
"""
import torch
x = torch.tensor([1., 2.])
x = x.cuda()
print(x)