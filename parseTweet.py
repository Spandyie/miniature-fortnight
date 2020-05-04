#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:01:22 2019

@author: spandan
"""




class RNN(nn.module):
    def __init__(self, input_size, hidden_layer, output_layer):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_layer = output_layer
        self.i2h = nn.LSTMCell(self.input_size, self.hidden_size)
        self.h2o = nn.Liner(self.hidden_size, self.output_layer)
        
    def forward(self,x):
        x = s.view(batch_size * self.input_size,-1)
        x = self.i2h(x)
        x = self.h2o(x)
        return x
    
    
    
    
    
    
        

