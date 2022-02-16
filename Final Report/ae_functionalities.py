#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer

# Create matrix of step functions

def stepfunctions(dim = 128, right_amplitudes = [0], left_amplitudes = [1]):

  """
  Inputs:
  -------
  - dim: int equal to the dimension of the step functions
  - left_amplitudes: array containing the values of the functions before the transition of size M. 
  - right_amplitudes: array containing the values of the functions after the transition of size M. 

  left_amplitudes and right_amplitudes should have the same size M. For each index i in range of M, dim step functions will be constructed having 
  the value before the transition equal to left_amplitudes[i] and the value after the transition equal to right_amplitudes[i]. 
  The step functions for the same index i, will only vary in the position of the transition. 
  Taking all possible cases, we obtain, for each index i, dim step functions.

  Output:
  ------
  The output is a matrix of dimension (dim, M*dim + 1). The lines of this matrix are the step functions.
  """
  assert len(left_amplitudes) == len(right_amplitudes), "right_amplitudes and left_amplitudes should have same length."

  M = len(left_amplitudes)

  X = torch.zeros(((dim+1)*M, dim))
  
  for i in range(M):
    X[i*(dim + 1): (i+1)*(dim) + i, :] = right_amplitudes[i] * torch.tril(torch.ones(dim, dim),diagonal=1) + left_amplitudes[i] * torch.triu(torch.ones(dim, dim))
    X[(i+1)*(dim) + i, :] = torch.ones(dim) * right_amplitudes[i]
                            
  return X

class AutoEncoder(nn.Module):

  def __init__(self, encoder, decoder):
    super(AutoEncoder, self).__init__()

    self.encoder = encoder
    self.decoder = decoder
    
    self.track_grads = False

    self.init_RMSprop_bool = False

  def forward(self, X):
    
    self.code = self.encoder(X)
    
    self.y = self.decoder(self.code)

    return self.y
            

  def track_gradients(self, track = False, params = 'all', names = []):
    """
    This function starts or stops the gradient tracking.
    In case we want to start the tracking, it takes the names of the weights to be tracked and prepares the containers.
    """
    self.track_grads = track

    if track:
        self.grads = []
        
        if params == 'all':
            for name, param in list(self.named_parameters()):
                self.grads.append({
                    'name': name,
                    'grad': torch.zeros(param.shape)
                })
            
        elif params == 'some' and len(names)>0:
            for name, param in list(self.named_parameters()):
                if name in names:
                    self.grads.append({
                    'name': name,
                    'grad': torch.zeros(param.shape)
                }) 
                
        else:
            print('params takes a string: \'all\' or \'some\'. If \'some\', names shouldn\'t be empty.')
        



  def SGD_step(self, alpha= 1e-3):
     
    """
    This function loops over the parameters of the autoencoder and update them using SGD
    """
        
    for name, param in list(self.named_parameters()):
    
        with torch.no_grad():
            
            param -= alpha * param.grad
            
            if self.track_grads:

                for i in range(len(self.grads)):
                    if self.grads[i]['name'] == name:
                        self.grads[i]['grad'] = param.grad.clone()
                        break
                
        param.grad.zero_()

  def init_RMSprop(self):

    self.S = []

    for name, param in list(self.named_parameters()):

      self.S.append( 1e-6 * torch.ones(param.shape) )

    self.init_RMSprop_bool = True

  def RMSprop_step(self, alpha= 1e-3, gamma = 0.9):
     
    """
    This function loops over the parameters of the autoencoder and update them using SGD
    """
    
    if not self.init_RMSprop_bool:

      print("Init RMSprop")

      self.init_RMSprop()

    j=0

    for name, param in list(self.named_parameters()):
    
        with torch.no_grad():
            
            self.S[j] = gamma * self.S[j] + (1-gamma) * param.grad **2

            param -= alpha * param.grad / torch.sqrt(self.S[j])
            
            if self.track_grads:

                for i in range(len(self.grads)):
                    if self.grads[i]['name'] == name:
                        self.grads[i]['grad'] = param.grad.clone()
                        break
                
        param.grad.zero_()
        
        j += 1
  

