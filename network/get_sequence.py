# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:34:57 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch
from .base_skip import NoChange
from .base_activations import Activation
from .base_sparse import Sparse

# %% get sequence


def generate_sequence(net_setup, layer = 'Linear', layer_parameters = {},
                      activation = 'ReLU', activation_parameters = {},
                      batch_norm = None, batch_norm_parameters = {},
                      dropout = None, dropout_parameters = {},
                      sequence: torch.nn.Module = None) -> torch.nn.Module:
    '''
    Generate an sequence based on the information in the setup_dict.

    Parameters
    ----------
    setup : dict
        The setup of the model saved in the dict. Needs to contain all information about the model.
    sequence : torch.nn.Module, optional
        Add to defined model to the sequence. If not included, the function will generate a new empty sequence.

    Returns
    -------
    sequence : torch.nn.Module
        The new generated sequence.

    '''
    network_setup = net_setup.split('-')
    sequence = torch.nn.Sequential() if sequence == None else sequence
    layer = getattr(torch.nn, layer) if layer else NoChange
    activation = getattr(torch.nn, activation) if activation else NoChange
    batch_norm = getattr(torch.nn, batch_norm) if batch_norm else NoChange
    dropout = torch.nn.Dropout if dropout else NoChange
    
    for idx in range(len(network_setup) - 1):
        batch_norm_dict = {**{'num_features': int(network_setup[idx+1])}, **batch_norm_parameters}
        sequence.add_module(str(len(sequence))+'_'+'Calc', layer(int(network_setup[idx]), 
                            int(network_setup[idx+1]), **layer_parameters))
        sequence.add_module(str(len(sequence))+'_Act', activation(**activation_parameters))
        sequence.add_module(str(len(sequence))+'_BatchNorm', batch_norm(**batch_norm_dict))
        sequence.add_module(str(len(sequence))+'_Drop', dropout(**dropout_parameters))
        sequence.add_module(str(len(sequence))+'_Activation', Activation())
        # sequence.add_module(str(len(sequence))+'_Fuzzy', fuzzy(**setup['fuzzy_args']))
    
    ### remove last activation
    sequence[-4] = NoChange()        
            
    return sequence




