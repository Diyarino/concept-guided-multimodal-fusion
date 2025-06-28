# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

from collections import OrderedDict
from .base_skip import NoChange
from .get_sequence import generate_sequence
from .dual_path_encoder import AdaptiveFusion

DEVICE = 'cuda'

# %% autoencoder

class MultimodalFusion(torch.nn.Module):
    '''
    Create a predefined Autoencoder Neural Network with concept fusion.

    Parameters
    ----------
    setup : dict
        The predefined training scenario.
    preprocess : torch.nn.Module, optional
        Data preprocessing module. The default is None.
    postprocess : torch.nn.Module, optional
        Data postprocessing module. The default is None.

    Returns
    -------
    None.

    '''

    def __init__(self, configs: list, name: str = '', debug: bool= False,
                 preprocess: torch.nn.Module = None, 
                 postprocess: torch.nn.Module = None):
        super().__init__(name, debug)
        
        
        preprocess = torch.nn.Sequential(OrderedDict(
            {'0_skip': NoChange()})) if preprocess == None else preprocess
        
        self.num_modalities = len(configs)
        
        encoder = torch.nn.Sequential()
        for idx, setup in enumerate(configs):
            encoder.add_module(str(idx)+'_encoder', generate_sequence(sequence=torch.nn.Sequential(), **setup))
        
        compute = AdaptiveFusion(feature_dim = 16)
        
        self.total_params = sum(p.numel() for p in compute.parameters())
        
        decoder = torch.nn.Sequential()
        for idx, setup in enumerate(configs):
            decode_setup = setup.copy()
            decode_setup['net_setup'] = '-'.join(list(reversed(setup['net_setup'].split('-'))))
            decode_setup['layer'] = decode_setup['layer'].replace('Conv', 'ConvTranspose')
            decoder.add_module(str(idx)+'_decoder', generate_sequence(sequence=torch.nn.Sequential(), **decode_setup))
        
        postprocess = torch.nn.Sequential(OrderedDict(
            {'1_skip': NoChange()})) if postprocess == None else postprocess

        self.network = torch.nn.Sequential()
        self.network.add_module('preprocess', preprocess)
        self.network.add_module('encoder', encoder)
        self.network.add_module('compute', compute)
        self.network.add_module('decoder', decoder)
        self.network.add_module('postprocess', postprocess)
        
        self.network.to(DEVICE)


    def forward(self, ins: torch.Tensor) -> tuple:
        '''
        Compute the output of the autoencoder with ins

        Parameters
        ----------
        ins : torch.Tensor
            The input of the neural network.

        Returns
        -------
        preprocess, representation, compute, reconstruction, postprocess : torch.Tensor
            The different steps of the autoencoder.

        '''
        # preprocess = self.network[0](ins)
        
        representation = [model(sample) for model, sample in zip(self.network[1], ins)]
        
        representation_flatten = [latent.flatten(start_dim = 1) for latent in representation]
        
        latent_shape = [latent.size() for latent in representation]
        
        sample_1, sample_2 = representation_flatten
                
        aggregated = self.network[2](sample_1, sample_2)
        
        aggregation_shaped = [aggregated.reshape(shape) for shape in latent_shape]
        
        reconstruction = [model(aggr_shaped) for model, aggr_shaped in zip(self.network[3], aggregation_shaped)]
        
        return representation_flatten, aggregation_shaped, reconstruction
    

    



    