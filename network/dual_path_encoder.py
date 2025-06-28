# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %%

class AdaptiveFusion(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Attention Mechanismus zum Lernen der Fusion
        self.concept = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, feature_dim),
            torch.nn.ReLU(),
        )
        self.fc2 = torch.nn.Linear(feature_dim, feature_dim)
        self._init_last_layer()
        self.concept.add_module('last_layer', self.fc2)
        
        
        # Projektionen für Residualverbindungen
        self.proj1 = torch.nn.Linear(feature_dim, feature_dim)
        self.proj2 = torch.nn.Linear(feature_dim, feature_dim)
        
        # Normalisierung
        self.norm = torch.nn.LayerNorm(feature_dim)
        self.sigmoid = torch.nn.Sigmoid()
        
    def _init_last_layer(self):
        # Set weights to 0 (so they don't affect the output)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        # Set biases to 0 (so input to sigmoid is 0 → output is 0.5)
        # torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.normal_(self.fc2.bias, mean=0.5, std=0.1)
        
    def forward(self, x1, x2):
        
        # Originalfeatures für Residualverbindung speichern
        res1, res2 = x1, x2
        
        # Kontextinformationen berechnen
        x_cat = torch.cat([x1, x2], dim=-1)
        self.attention_weights = self.concept(x_cat)  # [batch, num_heads]
        
        
        # tau = 10.0  # Hyperparameter
        # self.attention_weights = torch.softmax(self.attention_weights / tau, dim=-1)
        self.attention_weights = self.sigmoid(self.attention_weights)
        # Verschiedene Fusionsoperationen berechnen
        mean_fused = (x1 + x2) / 2
        max_fused = torch.minimum(x1, x2)
        
        # Unterschiedliche Fusionsstrategien
        fused = self.attention_weights * mean_fused + (1 - self.attention_weights) * max_fused

    
        # Residualverbindungen
        fused = fused + self.proj1(res1) + self.proj2(res2)
        
        # Normalisierung und Dropout
        fused = self.norm(fused)
        
        return fused

# %%

