from typing import Tuple, Optional
import torch
import numpy as np

from escnn import gspaces
from escnn import nn
from escnn import group
from escnn.nn import GeometricTensor

from torch_geometric.data import Data, Batch

class SO2MLP(nn.EquivariantModule):
    
    def __init__(self, n_classes=10):
        super(SO2MLP, self).__init__()
        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.in_type = self.gspace.type(self.G.standard_representation())

        activation1 = nn.FourierELU(
            self.gspace,
            channels=3, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=1).irreps, 
            inplace=True,
            type='regular', N=6,   
        )
        
        self.block1_mu = nn.SequentialModule(
            nn.Linear(self.in_type, activation1.in_type),
            nn.IIDBatchNorm1d(activation1.in_type),
            activation1,
        )
        
        activation2 = nn.FourierELU(
            self.gspace,
            channels=8, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=3).irreps, # include all frequencies up to L=3
            inplace=True,
            type='regular', N=16,
        )
        self.block2_mu = nn.SequentialModule(
            nn.Linear(self.block1_mu.out_type, activation2.in_type),
            nn.IIDBatchNorm1d(activation2.in_type),
            activation2,
        )
        
        activation3 = nn.FourierELU(
            self.gspace,
            channels=8, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=3).irreps, # include all frequencies up to L=3
            inplace=True,
            type='regular', N=16,
        )
        self.block3_mu = nn.SequentialModule(
            nn.Linear(self.block2_mu.out_type, activation3.in_type),
            nn.IIDBatchNorm1d(activation3.in_type),
            activation3,
        )
        
        activation4 = nn.FourierELU(
            self.gspace,
            channels=5, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=2).irreps, # include all frequencies up to L=2
            inplace=True,
            type='regular', N=12,
        )
        self.block4_mu = nn.SequentialModule(
            nn.Linear(self.block3_mu.out_type, activation4.in_type),
            nn.IIDBatchNorm1d(activation4.in_type),
            activation4,
        )
        
        self.out_type = self.gspace.type(self.G.irrep(2))
        self.block5_mu = nn.Linear(self.block4_mu.out_type, self.out_type)

        self.block1_sigma_sq = nn.SequentialModule(
            nn.Linear(self.in_type, activation1.in_type),
            nn.IIDBatchNorm1d(activation1.in_type),
            activation1,
        )

        self.block2_sigma_sq = nn.SequentialModule(
            nn.Linear(self.block1_sigma_sq.out_type, activation2.in_type),
            nn.IIDBatchNorm1d(activation2.in_type),
            activation2,
        )

        self.block3_sigma_sq = nn.SequentialModule(
            nn.Linear(self.block2_sigma_sq.out_type, activation3.in_type),
            nn.IIDBatchNorm1d(activation3.in_type),
            activation3,
        )

        self.block4_sigma_sq = nn.SequentialModule(
            nn.Linear(self.block3_sigma_sq.out_type, activation4.in_type),
            nn.IIDBatchNorm1d(activation4.in_type),
            activation4,
        )

        self.block5_sigma_sq = nn.Linear(self.block4_sigma_sq.out_type, self.out_type)
    
    def forward(self, positions: torch.Tensor):

        data_list = [Data(pos=pos.unsqueeze(0)) for pos in positions]
        batched_data = Batch.from_data_list(data_list)
        x_tensor = batched_data.pos
        x = GeometricTensor(x_tensor, self.in_type)
        #x = batched_data

        mu = self.block1_mu(x)
        mu = self.block2_mu(mu)
        mu = self.block3_mu(mu)
        mu = self.block4_mu(mu)
        mu = self.block5_mu(mu)
        
        sigma_sq = self.block1_sigma_sq(x)
        sigma_sq = self.block2_sigma_sq(sigma_sq)
        sigma_sq = self.block3_sigma_sq(sigma_sq)
        sigma_sq = self.block4_sigma_sq(sigma_sq)
        sigma_sq = self.block5_sigma_sq(sigma_sq)

        sigma_sq_tensor = torch.nn.functional.softplus(sigma_sq.tensor)
        sigma_sq = GeometricTensor(sigma_sq_tensor, sigma_sq.type)
     
        return mu, sigma_sq

    def evaluate_output_shape(self, input_shape: tuple):
        #shape = list(input_shape)
        #assert len(shape) ==2, shape
        #assert shape[1] == self.in_type.size, shape
        #shape[1] = self.out_type.size
        return #shape
