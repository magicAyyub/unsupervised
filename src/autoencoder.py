import numpy as np
import torch.nn as nn
import torch

from src.metrics import Codebook, Latent
from src.base import BaseModel

def make_layer_sizes(input_dim : int
                     , latent_dim : int
                     , n_layers : int) -> list[int] :
    sizes = np.geomspacee(input_dim, latent_dim, num=n_layers+1)
    return [int(s) for s in sizes]

class Encoder(nn.Module):
    def __init__(self
                 , input_dim : int
                 , encoder_layer_num : int
                 , latent_dim : int
    ) :
        super().__init__()
        sizes = make_layer_sizes(input_dim, latent_dim, encoder_layer_num)
        layers = []
        for i in range(len(sizes)-1) :
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2 :
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    
    def forward(self, tensor) -> torch.Tensor:
        tensor = tensor.view(tensor.size(0), -1) //pas compris
        return x
    pass

class Decoder(nn.Module):
    pass

class AutoEncoder(BaseModel):
    def __init__(self
                 , input_dim : int
                 , output_dim : int
                 , latent_dim : int
                 , encoder_layer_num : int
                 , decoder_layer_num : int
    ):
        self.encoder = Encoder(input_dim, encoder_layer_num, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_layer_num, output_dim)

    def fit(self, X: np.ndarray) -> BaseModel:
        pass

    
    def encode(self, X: np.ndarray) -> Latent:
        
        pass

    def decode(self, latent: Latent) -> np.ndarray:
       
        pass

    
    def get_codebook(self) -> Codebook:
        
        pass
