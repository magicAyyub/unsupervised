import numpy as np
import torch.nn as nn
import torch

from src.metrics import Codebook, Latent
from src.base import BaseModel

def make_layer_sizes(input_dim : int
                     , output_dim : int
                     , n_layers : int) -> list[int] :
    sizes = np.geomspace(input_dim, output_dim, num=n_layers+1)
    return [int(s) for s in sizes]

def build_layers(
        sizes : list[int]
        , activation : type[nn.Module]
) -> nn.Sequential :
    layers = []
    for i in range(len(sizes)-1) :
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2 :
            layers.append(activation())
    return layers

def _forward(self, tensor) -> torch.Tensor:
    tensor = tensor.view(tensor.size(0), -1) //pas compris
    return self.net(tensor)


class Encoder(nn.Module):
    def __init__(self
                 , input_dim : int
                 , encoder_layer_num : int
                 , latent_dim : int
                 , activation : type[nn.Module]
    ) :
        super().__init__()
        sizes = make_layer_sizes(input_dim, latent_dim, encoder_layer_num)
        layers = build_layers(sizes, activation)
        self.net = nn.Sequential(*layers)
        

    def forward(self, tensor) -> torch.Tensor:
        return _forward(self, tensor)
    
    

class Decoder(nn.Module):
    def __init__(self
                 , latent_dim : int
                 , decoder_layer_num : int
                 , output_dim : int
                 , activation : type[nn.Module]
    ) :
        super().__init__()
        sizes = make_layer_sizes(latent_dim, output_dim, decoder_layer_num)
        layers = build_layers(sizes, activation)
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    
    def forward(self, tensor) -> torch.Tensor:
        return _forward(self, tensor)

class AutoEncoder(BaseModel):
    def __init__(self
                 , input_dim : int
                 , output_dim : int
                 , latent_dim : int
                 , encoder_layer_num : int
                 , decoder_layer_num : int
                 , encoder_activation : type[nn.Module]
                 , decoder_activation : type[nn.Module] = None
    ):
        super().__init__()
        decoder_activation = encoder_activation if decoder_activation is None else decoder_activation
        self.encoder = Encoder(input_dim, encoder_layer_num, latent_dim, encoder_activation)
        self.decoder = Decoder(latent_dim, decoder_layer_num, output_dim, decoder_activation)

    def fit(self, X: np.ndarray) -> BaseModel:
        pass

    
    def encode(self, input_tensor: np.ndarray) -> Latent:
        with torch.no_grad() :
            input_tensor = torch.from_numpy(input_tensor).float()
            return Latent(
                array=self.encoder(input_tensor).cpu().numpy()
                , nature="continuous"
            )
        

    def decode(self, latent_object: Latent) -> np.ndarray:
        with torch.no_grad() :
            latent_tensor = torch.from_numpy(latent_object.array).float()
            return self.decoder(latent_tensor).cpu().numpy()
    

    
    def get_codebook(self) -> Codebook:
        
        pass
