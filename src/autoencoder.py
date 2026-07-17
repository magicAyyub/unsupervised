import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.metrics import Codebook, Latent
from src.base import BaseModel
from src.helper import get_device

# Seules activations bornees dans [0,1], donc les seules compatibles avec BCELoss.
ACTIVATIONS_BORNEES_UNITE = (nn.Sigmoid, nn.Hardsigmoid)

def make_layer_sizes(
        input_dim : int
        , output_dim : int
        , n_layers : int
    ) -> list[int] :
    sizes = np.geomspace(input_dim, output_dim, num=n_layers+1)
    return [int(s) for s in sizes]

def build_layers(
        sizes : list[int]
        , activation : type[nn.Module]
        , output_activation : type[nn.Module] = None
) -> list[nn.Module] :
    # `activation` remplit les couches cachees ; `output_activation` borne le tenseur
    # de sortie (None = aucune activation terminale, la sortie reste lineaire).
    layers = []
    for i in range(len(sizes)-1) :
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2 :
            layers.append(activation())
    if output_activation is not None :
        layers.append(output_activation())
    return layers

def _forward(self, tensor) -> torch.Tensor:
    tensor = tensor.view(tensor.size(0), -1)
    return self.net(tensor)


class Encoder(nn.Module):
    def __init__(self
                 , input_dim : int
                 , encoder_layer_num : int
                 , latent_dim : int
                 , activation : type[nn.Module]
                 , latent_activation : type[nn.Module] = None
    ) :
        super().__init__()
        sizes = make_layer_sizes(input_dim, latent_dim, encoder_layer_num)
        layers = build_layers(sizes, activation, latent_activation)
        self.net = nn.Sequential(*layers)
        

    def forward(self, tensor) -> torch.Tensor:
        return _forward(self, tensor)
    
    

class Decoder(nn.Module):
    def __init__(self
                 , latent_dim : int
                 , decoder_layer_num : int
                 , output_dim : int
                 , activation : type[nn.Module]
                 , output_activation : type[nn.Module] = nn.Sigmoid
    ) :
        super().__init__()
        sizes = make_layer_sizes(latent_dim, output_dim, decoder_layer_num)
        layers = build_layers(sizes, activation, output_activation)
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
                 , encoder_activation_function : type[nn.Module]
                 , latent_activation_function : type[nn.Module]
                 , loss_function : type[nn.Module] = nn.MSELoss
                 , decoder_activation : type[nn.Module] = None
                 , output_activation_function : type[nn.Module] = nn.Sigmoid
    ):
        super().__init__()
        if issubclass(loss_function, nn.BCELoss) and output_activation_function not in ACTIVATIONS_BORNEES_UNITE :
            nom = output_activation_function.__name__ if output_activation_function else "aucune (sortie lineaire)"
            raise ValueError(
                f"BCELoss exige des sorties dans [0,1], or output_activation={nom} ne les borne pas. "
                "Utiliser output_activation=nn.Sigmoid, ou nn.BCEWithLogitsLoss avec output_activation=None."
            )
        decoder_activation = encoder_activation_function if decoder_activation is None else decoder_activation
        self.encoder = Encoder(input_dim, encoder_layer_num, latent_dim, encoder_activation_function, latent_activation_function)
        self.decoder = Decoder(latent_dim, decoder_layer_num, output_dim, decoder_activation, output_activation_function)
        self.fonction_loss = loss_function
        self.latent_activation = latent_activation_function
        self.output_activation = output_activation_function

        self.device = get_device()
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.loss_history: list[float] = []

    def fit(self
            , feature_array : np.ndarray
            , epochs : int = 20
            , batch_size : int = 32
            , learning_rate : float = 1e-3
            ) -> BaseModel:
        feature_tensor = torch.from_numpy(feature_array).float().to(self.device)
        dataset = TensorDataset(feature_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        loss_fn = self.fonction_loss()

        self.loss_history = []
        for _ in range(epochs):
            running_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstruction = self.decoder(self.encoder(batch))
                target = batch.view(batch.size(0), -1)
                loss = loss_fn(reconstruction, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch.size(0)
            self.loss_history.append(running_loss / len(dataset))
        return self

    def encode(self, feature_array: np.ndarray) -> Latent:
        with torch.no_grad() :
            feature_tensor = torch.from_numpy(feature_array).float().to(self.device)
            return Latent(
                array=self.encoder(feature_tensor).cpu().numpy()
                , nature="continuous"
            )


    def decode(self, latent_object: Latent) -> np.ndarray:
        with torch.no_grad() :
            latent_tensor = torch.from_numpy(latent_object.array).float().to(self.device)
            return self.decoder(latent_tensor).cpu().numpy()


    def get_codebook(self) -> Codebook:
        # Le codebook partagé est constitué des poids du décodeur : ce sont exactement
        # les paramètres dont le récepteur a besoin pour reconstruire une image à partir d'un code latent.
        arrays = [parameter.detach().cpu().numpy() for parameter in self.decoder.parameters()]
        return Codebook(arrays=arrays)
