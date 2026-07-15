import numpy as np
import torch
import torch.nn as nn
from src.base import BaseModel
from src.metrics import Codebook, Latent
from src.helper import get_device


class AutoEncoderNet(nn.Module):
    """
    The actual PyTorch module: a simple encoder/decoder pair.
    Kept separate from the AutoEncoder wrapper below, which handles
    the BaseModel interface (numpy in/out, Codebook, Latent).
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Encoder: Linear -> Tanh -> Linear
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder: Linear -> Tanh -> Linear
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x: input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: reconstructed input and latent representation
        """
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z


class AutoEncoder(BaseModel):
    """
    Custom implementation of a simple AutoEncoder.

    Latent space: continuous, a vector of latent_dim real coordinates per sample.
    Codebook: the decoder's weights and biases (needed to reconstruct from latent).
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128,
                 epochs: int = 50, lr: float = 1e-3, batch_size: int = 256):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = get_device()
        self.net: AutoEncoderNet | None = None
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray) -> "AutoEncoder":
        """
        Trains the encoder/decoder on the data X.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            self
        """
        self.net = AutoEncoderNet(self.input_dim, self.latent_dim, self.hidden_dim).to(self.device)
        
        # Conversion du dataset complet en tenseur PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Optimiseur Adam et fonction de perte Mean Squared Error (MSE)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Entraînement en mini-batches pour MNIST (les données chargées tiennent en mémoire GPU/CPU)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.loss_history = []
        self.net.train()
        for epoch in range(self.epochs):
            epoch_losses = []
            for batch in loader:
                batch_x = batch[0]
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                x_recon, _ = self.net(batch_x)
                
                # Calcul de perte
                loss = criterion(x_recon, batch_x)
                
                # Rétropropagation et pas d'optimisation
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Stockage de la loss moyenne de l'époque pour tracé de courbe de convergence
            self.loss_history.append(float(np.mean(epoch_losses)))

        return self

    def encode(self, X: np.ndarray) -> Latent:
        """
        Projects X into the latent space.

        Args:
            X: shape (n_samples, n_features)

        Returns:
            latent: A Latent object containing the projected coordinates.
        """
        if self.net is None:
            raise ValueError("The model must be fitted before encoding.")

        self.net.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Pas de calcul de gradients en inférence pour économiser la mémoire
        with torch.no_grad():
            latent_coords = self.net.encoder(X_tensor)

        # Conversion et forçage en float32 NumPy
        latent_np = latent_coords.detach().cpu().numpy().astype(np.float32)
        return Latent(array=latent_np, nature="continuous")

    def decode(self, latent: Latent) -> np.ndarray:
        """
        Reconstructs an approximation of the data from the latent coordinates.

        Args:
            latent: A Latent object.

        Returns:
            X_reconstructed: shape (n_samples, n_features)
        """
        if self.net is None:
            raise ValueError("The model must be fitted before decoding.")

        self.net.eval()
        latent_tensor = torch.tensor(latent.array, dtype=torch.float32).to(self.device)

        # Passage dans le décodeur uniquement
        with torch.no_grad():
            x_reconstructed = self.net.decoder(latent_tensor)

        return x_reconstructed.detach().cpu().numpy().astype(np.float32)

    def get_codebook(self) -> Codebook:
        """
        Returns the standardized codebook of the model (decoder parameters).

        Returns:
            codebook: A Codebook object.
        """
        if self.net is None:
            raise ValueError("The model must be fitted before retrieving the codebook.")

        # Extraction des paramètres du décodeur (poids et biais de chaque couche nn.Linear)
        # PyTorch stocke déjà ces paramètres en float32 par défaut.
        # Nous appelons explicitement .astype(np.float32) pour garantir la neutralité
        # vis-à-vis des frameworks et la cohérence avec le PCA.
        decoder_arrays = []
        for param in self.net.decoder.parameters():
            param_np = param.detach().cpu().numpy().astype(np.float32)
            decoder_arrays.append(param_np)

        return Codebook(arrays=decoder_arrays)