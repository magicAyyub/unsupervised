import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.autoencoder import make_layer_sizes, build_layers, Decoder
from src.base import BaseModel
from src.helper import get_device
from src.metrics import Codebook, Latent


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_layers, activation):
        super().__init__()
        sizes = make_layer_sizes(input_dim, latent_dim, n_layers)
        hidden_dim = sizes[-2]

        if n_layers == 1:
            self.trunk = nn.Sequential()
        else:
            self.trunk = nn.Sequential(
                *build_layers(sizes[:-1], activation=activation, output_activation=None))

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = self.trunk(x)
        # clamp: exp() deborde en float32 au-dela de ~88, et rien ne borne logvar si beta=0.
        return self.mu_head(hidden), torch.clamp(self.logvar_head(hidden), -20, 10)


class VariationalAutoEncoder(BaseModel):

    def __init__(self, input_dim, output_dim, latent_dim, encoder_layer_num, decoder_layer_num,
                 encoder_activation_function, loss_function=nn.BCELoss, beta=1.0,
                 output_activation_function=nn.Sigmoid):
        if loss_function is nn.BCELoss and output_activation_function is not nn.Sigmoid:
            raise ValueError("BCELoss exige des sorties dans [0,1]: garder output_activation=nn.Sigmoid")

        self.device = get_device()
        self.encoder = VariationalEncoder(input_dim, latent_dim, encoder_layer_num,
                                          encoder_activation_function).to(self.device)
        self.decoder = Decoder(latent_dim, decoder_layer_num, output_dim,
                               encoder_activation_function, output_activation_function).to(self.device)

        self.latent_dim = latent_dim
        self.beta = beta
        self.fonction_loss = loss_function
        self.output_activation = output_activation_function
        self.loss_history = []
        self.recon_history = []
        self.kl_history = []

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps: le hasard est isole dans eps, donc le gradient passe.
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def kl_per_dim(self, mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)

    def fit(self, feature_array, epochs=20, batch_size=32, learning_rate=1e-3):
        feature_tensor = torch.from_numpy(feature_array).float().to(self.device)
        loader = DataLoader(TensorDataset(feature_tensor), batch_size=batch_size, shuffle=True)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        # reduction="sum" est indispensable: le KL somme sur les dimensions latentes, donc la
        # reconstruction doit sommer sur les 784 pixels. Avec le defaut "mean" elle serait 784x
        # trop petite, ce qui revient exactement a beta=784: le KL ecrase tout et le posterior
        # s'effondre (mu -> 0, sigma -> 1, KL -> 0).
        recon_loss_fn = self.fonction_loss(reduction="sum")

        self.loss_history, self.recon_history, self.kl_history = [], [], []
        for _ in range(epochs):
            total_loss = total_recon = total_kl = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                mu, logvar = self.encoder(batch)
                reconstruction = self.decoder(self.reparameterize(mu, logvar))

                recon = recon_loss_fn(reconstruction, batch) / batch.size(0)
                kl = self.kl_per_dim(mu, logvar).sum()
                loss = recon + self.beta * kl

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.size(0)
                total_recon += recon.item() * batch.size(0)
                total_kl += kl.item() * batch.size(0)

            n = len(feature_tensor)
            self.loss_history.append(total_loss / n)
            self.recon_history.append(total_recon / n)
            self.kl_history.append(total_kl / n)
        return self

    def encode_distribution(self, feature_array):
        """Les deux parametres du posterior: encode() ne rend que mu."""
        with torch.no_grad():
            feature_tensor = torch.from_numpy(feature_array).float().to(self.device)
            mu, logvar = self.encoder(feature_tensor)
            return mu.cpu().numpy(), logvar.cpu().numpy()

    def encode(self, feature_array):
        # On renvoie mu, pas un tirage: c'est le code que l'emetteur transmet, et cela garde
        # compression_report reproductible.
        mu, _ = self.encode_distribution(feature_array)
        return Latent(array=mu, nature="continuous")

    def decode(self, latent_object):
        with torch.no_grad():
            latent_tensor = torch.from_numpy(latent_object.array).float().to(self.device)
            return self.decoder(latent_tensor).cpu().numpy()

    def sample_prior(self, n_samples, seed=0):
        """Echantillonne N(0, I) sans regarder les donnees: c'est ce que le KL rend possible."""
        rng = np.random.default_rng(seed)
        codes = rng.standard_normal((n_samples, self.latent_dim)).astype(np.float32)
        return Latent(array=codes, nature="continuous")

    def get_codebook(self):
        arrays = [parameter.detach().cpu().numpy() for parameter in self.decoder.parameters()]
        return Codebook(arrays=arrays)
