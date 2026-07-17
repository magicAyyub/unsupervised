"""
AutoEncodeur Variationnel (VAE).

Difference avec l'AutoEncoder classique: l'encodeur ne rend plus un POINT mais une
DISTRIBUTION q(z|x) = N(mu(x), diag(exp(logvar(x)))), et la perte ajoute un terme KL
qui pousse cette distribution vers le prior N(0, I).

C'est exactement ce qui manquait a l'AutoEncoder: son espace latent n'obeissant a
aucune loi connue, generer imposait d'AJUSTER apres coup une gaussienne sur les codes
observes (src.viz.sample_gaussian_latent), avec une bonne part des tirages hors de la
zone ou le decodeur a ete entraine. Le VAE, lui, connait la loi de son latent par
construction: echantillonner N(0, I) suffit (sample_prior).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.autoencoder import AutoEncoder, make_layer_sizes, build_layers
from src.metrics import Latent

# Bornes sur log(sigma^2). exp() deborde en float32 au-dela de ~88, et rien ne contraint
# logvar quand beta=0. min=-20 -> sigma ~ 4.5e-5 ; max=10 -> sigma ~ 148.
LOGVAR_MIN, LOGVAR_MAX = -20.0, 10.0


class VariationalEncoder(nn.Module):
    """
    Encodeur a deux tetes: q(z|x) = N(mu(x), diag(exp(logvar(x)))).

    Le tronc reprend make_layer_sizes / build_layers de l'AutoEncoder mais s'arrete AVANT
    la derniere couche: celle-ci est dedoublee en deux tetes lineaires (mu, logvar) qui
    partagent le tronc. La profondeur lineaire totale reste donc encoder_layer_num,
    exactement comme l'AutoEncoder, ce qui rend les deux modeles comparables.
    """

    def __init__(self
                 , input_dim : int
                 , encoder_layer_num : int
                 , latent_dim : int
                 , activation : type[nn.Module]
    ) :
        super().__init__()
        sizes = make_layer_sizes(input_dim, latent_dim, encoder_layer_num)
        trunk_sizes = sizes[:-1]

        if len(trunk_sizes) == 1 :
            # encoder_layer_num=1 -> sizes=[input_dim, latent_dim] -> trunk_sizes=[input_dim].
            # build_layers([784], act, act) ne renverrait qu'une activation NUE posee sur
            # l'entree brute, sans aucun Linear (silencieux sur MNIST ou ReLU est l'identite
            # sur [0,1], mais faux avec Tanh). Le tronc est donc l'identite et les deux tetes
            # lisent directement l'entree.
            self.net = nn.Sequential()
        else :
            # output_activation=activation: le tronc se TERMINE par une activation, sinon les
            # tetes (des Linear) suivraient un Linear nu, soit deux couches lineaires
            # consecutives, equivalentes a une seule.
            self.net = nn.Sequential(*build_layers(trunk_sizes, activation, output_activation=activation))

        self.mu_head = nn.Linear(trunk_sizes[-1], latent_dim)
        self.logvar_head = nn.Linear(trunk_sizes[-1], latent_dim)

    def forward(self, tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ATTENTION: renvoie un TUPLE, pas un tenseur, contrairement a Encoder.
        hidden = self.net(tensor.view(tensor.size(0), -1))
        logvar = torch.clamp(self.logvar_head(hidden), LOGVAR_MIN, LOGVAR_MAX)
        return self.mu_head(hidden), logvar


class VariationalAutoEncoder(AutoEncoder):
    """
    VAE construit sur AutoEncoder: le decodeur, le codebook, le garde-fou BCE/sortie et le
    device sont ceux du parent. Seules changent la facon de produire z (une distribution
    au lieu d'un point) et la perte (reconstruction + beta * KL).
    """

    def __init__(self
                 , input_dim : int
                 , output_dim : int
                 , latent_dim : int
                 , encoder_layer_num : int
                 , decoder_layer_num : int
                 , encoder_activation_function : type[nn.Module]
                 , loss_function : type[nn.Module] = nn.BCELoss
                 , beta : float = 1.0
                 , decoder_activation : type[nn.Module] = None
                 , output_activation_function : type[nn.Module] = nn.Sigmoid
    ):
        # latent_activation_function=None et volontairement NON expose: borner mu n'a pas de
        # sens pour un VAE. mu est un parametre de distribution, pas un code: Sigmoid(mu)
        # dans (0,1) ne pourra jamais coller a N(0,I), donc le KL aurait un plancher
        # structurel non nul. Voir le notebook 05_vae.
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, latent_dim=latent_dim,
            encoder_layer_num=encoder_layer_num, decoder_layer_num=decoder_layer_num,
            encoder_activation_function=encoder_activation_function,
            latent_activation_function=None,
            loss_function=loss_function,
            decoder_activation=decoder_activation,
            output_activation_function=output_activation_function,
        )
        # Le parent a construit un Encoder deterministe: on le remplace par la version a deux
        # tetes. Le gaspillage est negligeable et evite de modifier autoencoder.py.
        self.encoder = VariationalEncoder(input_dim, encoder_layer_num, latent_dim,
                                          encoder_activation_function)
        self.encoder.to(self.device)

        self.latent_dim = latent_dim   # le parent ne le stocke pas, sample_prior en a besoin
        self.beta = beta
        self.recon_history: list[float] = []
        self.kl_history: list[float] = []

    @staticmethod
    def reparameterize(mu : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor:
        # z = mu + sigma * eps. Le hasard est isole dans eps, donc le gradient remonte
        # jusqu'a mu et logvar: c'est l'astuce de reparametrisation.
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    @staticmethod
    def kl_divergence_per_dim(mu : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor:
        # KL(N(mu, sigma^2) || N(0,1)), dimension latente par dimension latente, moyennee sur
        # le batch. .sum() redonne le KL total: somme et moyenne commutent, donc une seule
        # fonction sert a la fois a fit() et au diagnostic de collapse.
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)

    def fit(self
            , feature_array : np.ndarray
            , epochs : int = 20
            , batch_size : int = 32
            , learning_rate : float = 1e-3
            ) -> "VariationalAutoEncoder":
        feature_tensor = torch.from_numpy(feature_array).float().to(self.device)
        dataset = TensorDataset(feature_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        # reduction="sum" est LE point critique, et c'est pourquoi fit() est entierement
        # reecrit au lieu d'appeler super().fit(). Le KL somme sur les dimensions latentes;
        # la reconstruction doit donc sommer sur les dimensions de l'image pour lui etre
        # homogene. Avec le defaut reduction="mean", la reconstruction serait 784x plus
        # petite: cela revient EXACTEMENT a beta=784, le KL ecrase tout et le posterior
        # s'effondre (mu -> 0, sigma -> 1, KL -> 0).
        recon_loss_fn = self.fonction_loss(reduction="sum")

        self.loss_history, self.recon_history, self.kl_history = [], [], []
        for _ in range(epochs):
            running_loss = running_recon = running_kl = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                mu, logvar = self.encoder(batch)
                z = self.reparameterize(mu, logvar)
                reconstruction = self.decoder(z)
                target = batch.view(batch.size(0), -1)

                recon = recon_loss_fn(reconstruction, target) / batch.size(0)
                kl = self.kl_divergence_per_dim(mu, logvar).sum()
                loss = recon + self.beta * kl

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch.size(0)
                running_recon += recon.item() * batch.size(0)
                running_kl += kl.item() * batch.size(0)

            n_samples = len(dataset)
            self.loss_history.append(running_loss / n_samples)
            self.recon_history.append(running_recon / n_samples)
            self.kl_history.append(running_kl / n_samples)
        return self

    def encode_distribution(self, feature_array : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Les DEUX parametres du posterior: encode() ne rend que mu, or les diagnostics
        de collapse et de sigma ont besoin de logvar."""
        with torch.no_grad() :
            feature_tensor = torch.from_numpy(feature_array).float().to(self.device)
            mu, logvar = self.encoder(feature_tensor)
            return mu.cpu().numpy(), logvar.cpu().numpy()

    def encode(self, feature_array : np.ndarray) -> Latent:
        # On renvoie mu, PAS un tirage: c'est le code deterministe que l'emetteur
        # transmettrait, et cela garde compression_report reproductible d'un appel a l'autre.
        mu, _ = self.encode_distribution(feature_array)
        return Latent(array=mu, nature="continuous")

    def sample_latent(self, feature_array : np.ndarray, seed : int = None) -> Latent:
        """Tirage z ~ q(z|x): montre le bruit du posterior autour de chaque image."""
        if seed is not None :
            torch.manual_seed(seed)
        with torch.no_grad() :
            feature_tensor = torch.from_numpy(feature_array).float().to(self.device)
            mu, logvar = self.encoder(feature_tensor)
            return Latent(array=self.reparameterize(mu, logvar).cpu().numpy(), nature="continuous")

    def sample_prior(self, n_samples : int, seed : int = 0) -> Latent:
        """
        LE test du VAE: echantillonner le prior N(0, I) sans jamais regarder les donnees.
        A comparer a src.viz.sample_gaussian_latent, qui doit d'abord AJUSTER une gaussienne
        sur les codes: l'aveu que l'espace latent de l'AutoEncoder n'est pas connu d'avance.
        """
        rng = np.random.default_rng(seed)
        codes = rng.standard_normal((n_samples, self.latent_dim)).astype(np.float32)
        return Latent(array=codes, nature="continuous")
