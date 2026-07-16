from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.helper import get_device
from src.metrics import Codebook, Latent


def build_mlp(
    sizes: list[int],
    hidden_activation: Callable[[], nn.Module],
    final_activation: Callable[[], nn.Module] | None,
    dropout: float = 0.0,
) -> nn.Sequential:
    # sizes = [in, h1, ..., out] -> Linear enchaines, activation entre les couches cachees seulement
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(hidden_activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class Generator(nn.Module):
    # forward : z (B, latent_dim) -> x_fake (B, output_dim), valeurs dans [-1, 1] via Tanh
    def __init__(self, latent_dim: int, hidden_sizes: list[int], output_dim: int):
        super().__init__()
        self.net = build_mlp(
            [latent_dim, *hidden_sizes, output_dim], lambda: nn.LeakyReLU(0.2), nn.Tanh
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class Discriminator(nn.Module):
    # forward : x (B, input_dim) -> logit (B, 1), score non borne : > 0 penche "reelle", < 0 "fausse"
    # LeakyReLU plutot que ReLU : evite de tuer le gradient qui doit remonter jusqu'au generateur
    # Dropout : handicape volontairement D, sinon il gagne trop vite et G n'apprend plus rien
    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float = 0.3):
        super().__init__()
        self.net = build_mlp(
            [input_dim, *hidden_sizes, 1], lambda: nn.LeakyReLU(0.2), None, dropout=dropout
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.net(sample)

    def features(self, sample: torch.Tensor) -> torch.Tensor:
        # Tout le reseau sauf le dernier Linear : x (B, input_dim) -> (B, dernier_hidden)
        # Ce sont les descripteurs que D s'est construits pour juger du vrai/faux.
        return self.net[:-1](sample)


class GAN:
    """
    Minimal MLP GAN (Goodfellow 2014) with the non-saturating generator loss.

    Deliberately does not implement BaseModel: a GAN has no encode(x). The closest thing,
    invert(), is a per-sample gradient descent, not a forward pass, so advertising it as an
    encoder would misrepresent its cost. The three syllabus axes map as follows:
      - generation: generate(), the model's actual job.
      - compression: invert() then decode(), workable but expensive and lossy.
      - projection: extract_features(), reusing the discriminator as a feature extractor.

    Public API works in [0, 1] like the rest of the project pipeline; the [-1, 1]
    rescaling required by the generator's Tanh is handled internally.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 100,
        generator_hidden: list[int] | None = None,
        discriminator_hidden: list[int] | None = None,
        discriminator_dropout: float = 0.3,
    ):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.device = get_device()

        # Le generateur va du petit vers le grand, le discriminateur l'inverse
        self.generator = Generator(
            latent_dim, generator_hidden or [256, 512, 1024], data_dim
        ).to(self.device)
        self.discriminator = Discriminator(
            data_dim, discriminator_hidden or [512, 256], dropout=discriminator_dropout
        ).to(self.device)

        self.loss_history: dict[str, list[float]] = {"generator": [], "discriminator": []}

    def sample_noise(self, n_samples: int) -> torch.Tensor:
        # -> (n_samples, latent_dim), gaussienne centree reduite
        return torch.randn(n_samples, self.latent_dim, device=self.device)

    def fit(
        self,
        feature_array: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 2e-4,
    ) -> "GAN":
        """
        Trains both networks in the adversarial min-max game.

        Args:
            feature_array: shape (N, data_dim), float in [0, 1].

        Returns:
            self, with loss_history filled with one mean loss per epoch and per network.
        """
        # [0, 1] -> [-1, 1] pour coller a la sortie Tanh du generateur
        feature_tensor = torch.from_numpy(feature_array).float() * 2.0 - 1.0
        loader = DataLoader(
            TensorDataset(feature_tensor), batch_size=batch_size, shuffle=True, drop_last=True
        )

        # betas=(0.5, 0.999) : reglage standard des GAN, un momentum a 0.9 rend le jeu instable
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
        loss_fn = nn.BCEWithLogitsLoss()

        self.loss_history = {"generator": [], "discriminator": []}
        for _ in range(epochs):
            generator_running, discriminator_running, n_batches = 0.0, 0.0, 0

            for (real_batch,) in loader:
                real_batch = real_batch.to(self.device)
                current_size = real_batch.size(0)
                real_targets = torch.ones(current_size, 1, device=self.device)
                fake_targets = torch.zeros(current_size, 1, device=self.device)

                # 1. Discriminateur : reconnaitre les vraies comme vraies et les fausses comme fausses
                fake_batch = self.generator(self.sample_noise(current_size))
                discriminator_loss = loss_fn(
                    self.discriminator(real_batch), real_targets
                ) + loss_fn(
                    # detach : on ne remonte pas le gradient dans le generateur a cette etape
                    self.discriminator(fake_batch.detach()), fake_targets
                )
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # 2. Generateur : faire passer ses fausses images pour vraies (loss non saturante)
                generator_loss = loss_fn(self.discriminator(fake_batch), real_targets)
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

                discriminator_running += discriminator_loss.item()
                generator_running += generator_loss.item()
                n_batches += 1

            self.loss_history["discriminator"].append(discriminator_running / n_batches)
            self.loss_history["generator"].append(generator_running / n_batches)
        return self

    def generate(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        """
        Samples new synthetic data from the latent prior.

        Returns:
            shape (n_samples, data_dim), float32 back in [0, 1].
        """
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            fake = self.generator(self.sample_noise(n_samples))
        # [-1, 1] -> [0, 1] : on rend au reste du projet sa convention d'affichage
        return ((fake.cpu().numpy() + 1.0) / 2.0).astype(np.float32)

    def invert(
        self,
        feature_array: np.ndarray,
        steps: int = 300,
        learning_rate: float = 0.05,
        seed: int | None = None,
    ) -> Latent:
        """
        Finds, for each real sample, the latent code whose generated image is closest to it.

        This is NOT an encoder: there is no x -> z function in a GAN. We freeze G and run a
        gradient descent on z itself, which costs `steps` iterations per call. It is the only
        way to place a GAN on the compression axis, and its cost is precisely the point to
        discuss against the AutoEncoder's single forward pass.

        Args:
            feature_array: shape (N, data_dim), float in [0, 1].

        Returns:
            Latent of shape (N, latent_dim), nature "continuous".
        """
        if seed is not None:
            torch.manual_seed(seed)
        target = torch.from_numpy(feature_array).float().to(self.device) * 2.0 - 1.0

        # z est la seule variable optimisee ; les poids de G restent figes
        latent_code = self.sample_noise(len(feature_array)).requires_grad_(True)
        optimizer = torch.optim.Adam([latent_code], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.generator(latent_code), target)
            loss.backward()
            optimizer.step()

        self.generator.zero_grad(set_to_none=True)
        return Latent(array=latent_code.detach().cpu().numpy(), nature="continuous")

    def decode(self, latent_object: Latent) -> np.ndarray:
        """
        Rebuilds images from latent codes, typically those returned by invert().

        Returns:
            shape (n_samples, data_dim), float32 in [0, 1].
        """
        with torch.no_grad():
            latent_tensor = torch.from_numpy(latent_object.array).float().to(self.device)
            images = self.generator(latent_tensor)
        return ((images.cpu().numpy() + 1.0) / 2.0).astype(np.float32)

    def extract_features(self, feature_array: np.ndarray) -> np.ndarray:
        """
        Projects real samples into the discriminator's learned feature space.

        Returns:
            shape (n_samples, last_discriminator_hidden), float32.
        """
        # eval() : on coupe le Dropout, sinon les descripteurs seraient bruites au hasard
        self.discriminator.eval()
        with torch.no_grad():
            samples = torch.from_numpy(feature_array).float().to(self.device) * 2.0 - 1.0
            features = self.discriminator.features(samples)
        self.discriminator.train()
        return features.cpu().numpy().astype(np.float32)

    def get_codebook(self) -> Codebook:
        # Seuls les poids du generateur servent a produire des images : le discriminateur
        # est un echafaudage d'entrainement, il est jete a la fin.
        arrays = [parameter.detach().cpu().numpy() for parameter in self.generator.parameters()]
        return Codebook(arrays=arrays)
