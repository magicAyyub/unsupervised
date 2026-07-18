from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.helper import get_device


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    # Gel / degel explicite des poids d'un reseau. Le gradient continue de traverser le module
    # pour atteindre l'autre reseau, il n'est simplement plus accumule sur ces poids-la.
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


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
    Vanilla MLP GAN (Goodfellow 2014), binary cross-entropy, non-saturating generator loss.

    Does not implement BaseModel: a GAN has no encode(x), it is not an encoder/decoder.
    Two tasks only:
      - generation: generate().
      - projection: extract_features(), reusing the discriminator as a feature extractor.

    Public API works in [0, 1]; the [-1, 1] rescaling required by the generator's Tanh is
    handled internally.
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
        self.metric_history: dict[str, list[float]] = {
            "discriminator_accuracy": [],
            "generator_variance": [],
        }

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

        The generator minimises -log D(G(z)) (non-saturating form) rather than the original
        log(1 - D(G(z))): once D confidently rejects a fake, the latter's derivative goes to 0
        and the generator stops learning exactly when it needs the signal most.

        Args:
            feature_array: shape (N, data_dim), float in [0, 1].

        Returns:
            self, with loss_history and metric_history filled with one value per epoch.
        """
        # [0, 1] -> [-1, 1] pour coller a la sortie Tanh du generateur
        feature_tensor = torch.from_numpy(feature_array).float() * 2.0 - 1.0
        # shuffle : chaque batch doit melanger les classes. Des batchs tries par classe donneraient
        # a D une statistique de lot a exploiter au lieu de juger chaque image.
        # drop_last : un dernier batch incomplet fausserait les moyennes de fin d'epoch.
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
        # BCEWithLogitsLoss plutot que Sigmoid + BCELoss : le log-sum-exp est calcule de facon
        # stable, un logit tres confiant ne produit pas un gradient nul par arrondi.
        loss_fn = nn.BCEWithLogitsLoss()

        self.loss_history = {"generator": [], "discriminator": []}
        self.metric_history = {"discriminator_accuracy": [], "generator_variance": []}
        for _ in range(epochs):
            generator_running, discriminator_running, n_batches = 0.0, 0.0, 0
            accuracy_running, variance_running = 0.0, 0.0

            for (real_batch,) in loader:
                real_batch = real_batch.to(self.device)
                current_size = real_batch.size(0)
                real_targets = torch.ones(current_size, 1, device=self.device)
                fake_targets = torch.zeros(current_size, 1, device=self.device)

                # 1. Discriminateur : reconnaitre les vraies comme vraies et les fausses comme fausses.
                #    G est gele par le detach ci-dessous : le gradient de cette etape ne doit pas
                #    remonter jusqu'a ses poids.
                set_requires_grad(self.discriminator, True)
                fake_batch = self.generator(self.sample_noise(current_size))
                real_logits = self.discriminator(real_batch)
                fake_logits = self.discriminator(fake_batch.detach())
                discriminator_loss = loss_fn(real_logits, real_targets) + loss_fn(
                    fake_logits, fake_targets
                )
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # 2. Generateur : faire passer ses fausses images pour vraies.
                #    Symetrique de l'etape 1 : on gele D, le gradient le traverse pour atteindre G
                #    mais ne modifie pas ses poids.
                set_requires_grad(self.discriminator, False)
                generator_loss = loss_fn(self.discriminator(fake_batch), real_targets)
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

                # Metriques d'equilibre : accuracy de D sur le lot vraies + fausses (0.5 = il ne
                # fait pas mieux que le hasard), et variance des images produites (mesure de diversite)
                with torch.no_grad():
                    correct = (real_logits > 0).sum() + (fake_logits <= 0).sum()
                    accuracy_running += (correct / (2 * current_size)).item()
                    variance_running += fake_batch.var(dim=0).mean().item()

                discriminator_running += discriminator_loss.item()
                generator_running += generator_loss.item()
                n_batches += 1

            self.loss_history["discriminator"].append(discriminator_running / n_batches)
            self.loss_history["generator"].append(generator_running / n_batches)
            self.metric_history["discriminator_accuracy"].append(accuracy_running / n_batches)
            self.metric_history["generator_variance"].append(variance_running / n_batches)

        set_requires_grad(self.discriminator, True)
        return self

    def generate(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        """
        Samples new synthetic data from the latent prior.

        Returns:
            shape (n_samples, data_dim), float32 back in [0, 1].
        """
        if seed is not None:
            torch.manual_seed(seed)
        # eval() : impose les statistiques courantes a une eventuelle BatchNorm. Sans cela G(z)
        # dependrait des autres elements du lot, donc z ne determinerait plus son image.
        # Sans BatchNorm (le G du MLP) c'est un no-op.
        self.generator.eval()
        with torch.no_grad():
            fake = self.generator(self.sample_noise(n_samples))
        self.generator.train()
        # [-1, 1] -> [0, 1] : on rend au reste du projet sa convention d'affichage
        return ((fake.cpu().numpy() + 1.0) / 2.0).astype(np.float32)

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
