from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.helper import get_device
from src.metrics import Codebook, Latent


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


class ConvGenerator(nn.Module):
    # forward : z (B, latent_dim) -> x_fake (B, C*H*W) aplati, valeurs dans [-1, 1] via Tanh
    # Projection du bruit en carte 4x4 puis triplement de la resolution : 4 -> 8 -> 16 -> 32.
    # BatchNorm entre les couches : sans elle les activations derivent et l'entrainement casse.
    def __init__(self, latent_dim: int, image_shape: tuple[int, int, int], base: int = 64):
        super().__init__()
        channels, _, _ = image_shape
        self.base = base
        self.project = nn.Linear(latent_dim, base * 4 * 4 * 4)
        self.net = nn.Sequential(
            nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, channels, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        feature_map = self.project(noise).view(-1, self.base * 4, 4, 4)
        return self.net(feature_map).flatten(start_dim=1)


class ConvCritic(nn.Module):
    # forward : x (B, C*H*W) -> score (B, 1) non borne. Ce n'est pas un classifieur : le score
    # n'est pas une probabilite, seul l'ecart moyen entre vraies et fausses a un sens.
    # LayerNorm et non BatchNorm : la penalite de gradient est definie echantillon par echantillon,
    # une BatchNorm melangerait le lot et rendrait la contrainte invalide.
    def __init__(self, image_shape: tuple[int, int, int], base: int = 64):
        super().__init__()
        channels, _, _ = image_shape
        self.image_shape = image_shape
        self.body = nn.Sequential(
            nn.Conv2d(channels, base, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.LayerNorm([base * 2, 8, 8]), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.LayerNorm([base * 4, 4, 4]), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
        )
        self.head = nn.Linear(base * 4 * 4 * 4, 1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(sample))

    def features(self, sample: torch.Tensor) -> torch.Tensor:
        # Tout le corps convolutif sauf la tete lineaire : x (B, C*H*W) -> (B, base*4*4*4)
        return self.body(sample.view(-1, *self.image_shape))


class GAN:
    """
    Vanilla MLP GAN (Goodfellow 2014), binary cross-entropy, non-saturating generator loss.

    Does not implement BaseModel: a GAN has no encode(x). The closest thing, invert(), is a
    per-sample gradient descent rather than a forward pass, so exposing it as an encoder would
    misrepresent its cost. Methods per task:
      - generation: generate().
      - compression: invert() then decode().
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
        self.generator.eval()
        latent_code = self.sample_noise(len(feature_array)).requires_grad_(True)
        optimizer = torch.optim.Adam([latent_code], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.generator(latent_code), target)
            loss.backward()
            optimizer.step()

        self.generator.zero_grad(set_to_none=True)
        self.generator.train()
        return Latent(array=latent_code.detach().cpu().numpy(), nature="continuous")

    def decode(self, latent_object: Latent) -> np.ndarray:
        """
        Rebuilds images from latent codes, typically those returned by invert().

        Returns:
            shape (n_samples, data_dim), float32 in [0, 1].
        """
        self.generator.eval()
        with torch.no_grad():
            latent_tensor = torch.from_numpy(latent_object.array).float().to(self.device)
            images = self.generator(latent_tensor)
        self.generator.train()
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
        # state_dict() et non parameters() : les buffers d'une BatchNorm (running_mean / running_var)
        # ne sont pas des parametres appris mais sont indispensables pour decoder, donc ils comptent
        # dans le poids du codebook. Sur un G sans BatchNorm les deux sont equivalents.
        arrays = [tensor.detach().cpu().numpy() for tensor in self.generator.state_dict().values()]
        return Codebook(arrays=arrays)


class WGANGP(GAN):
    """
    Convolutional WGAN with gradient penalty (Gulrajani et al., 2017).

    Same public API as GAN: only __init__ and fit change. Two departures from the vanilla GAN,
    each answering a pathology measured on the shapes dataset (see journal_dev_gan.md):
      - convolutions: a dense discriminator cannot see an edge, so G never learns geometry.
      - Wasserstein loss + gradient penalty: with BCE the convolutional discriminator saturates
        (accuracy 0.99) and starves G, and handicapping it with dropout destroys its edge
        detectors. The critic gives a usable gradient whatever the gap between the two
        distributions.

    Args:
        image_shape: (channels, height, width). data_dim is derived from it.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        latent_dim: int = 100,
        base_channels: int = 64,
    ):
        super().__init__(data_dim=int(np.prod(image_shape)), latent_dim=latent_dim)
        self.image_shape = image_shape
        self.generator = ConvGenerator(latent_dim, image_shape, base_channels).to(self.device)
        # Le discriminateur devient un critique : meme attribut, pour que generate / invert /
        # decode / extract_features / get_codebook restent ceux de GAN.
        self.discriminator = ConvCritic(image_shape, base_channels).to(self.device)
        self.loss_history = {"generator": [], "discriminator": []}
        self.metric_history = {"wasserstein": [], "generator_variance": []}

    def gradient_penalty(self, real_batch: torch.Tensor, fake_batch: torch.Tensor) -> torch.Tensor:
        # -> scalaire. Contraint ||grad|| a 1 sur des points tires entre une vraie et une fausse
        # image : c'est la condition de 1-Lipschitz que le critique doit respecter, imposee par
        # une penalite plutot que par un clipping des poids (qui, lui, ampute sa capacite).
        epsilon = torch.rand(real_batch.size(0), 1, device=self.device)
        interpolated = (epsilon * real_batch + (1 - epsilon) * fake_batch).requires_grad_(True)
        scores = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def fit(
        self,
        feature_array: np.ndarray,
        epochs: int = 250,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
    ) -> "WGANGP":
        """
        Trains the critic and the generator on the Wasserstein objective.

        The critic maximises the gap between its scores on real and fake samples; the generator
        minimises it. n_critic critic steps per generator step: the theory needs a critic close
        to optimal for the gap to estimate the Wasserstein distance.

        Args:
            feature_array: shape (N, data_dim), float in [0, 1].

        Returns:
            self, with loss_history and metric_history filled with one value per epoch.
            metric_history["wasserstein"] estimates the distance between the two distributions:
            unlike a BCE loss it decreases as the generated images improve.
        """
        feature_tensor = torch.from_numpy(feature_array).float() * 2.0 - 1.0
        loader = DataLoader(
            TensorDataset(feature_tensor), batch_size=batch_size, shuffle=True, drop_last=True
        )
        # betas=(0.5, 0.9) : valeurs de l'article WGAN-GP
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
        )
        critic_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
        )

        self.loss_history = {"generator": [], "discriminator": []}
        self.metric_history = {"wasserstein": [], "generator_variance": []}
        for _ in range(epochs):
            critic_running, generator_running = 0.0, 0.0
            wasserstein_running, variance_running = 0.0, 0.0
            n_critic_steps, n_generator_steps = 0, 0

            for step, (real_batch,) in enumerate(loader):
                real_batch = real_batch.to(self.device)
                current_size = real_batch.size(0)

                # 1. Critique : creuser l'ecart entre vraies et fausses, sous contrainte de Lipschitz.
                #    detach() : le gradient ne doit pas remonter jusqu'a G.
                set_requires_grad(self.discriminator, True)
                fake_batch = self.generator(self.sample_noise(current_size)).detach()
                real_score = self.discriminator(real_batch).mean()
                fake_score = self.discriminator(fake_batch).mean()
                penalty = self.gradient_penalty(real_batch, fake_batch)
                critic_loss = fake_score - real_score + lambda_gp * penalty
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                critic_running += critic_loss.item()
                wasserstein_running += (real_score - fake_score).item()
                n_critic_steps += 1

                # 2. Generateur, un pas tous les n_critic : faire monter le score de ses fausses.
                #    Pas de forme saturante ici, le score n'est pas borne, le gradient ne meurt pas.
                if step % n_critic == 0:
                    set_requires_grad(self.discriminator, False)
                    generated = self.generator(self.sample_noise(current_size))
                    generator_loss = -self.discriminator(generated).mean()
                    generator_optimizer.zero_grad()
                    generator_loss.backward()
                    generator_optimizer.step()

                    generator_running += generator_loss.item()
                    with torch.no_grad():
                        variance_running += generated.var(dim=0).mean().item()
                    n_generator_steps += 1

            self.loss_history["discriminator"].append(critic_running / n_critic_steps)
            self.loss_history["generator"].append(generator_running / max(n_generator_steps, 1))
            self.metric_history["wasserstein"].append(wasserstein_running / n_critic_steps)
            self.metric_history["generator_variance"].append(
                variance_running / max(n_generator_steps, 1)
            )

        set_requires_grad(self.discriminator, True)
        return self
