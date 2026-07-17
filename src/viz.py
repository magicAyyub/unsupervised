"""Helpers de visualisation et d'echantillonnage communs aux notebooks."""

import numpy as np
import matplotlib.pyplot as plt

from src.metrics import Latent


def format_fixed_note(fields: dict, omit=(), extra=None) -> str:
    """Met en forme la note du bas d'une figure a partir des hyper-parametres fixes."""
    parts = [f"{key}={value}" for key, value in fields.items() if key not in omit]
    if extra:
        parts.append(extra)
    return "fixe: " + " | ".join(parts)


def flattened_vector_to_image(flat_vector, image_shape):
    channels, height, width = image_shape
    image = flat_vector.reshape(channels, height, width)
    return image[0] if channels == 1 else np.transpose(image, (1, 2, 0))


def finish_figure(fig, title=None, config=None, layout=True):
    """
    Titre en haut, configuration en petit en bas
    layout=False pour les figures deja construites en layout='constrained'.
    """
    if title:
        fig.suptitle(title)
    if config:
        fig.supxlabel(config, fontsize=7.5, color="0.45")
    if layout:
        fig.tight_layout()
    plt.show()


def show_original_vs_reconstruction_grid(originals, reconstructions, image_shape, n=8,
                                         title=None, config=None):
    cmap = "gray" if image_shape[0] == 1 else None
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3.2))
    for i in range(n):
        axes[0, i].imshow(flattened_vector_to_image(originals[i], image_shape), cmap=cmap)
        axes[1, i].imshow(np.clip(flattened_vector_to_image(reconstructions[i], image_shape), 0, 1), cmap=cmap)
        for row in (0, 1):
            axes[row, i].set_xticks([]); axes[row, i].set_yticks([])
    axes[0, 0].set_ylabel("original"); axes[1, 0].set_ylabel("reconstruit")
    finish_figure(fig, title, config)


def show_image_grid(flat_images, image_shape, nrow=4, ncol=8, title=None, config=None):
    cmap = "gray" if image_shape[0] == 1 else None
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.1, nrow * 1.1 + 0.6))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(flat_images):
            ax.imshow(np.clip(flattened_vector_to_image(flat_images[i], image_shape), 0, 1), cmap=cmap)
    finish_figure(fig, title, config)


def show_labeled_image_rows(rows, image_shape, row_labels, n=8, title=None, config=None):
    """Une ligne d'images par configuration"""
    cmap = "gray" if image_shape[0] == 1 else None
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.3, len(rows) * 1.45))
    axes = np.atleast_2d(axes)
    for row_index, (images, label) in enumerate(zip(rows, row_labels)):
        for col in range(n):
            ax = axes[row_index, col]
            ax.set_xticks([]); ax.set_yticks([])
            if col < len(images):
                ax.imshow(np.clip(flattened_vector_to_image(images[col], image_shape), 0, 1), cmap=cmap)
        axes[row_index, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    finish_figure(fig, title, config)


def plot_latent_scatter(latent_2d, labels, class_names=None, title=None, config=None):
    fig = plt.figure(figsize=(6, 5))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="tab10", s=6, alpha=0.6)
    if class_names is not None:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_names, title="classe", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        plt.colorbar(scatter, label="chiffre")
    plt.xlabel("z1"); plt.ylabel("z2")
    finish_figure(fig, title, config)


def print_compression_report(report):
    for key, value in report.items():
        print(f"{key:>24}: {value:,.4f}" if isinstance(value, float) else f"{key:>24}: {value}")


def sample_gaussian_latent(latent_codes, n_samples, seed=0):
    """
    Ajuste une gaussienne (moyenne + covariance PLEINE) sur les codes observes et en tire
    de nouveaux. La covariance pleine capture l'orientation du nuage, pas seulement son
    etendue par axe: les dimensions latentes d'un AutoEncoder sont correlees, rien ne les
    decorrelant pendant l'entrainement.

    Note: c'est un PIS-ALLER propre a l'AutoEncoder, dont l'espace latent n'obeit a aucune
    loi connue d'avance. Un VAE n'en a pas besoin (cf. sample_prior).
    """
    rng = np.random.default_rng(seed)
    mean = latent_codes.mean(axis=0)
    cov = np.cov(latent_codes, rowvar=False)
    return rng.multivariate_normal(mean, cov, size=n_samples).astype(np.float32)


def generate_from_latent_using_gaussian(model, latent, n_samples, seed=0):
    codes = sample_gaussian_latent(latent.array, n_samples, seed=seed)
    return model.decode(Latent(array=codes, nature="continuous"))


def interpolate_latent(z_start, z_end, steps=10):
    alphas = np.linspace(0, 1, steps)[:, None]
    return ((1 - alphas) * z_start[None, :] + alphas * z_end[None, :]).astype(np.float32)


def subsample_dataset(images, labels, n, seed=0):
    if n >= len(images):
        return images, labels
    idx = np.random.default_rng(seed).choice(len(images), size=n, replace=False)
    return images[idx], labels[idx]
