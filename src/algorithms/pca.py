import numpy as np
import torch

def _compress_image(img: np.ndarray, k: int) -> dict:
    """
    Compresse une image 2D ou 3D en aplatissant les canaux de couleur si nécessaire.
    """
    original_ndim = img.ndim
    
    if img.ndim == 2:
        H, W = img.shape
        flat_img = img
        layout = None
    elif img.ndim == 3:
        # Déterminer la disposition : (C, H, W) ou (H, W, C)
        if img.shape[0] in (1, 3):
            layout = 'CHW'
            # Transposer en HWC pour aplatir correctement les dimensions spatiales et canaux
            img_hwc = img.transpose(1, 2, 0)
        else:
            layout = 'HWC'
            img_hwc = img
            
        H, W, C = img_hwc.shape
        # Aplatir les canaux dans la dimension des colonnes : (H, W * C)
        flat_img = img_hwc.reshape(H, W * C)
    else:
        raise ValueError(f"Image shape {img.shape} not supported")
        
    # ACP sur la matrice 2D aplatie (H, W_flat)
    mean = np.mean(flat_img, axis=0)
    centered = flat_img - mean
    
    cov_matrix = np.cov(flat_img, rowvar=False)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix.item()]])
        
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Trier par ordre décroissant
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Sélectionner les k premières composantes
    k = min(k, flat_img.shape[1])
    components = eigenvectors[:, :k]
    
    # Projection
    compressed = centered @ components
    
    return {
        'compressed': compressed,
        'components': components,
        'mean': mean,
        'ndim': original_ndim,
        'layout': layout,
        'img_shape': img.shape
    }

def _decompress_image(compressed_dict: dict) -> np.ndarray:
    """
    Décompresse une image compressée par ACP en utilisant la reconstruction.
    """
    compressed = compressed_dict['compressed']
    components = compressed_dict['components']
    mean = compressed_dict['mean']
    ndim = compressed_dict['ndim']
    layout = compressed_dict['layout']
    img_shape = compressed_dict['img_shape']
    
    # Reconstruction de la forme plate : (H, k) @ (k, W_flat) + Moyenne
    reconstructed_flat = np.dot(compressed, components.T) + mean
    
    if ndim == 2:
        return reconstructed_flat
    elif ndim == 3:
        if layout == 'CHW':
            C, H, W = img_shape
            # Re-former en HWC, puis transposer en CHW
            reconstructed_hwc = reconstructed_flat.reshape(H, W, C)
            return reconstructed_hwc.transpose(2, 0, 1)
        else:
            H, W, C = img_shape
            return reconstructed_flat.reshape(H, W, C)
    else:
        raise ValueError(f"Unsupported dimensions for decompression: {ndim}")

def compress(images, k: int) -> dict:
    """
    Compresses an image or a batch of images using PCA.
    Flattens color channels into a single dimension.
    Works with both NumPy arrays and PyTorch Tensors.

    Args:
        images: PyTorch tensor or NumPy array.
                Shapes supported: (H, W), (C, H, W), (H, W, C), (B, C, H, W), (B, H, W, C).
        k: Number of principal components to keep.

    Returns:
        dict: Compressed data and projection metadata.
    """
    is_tensor = isinstance(images, torch.Tensor)
    if is_tensor:
        device = images.device
        img_np = images.detach().cpu().numpy()
    else:
        device = None
        img_np = np.array(images)
        
    original_dtype = img_np.dtype
    original_shape = img_np.shape
    
    if img_np.ndim == 4:
        # Traiter un lot (batch) d'images
        B = img_np.shape[0]
        compressed_list = []
        components_list = []
        mean_list = []
        img_shape_list = []
        
        if img_np.shape[1] in (1, 3):
            layout = 'BCHW'
        else:
            layout = 'BHWC'
            
        for i in range(B):
            res = _compress_image(img_np[i], k)
            compressed_list.append(res['compressed'])
            components_list.append(res['components'])
            mean_list.append(res['mean'])
            img_shape_list.append(res['img_shape'])
            
        compressed_dict = {
            'compressed': np.stack(compressed_list, axis=0),
            'components': np.stack(components_list, axis=0),
            'mean': np.stack(mean_list, axis=0),
            'img_shape': img_shape_list,
            'is_batch': True,
            'layout': layout,
            'ndim': 4
        }
    else:
        # Traiter une seule image
        res = _compress_image(img_np, k)
        compressed_dict = {
            'compressed': res['compressed'],
            'components': res['components'],
            'mean': res['mean'],
            'img_shape': res['img_shape'],
            'is_batch': False,
            'layout': res['layout'],
            'ndim': res['ndim']
        }
        
    compressed_dict.update({
        'is_tensor': is_tensor,
        'device': device,
        'dtype': original_dtype,
        'original_shape': original_shape
    })
    
    return compressed_dict

def decompress(compressed_dict: dict):
    """
    Decompresses the PCA-compressed representation back into the original image shape.

    Args:
        compressed_dict: Dictionary returned by the `compress` function.

    Returns:
        Reconstructed image in its original format (Tensor/ndarray) and shape.
    """
    is_batch = compressed_dict['is_batch']
    ndim = compressed_dict['ndim']
    compressed = compressed_dict['compressed']
    components = compressed_dict['components']
    mean = compressed_dict['mean']
    layout = compressed_dict.get('layout')
    img_shape = compressed_dict['img_shape']
    
    if is_batch:
        B = compressed.shape[0]
        reconstructed_batch = []
        for i in range(B):
            single_dict = {
                'compressed': compressed[i],
                'components': components[i],
                'mean': mean[i],
                'ndim': ndim - 1,
                'layout': 'CHW' if layout == 'BCHW' else 'HWC',
                'img_shape': img_shape[i]
            }
            reconstructed_batch.append(_decompress_image(single_dict))
        reconstructed = np.stack(reconstructed_batch, axis=0)
    else:
        single_dict = {
            'compressed': compressed,
            'components': components,
            'mean': mean,
            'ndim': ndim,
            'layout': layout,
            'img_shape': img_shape
        }
        reconstructed = _decompress_image(single_dict)
        
    # Cliping pour éviter les débordements avec les entiers
    dtype = compressed_dict['dtype']
    if np.issubdtype(dtype, np.integer):
        reconstructed = np.clip(reconstructed, 0, 255).astype(dtype)
    else:
        reconstructed = reconstructed.astype(dtype)
        
    if compressed_dict['is_tensor']:
        return torch.from_numpy(reconstructed).to(compressed_dict['device'])
    return reconstructed