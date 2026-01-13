import numpy as np
from typing import Optional
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def load_mnist_images(num_images: int, pca_dim: Optional[int] = None, rng: np.random.Generator = None):
    """
    Load MNIST images and optionally apply PCA.
    
    Args:
        num_images: Number of images to sample
        pca_dim: If provided, reduce to this dimension using PCA
        rng: Random number generator for sampling images
    
    Returns:
        images: (num_images, dim) array of normalized images
        pca_fitted: Fitted PCA object (None if pca_dim is None)
    """
    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required for MNIST/CIFAR10. Install with: pip install torchvision")
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Sample random indices
    indices = rng.integers(0, len(dataset), size=num_images)
    
    # Load and flatten images
    images = []
    for idx in indices:
        img, _ = dataset[idx]  # Ignore labels
        images.append(img.numpy().flatten())  # (784,)
    
    images = np.array(images, dtype=np.float64)  # (num_images, 784)
    
    # Normalize to [0, 1] (already done by ToTensor, but ensure)
    images = np.clip(images, 1e-4, 1.0)
    
    # Optional PCA reduction
    pca_fitted = None
    if pca_dim is not None:
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for PCA. Install with: pip install scikit-learn")
        # Limit PCA components to min(num_images, pca_dim, original_dim) to avoid errors
        original_dim = images.shape[1]
        max_components = min(num_images, pca_dim, original_dim)
        pca_fitted = PCA(n_components=max_components, random_state=rng.integers(0, 2**31))
        images = pca_fitted.fit_transform(images)
    
    return images.astype(np.float32), pca_fitted


def load_cifar10_images(num_images: int, pca_dim: Optional[int] = None, rng: np.random.Generator = None):
    """
    Load CIFAR10 images and optionally apply PCA.
    
    Args:
        num_images: Number of images to sample
        pca_dim: If provided, reduce to this dimension using PCA
        rng: Random number generator for sampling images
    
    Returns:
        images: (num_images, dim) array of normalized images
        pca_fitted: Fitted PCA object (None if pca_dim is None)
    """
    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required for MNIST/CIFAR10. Install with: pip install torchvision")
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Load CIFAR10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Sample random indices
    indices = rng.integers(0, len(dataset), size=num_images)
    
    # Load and flatten images
    images = []
    for idx in indices:
        img, _ = dataset[idx]  # Ignore labels
        images.append(img.numpy().flatten())  # (3072,)
    
    images = np.array(images, dtype=np.float64)  # (num_images, 3072)
    
    # Normalize to [0, 1] (already done by ToTensor, but ensure)
    images = np.clip(images, 1e-4, 1.0)
    
    # Optional PCA reduction
    pca_fitted = None
    if pca_dim is not None:
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for PCA. Install with: pip install scikit-learn")
        # Limit PCA components to min(num_images, pca_dim, original_dim) to avoid errors
        original_dim = images.shape[1]
        max_components = min(num_images, pca_dim, original_dim)
        pca_fitted = PCA(n_components=max_components, random_state=rng.integers(0, 2**31))
        images = pca_fitted.fit_transform(images)
    
    return images.astype(np.float32), pca_fitted


def sample_images_from_dataset(
    dataset_name: str,
    M: int,
    dim: Optional[int] = None,
    rng: np.random.Generator = None
):
    """
    Unified interface for sampling images from datasets.
    Compatible with sample_euclidean_points_from_tangent_ball signature.
    
    Args:
        dataset_name: "mnist" or "cifar10"
        M: Number of images to sample
        dim: If provided, use PCA to reduce to this dimension
        rng: Random number generator
    
    Returns:
        images: (M, dim) array of images
        None: Placeholder for compatibility (like tangent ball returns v)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if dataset_name.lower() == "mnist":
        images, _ = load_mnist_images(M, pca_dim=dim, rng=rng)
    elif dataset_name.lower() == "cifar10":
        images, _ = load_cifar10_images(M, pca_dim=dim, rng=rng)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist' or 'cifar10'")
    
    return images, None
