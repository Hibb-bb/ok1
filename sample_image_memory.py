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


def load_mnist_images(
    num_images: int,
    pca_dim: Optional[int] = None,
    rng: np.random.Generator = None,
    R: float = 2.0,
):
    """
    Load MNIST images and optionally apply PCA.
    
    Args:
        num_images: Number of images to sample
        pca_dim: If provided, reduce to this dimension using PCA
        rng: Random number generator for sampling images
        R: Pixel-space scale after clip; before PCA if pca_dim is set, else raw pixels.
    
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

    if num_images > len(dataset):
        raise ValueError(
            f"Requested {num_images} images but MNIST train set has only {len(dataset)}."
        )

    # Sample random indices without replacement
    indices = rng.choice(len(dataset), size=num_images, replace=False)

    # Load and flatten images
    images = []
    for idx in indices:
        img, _ = dataset[int(idx)]  # Ignore labels
        images.append(img.numpy().flatten())  # (784,)
    
    images = np.array(images, dtype=np.float64)  # (num_images, 784)
    
    # Normalize to [0, 1] (already done by ToTensor, but ensure)
    images = np.clip(images, 1e-4, 1.0)
    images = images * R

    pca_fitted = None
    if pca_dim is not None:
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for PCA. Install with: pip install scikit-learn")
        original_dim = images.shape[1]
        max_components = min(num_images, pca_dim, original_dim)
        pca_fitted = PCA(n_components=max_components, random_state=rng.integers(0, 2**31))
        images = pca_fitted.fit_transform(images)

    return images.astype(np.float32), pca_fitted


def load_cifar10_images(
    num_images: int,
    pca_dim: Optional[int] = None,
    rng: np.random.Generator = None,
    R: float = 2.0,
):
    """
    Load CIFAR10 images and optionally apply PCA.
    
    Args:
        num_images: Number of images to sample
        pca_dim: If provided, reduce to this dimension using PCA
        rng: Random number generator for sampling images
        R: Pixel-space scale after clip; applied before PCA if pca_dim is set, else on raw pixels.
    
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

    if num_images > len(dataset):
        raise ValueError(
            f"Requested {num_images} images but CIFAR10 train set has only {len(dataset)}."
        )

    # Sample random indices without replacement
    indices = rng.choice(len(dataset), size=num_images, replace=False)

    # Load and flatten images
    images = []
    for idx in indices:
        img, _ = dataset[int(idx)]  # Ignore labels
        images.append(img.numpy().flatten())  # (3072,)
    
    images = np.array(images, dtype=np.float64)  # (num_images, 3072)
    
    # Normalize to [0, 1] (already done by ToTensor, but ensure)
    images = np.clip(images, 1e-6, 1.0)
    images = images * R

    pca_fitted = None
    if pca_dim is not None:
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for PCA. Install with: pip install scikit-learn")
        original_dim = images.shape[1]
        max_components = min(num_images, pca_dim, original_dim)
        pca_fitted = PCA(n_components=max_components, random_state=rng.integers(0, 2**31))
        images = pca_fitted.fit_transform(images)

    return images.astype(np.float32), pca_fitted


def load_images_single_class(
    dataset_name: str,
    class_id: int,
    num_images: int,
    pca_dim: Optional[int] = None,
    R: float = 2.0,
    rng: np.random.Generator = None,
):
    """
    Load images from one label only; optional PCA is fit on that class's sample only.

    Args:
        dataset_name: "mnist" or "cifar10".
        class_id: Label 0..9.
        num_images: How many images to draw (without replacement) from that class.
        pca_dim: If set, PCA dimension after clip and pixel scaling by R.
        R: Pixel-space scale after clip and before PCA.
        rng: Random generator.
    """
    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required. Install with: pip install torchvision")
    if rng is None:
        rng = np.random.default_rng()

    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name.lower() == "mnist":
        dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_name.lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist' or 'cifar10'")

    targets = np.asarray(dataset.targets)
    class_indices = np.flatnonzero(targets == class_id).astype(int)
    if len(class_indices) == 0:
        raise ValueError(f"No training images for class_id={class_id} in {dataset_name}")
    if num_images > len(class_indices):
        raise ValueError(
            f"Need {num_images} images but class {class_id} has only {len(class_indices)} samples."
        )

    pick = rng.choice(class_indices, size=num_images, replace=False)
    images = np.stack([dataset[int(i)][0].numpy().ravel() for i in pick], axis=0)
    images = images.astype(np.float64)
    images = np.clip(images, 1e-4, 1.0)
    images = images * R

    pca_fitted = None
    if pca_dim is not None:
        if not HAS_SKLEARN:
            raise ImportError("sklearn is required for PCA. Install with: pip install scikit-learn")
        max_components = min(num_images, pca_dim, images.shape[1])
        pca_fitted = PCA(n_components=max_components, random_state=rng.integers(0, 2**31))
        images = pca_fitted.fit_transform(images)

    return images.astype(np.float32), pca_fitted


def sample_images_from_dataset(
    dataset_name: str,
    M: int,
    dim: Optional[int] = None,
    R: float = 2.0,
    rng: np.random.Generator = None,
):
    """
    Unified interface for sampling images from datasets.
    Compatible with sample_euclidean_points_from_tangent_ball signature.
    
    Args:
        dataset_name: "mnist" or "cifar10"
        M: Number of images to sample
        dim: If provided, PCA dimensions; if None, raw pixels (after R scaling).
        R: Pixel-space scale after clip (always); before PCA when dim is set.
        rng: Random number generator
    
    Returns:
        images: (M, dim) array of images
        None: Placeholder for compatibility (like tangent ball returns v)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if dataset_name.lower() == "mnist":
        images, _ = load_mnist_images(M, pca_dim=dim, rng=rng, R=R)
    elif dataset_name.lower() == "cifar10":
        images, _ = load_cifar10_images(M, pca_dim=dim, rng=rng, R=R)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist' or 'cifar10'")
    
    return images, None
