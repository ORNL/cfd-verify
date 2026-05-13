import numpy as np


def mesh_size(domain: float, count: int, dim: int=3) -> np.ndarray:
    """Compute average discretization size

    Parameters
    ----------
    domain : float
        Size of domain
    count : int
        Number of cells/elements/etc. domain is divided by
    dim : int
        Dimension of domain

    Returns
    -------
    : np.ndarray
        Average discretization size
    """
    return np.power(np.array(domain)/count, 1/dim)
