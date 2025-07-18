"""Stable linear algebra operations for SSL.

These functions provide numerically stable versions of common linear algebra
operations by applying preconditioning to avoid numerical issues.
"""

import torch


def stable_eigvalsh(C: torch.Tensor, eps: float = None):
    """Compute eigenvalues of a symmetric matrix with numerical stability.

    Applies preconditioning to ensure stable decomposition by normalizing
    the matrix and adding identity. This has no impact on gradients but
    improves numerical stability.

    Args:
        C: Symmetric matrix
        eps: Small value for numerical stability. If None, uses machine epsilon

    Returns:
        Eigenvalues of the matrix
    """
    # preconditioning to ensure stable decomposition
    # we first normalize the sdp matrix by its max value
    # and also add identity. This pre-processing step has
    # no impact on the gradient of the loss, it is only here
    # for numerical stability over the eigen solver
    if eps is None:
        eps = torch.finfo(C.dtype).eps
    else:
        assert eps >= 0
    rescalor = C.abs().max()
    Id = torch.eye(C.size(1), device=C.device, dtype=C.dtype)
    C = C.div(rescalor).add(Id)

    eigvals = torch.linalg.eigvalsh(C)

    # - step 1: removing the effect of adding the Identity matrix
    # - step 2: removing the effect of dividing by P
    shift = torch.where(eigvals - 1 > eps, 1, eigvals - eps)
    return eigvals.sub(shift).mul(rescalor)


def stable_eigh(C, eps=None):
    """Compute eigenvalues and eigenvectors of a symmetric matrix with stability.

    Applies preconditioning to ensure stable decomposition by normalizing
    the matrix and adding identity. This has no impact on gradients but
    improves numerical stability.

    Args:
        C: Symmetric matrix
        eps: Small value for numerical stability. If None, uses machine epsilon

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # preconditioning to ensure stable decomposition
    # we first normalize the sdp matrix by its max value
    # and also add identity. This pre-processing step has
    # no impact on the gradient of the loss, it is only here
    # for numerical stability over the eigen solver
    if eps is None:
        eps = torch.finfo(C.dtype).eps
    else:
        assert eps >= 0
    rescalor = C.abs().max()
    Id = torch.eye(C.size(1), device=C.device, dtype=C.dtype)
    C = C.div(rescalor).add(Id)

    eigvals, eigvecs = torch.linalg.eigh(C)

    # - step 1: removing the effect of adding the Identity matrix
    # - step 2: removing the effect of dividing by P
    shift = torch.where(eigvals - 1 > eps, 1, eigvals - eps)
    return eigvals.sub(shift).mul(rescalor), eigvecs


def stable_svd(C, eps=None, full_matrices=True):
    """Compute SVD with numerical stability.

    Applies preconditioning by normalizing the matrix to improve stability.

    Args:
        C: Input matrix
        eps: Small value for numerical stability. If None, uses machine epsilon
        full_matrices: Whether to compute full-sized U and V matrices

    Returns:
        Tuple of (U, singular_values, Vh)
    """
    # preconditioning to ensure stable decomposition
    # we first normalize the sdp matrix by its max value
    # and also add identity. This pre-processing step has
    # no impact on the gradient of the loss, it is only here
    # for numerical stability over the eigen solver
    if eps is None:
        eps = torch.finfo(C.dtype).eps
    else:
        assert eps >= 0
    rescalor = C.abs().max()
    C = C.div(rescalor)

    U, vals, Vh = torch.linalg.svd(C, full_matrices=full_matrices)

    # removing the effect of dividing by P
    return U, vals.clip(eps).mul(rescalor), Vh


def stable_svdvals(C, eps=None):
    """Compute singular values with numerical stability.

    Applies preconditioning by normalizing the matrix to improve stability.

    Args:
        C: Input matrix
        eps: Small value for numerical stability. If None, uses machine epsilon

    Returns:
        Singular values of the matrix
    """
    # preconditioning to ensure stable decomposition
    # we first normalize the sdp matrix by its max value
    # and also add identity. This pre-processing step has
    # no impact on the gradient of the loss, it is only here
    # for numerical stability over the eigen solver
    if eps is None:
        eps = torch.finfo(C.dtype).eps
    else:
        assert eps >= 0
    rescalor = C.abs().max()
    C = C.div(rescalor)
    vals = torch.linalg.svdvals(C)
    return vals.clip(eps).mul(rescalor)
