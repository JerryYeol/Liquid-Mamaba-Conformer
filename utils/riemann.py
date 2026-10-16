# -*- coding: utf-8 -*-
# Ported verbatim from pyRiemann (https://github.com/pyRiemann/pyRiemann),
# BSD-3-Clause, from pyriemann/geometry/base.py and
# pyriemann/geometry/mean.py.  Adapted for numpy 1.26: the upstream code goes
# through array_api_compat so it can run on numpy>=2 and torch, which this
# environment does not have.  The numerical content is unchanged -- every
# function body below is the upstream body with `xp` -> `np`, `X.mT` ->
# np.swapaxes(X, -1, -2), and the array-api device/dtype plumbing dropped.
#
# Deliberately NOT ported: the `_vectorize_nd` decorator and the array-api /
# torch backends.  Every caller in this repo passes exactly one 3D array,
# (n_matrices, n, n); mean_riemann asserts that rather than carrying ~60 lines
# of batching machinery nothing here uses.
import warnings

import numpy as np


def ctranspose(X):
    """Conjugate transpose operator.

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        Matrices.

    Returns
    -------
    X_new : ndarray, shape (..., m, n)
        Conjugate transpose of X.

    Notes
    -----
    .. versionadded:: 0.9
    """
    return np.conj(X).swapaxes(-1, -2)


def _matrix_operator(X, operator):
    """Matrix function for SPD matrices.

    Mirrors pyriemann.geometry.base._matrix_operator.

    Note the guard below only rejects inf/nan.  It does *not* reject negative
    eigenvalues, so `logm` of a matrix with one silently returns nan (with a
    numpy RuntimeWarning) and the ValueError surfaces later, inside whichever
    caller eventually feeds that nan back through `expm`.  That is upstream
    behaviour and is kept as-is -- the fix belongs upstream of this call, in
    nearest_sym_pos_def.
    """
    if not hasattr(X, "ndim") or X.ndim < 2:
        raise ValueError("Input must be at least a 2D ndarray")
    if X.shape[-2] != X.shape[-1]:
        raise ValueError("Input must contain square matrices")
    if np.issubdtype(X.dtype, np.inexact) and not bool(np.all(np.isfinite(X))):
        raise ValueError(
            "Matrices must be positive definite. "
            "You should add regularization to avoid this error."
        )

    eigvals, eigvecs = np.linalg.eigh(X)
    eigvals = operator(eigvals)
    X_new = eigvecs @ (eigvals[..., None] * ctranspose(eigvecs))
    return X_new


def expm(C):
    r"""Exponential of SPD matrices.

    .. math::
        \mathbf{D} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^H

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix exponential of C.
    """
    return _matrix_operator(C, np.exp)


def invsqrtm(C):
    r"""Inverse square root of SPD matrices.

    .. math::
        \mathbf{D} = \mathbf{V} \left( \mathbf{\Lambda} \right)^{-1/2} \mathbf{V}^H

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix inverse square root of C.
    """
    def isqrt(x):
        return 1. / np.sqrt(x)
    return _matrix_operator(C, isqrt)


def logm(C):
    r"""Logarithm of SPD matrices.

    .. math::
        \mathbf{D} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^H

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix logarithm of C.
    """
    return _matrix_operator(C, np.log)


def powm(C, alpha):
    r"""Power of SPD matrices.

    .. math::
        \mathbf{D} = \mathbf{V} \left( \mathbf{\Lambda} \right)^{\alpha} \mathbf{V}^H

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices.
    alpha : float
        The power to apply.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix power of C.
    """
    def power(x):
        return x**alpha
    return _matrix_operator(C, power)


def sqrtm(C):
    r"""Square root of SPD matrices.

    .. math::
        \mathbf{D} = \mathbf{V} \left( \mathbf{\Lambda} \right)^{1/2} \mathbf{V}^H

    Parameters
    ----------
    C : ndarray, shape (..., n, n)
        SPD matrices.

    Returns
    -------
    D : ndarray, shape (..., n, n)
        Matrix square root of C.
    """
    return _matrix_operator(C, np.sqrt)


def nearest_sym_pos_def(X, reg=1e-6):
    """Find the nearest SPD matrices.

    A NumPy port of John D'Errico's ``nearestSPD`` MATLAB code [1]_, which
    credits [2]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Square matrices.
    reg : float, default=1e-6
        Regularization parameter.

    Returns
    -------
    P : ndarray, shape (..., n, n)
        Nearest SPD matrices.

    Notes
    -----
    The final block is the one that matters for this repo: when
    ``min(eig)/max(eig) < reg`` it adds ``reg`` to *every* eigenvalue, which
    puts a floor of ``reg`` under the smallest one and therefore a ceiling of
    ``1/sqrt(reg)`` on ``invsqrtm`` of anything derived from the result.

    References
    ----------
    .. [1] `nearestSPD
        <https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd>`_
        J. D'Errico, MATLAB Central File Exchange
    .. [2] `Computing a nearest symmetric positive semidefinite matrix
        <https://www.sciencedirect.com/science/article/pii/0024379588902236>`_
        N.J. Higham, Linear Algebra and its Applications, vol 103, 1988
    """
    n = X.shape[-1]
    eps = np.finfo(X.dtype).eps

    # Symmetrize
    A = (X + ctranspose(X)) / 2

    _, s, Vh = np.linalg.svd(A)
    H = ctranspose(Vh) @ (s[..., None] * Vh)
    B = (A + H) / 2
    P = (B + ctranspose(B)) / 2

    # PD fix: iteratively shift non-PD matrices
    eigvals = np.linalg.eigvalsh(P)
    neg_ev = np.any(eigvals <= 0, axis=-1)  # (...,)

    if bool(np.any(neg_ev)):
        # Frobenius norm, batched.  Upstream uses xp.linalg.matrix_norm, which
        # returns shape (...,); np.linalg.norm(ord='fro') rejects 3D input, so
        # spell the batch-aware form out.
        spacing = np.abs(np.sqrt(np.sum(A * A, axis=(-2, -1)))) * eps
        eye_n = np.eye(n, dtype=X.dtype)
        k = 1
        while bool(np.any(neg_ev)) and k < 100:
            mineig = np.min(np.linalg.eigvalsh(P), axis=-1)
            shift = np.where(neg_ev, -mineig * k**2 + spacing, 0.0)
            P = P + shift[..., None, None] * eye_n
            eigvals = np.linalg.eigvalsh(P)
            neg_ev = np.any(eigvals <= 0, axis=-1)
            k += 1

    # Regularize
    ei, ev = np.linalg.eigh(P)
    ratio = np.min(ei, axis=-1) / np.max(ei, axis=-1)
    needs_reg = ratio < reg  # (...,)
    if bool(np.any(needs_reg)):
        ei_reg = ei + reg
        P_reg = ev @ (ei_reg[..., None] * ctranspose(ev))
        P = np.where(needs_reg[..., None, None], P_reg, P)

    return P


def _check_weights(weights, n_weights):
    """Port of pyriemann.geometry._check.check_weights.

    ``None`` (the only value any caller here passes) becomes uniform weights.
    """
    if weights is None:
        return np.full(n_weights, 1.0 / n_weights)
    weights = np.asarray(weights)
    if len(weights) != n_weights:
        raise ValueError(
            "Weights do not have the good length, expected "
            f"{n_weights}, got {len(weights)}")
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    if np.sum(weights) != 1:
        warnings.warn(
            "Weights do not sum to 1. Check your inputs; they will be used "
            "as-is.", stacklevel=2)
    return weights


def _check_init(init, n):
    """Port of the shape/symmetry part of pyriemann.geometry._check.check_init."""
    init = np.asarray(init)
    if init.shape != (n, n):
        raise ValueError(
            f"Init matrix must be of shape ({n}, {n}), got {init.shape}")
    if not np.allclose(init, ctranspose(init)):
        raise ValueError("Init matrix must be symmetric")
    return init


def mean_euclid(X, sample_weight=None):
    r"""Mean of matrices according to the Euclidean metric.

    .. math::
        \mathbf{M} = \sum_i w_i \ \mathbf{X}_i

    Parameters
    ----------
    X : ndarray, shape (..., n_matrices, n, m)
        Set of matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (..., n, m)
        Euclidean mean.
    """
    if sample_weight is None:
        return np.mean(X, axis=-3)
    n_matrices = X.shape[-3]
    sample_weight = _check_weights(sample_weight, n_matrices)
    return np.tensordot(sample_weight, X, axes=([0], [-3]))


def mean_logeuclid(X, sample_weight=None):
    r"""Mean of SPD matrices according to the log-Euclidean metric.

    .. math::
        \mathbf{M} = \exp{ \left( \sum_i w_i \ \log{\mathbf{X}_i} \right) }

    Parameters
    ----------
    X : ndarray, shape (..., n_matrices, n, n)
        Set of SPD matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (..., n, n)
        Log-Euclidean mean.
    """
    M = expm(mean_euclid(logm(X), sample_weight=sample_weight))
    return M


def mean_riemann(X, *, tol=10e-9, maxiter=50, init=None, sample_weight=None):
    r"""Mean of SPD matrices according to the Riemannian metric.

    The affine-invariant Riemannian mean minimizes the sum of squared
    affine-invariant Riemannian distances :math:`d_R` to all SPD matrices [1]_:

    .. math::
         \arg \min_{\mathbf{M}} \sum_i w_i \ d_R (\mathbf{M}, \mathbf{X}_i)^2

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD matrices.  (Upstream also accepts leading batch dimensions;
        this port requires exactly 3D.)
    tol : float, default=10e-9
        Tolerance to stop the gradient descent.
    maxiter : int, default=50
        Maximum number of iterations.
    init : None | ndarray, shape (n, n), default=None
        A SPD matrix used to initialize the gradient descent.
        If None, the weighted Euclidean mean is used.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, n)
        Affine-invariant Riemannian mean.

    References
    ----------
    .. [1] `Principal geodesic analysis for the study of nonlinear statistics
        of shape
        <https://ieeexplore.ieee.org/document/1318725>`_
        P.T. Fletcher, C. Lu, S. M. Pizer, S. Joshi.
        IEEE Trans Med Imaging, 2004, 23(8), pp. 995-1005
    .. [2] `Approximate Joint Diagonalization and Geometric Mean of Symmetric
        Positive Definite Matrices
        <https://arxiv.org/abs/1505.07343>`_
        M. Congedo, B. Afsari, A. Barachant, M. Moakher. PLOS ONE, 2015
    """
    if X.ndim != 3:
        raise ValueError(
            f"This port expects a 3D array, got shape {X.shape}. "
            "Upstream _vectorize_nd batching was not ported; see module note.")
    n_matrices, n, _ = X.shape
    sample_weight = _check_weights(sample_weight, n_matrices)
    if init is None:
        M = mean_euclid(X, sample_weight=sample_weight)
    else:
        M = _check_init(init, n)

    nu = 1.0
    tau = np.finfo(np.float64).max
    for _ in range(maxiter):
        M12, Mm12 = sqrtm(M), invsqrtm(M)
        J = np.tensordot(
            sample_weight, logm(Mm12 @ X @ Mm12), axes=([0], [0]),
        )
        M = M12 @ expm(nu * J) @ M12

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    else:
        warnings.warn("Convergence not reached", stacklevel=2)

    return M
