"""Information value of a prospective observation without observer mutation."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, xlogy


def entropy(weights):
    return -xlogy(weights, weights).sum(axis=-1)


def log_weights(weights):
    with np.errstate(divide="ignore"):
        return np.log(weights)


def likelihood_posterior(weights, residual, scale):
    log_joint = log_weights(weights)-.5*np.square(residual/scale)
    return np.exp(log_joint-logsumexp(log_joint, axis=-1, keepdims=True))


def information_values(observation, weights, score, scale):
    """Gaussian quadrature over the scalar observation of each candidate.

    observation and score have shape [hypothesis, action]. Decision value
    is the expected improvement from selecting a different continuation.
    """
    if len(weights) == 1:
        return np.zeros(observation.shape[1]), np.zeros(observation.shape[1])
    nodes, mass = QUADRATURE
    means = observation.T
    values = means[..., None]+scale*nodes
    posterior = likelihood_posterior(weights, values[..., None]-means[:, None, None, :], scale)
    density = weights[:, None]*mass[None, :]
    gain = entropy(weights)-np.einsum("akq,kq->a", entropy(posterior), density)
    expected = np.einsum("akqi,ij->akqj", posterior, score).max(axis=-1)
    value = np.einsum("akq,kq->a", expected, density)-(weights@score).max()
    return np.maximum(gain, 0), np.maximum(value, 0)


_NODES, _MASS = np.polynomial.hermite.hermgauss(15)
QUADRATURE = (_NODES*np.sqrt(2), _MASS/np.sqrt(np.pi))
