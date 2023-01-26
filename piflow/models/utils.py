from typing import Dict
import gpflow
from gpflow import default_float
from gpflow.utilities import multiple_assign, parameter_dict, select_dict_parameters_with_prior
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def initialise_hyperparameters(model: gpflow.models.GPR, num_samples: int = 100):
    """Sample from hyperparameter priors and initialise with the best.
    
    :param model: The model.
    :param num_samples: Number of samples.
    """
    best_hypers = parameter_dict(model)
    best_lml = model.log_marginal_likelihood()
    for _ in range(num_samples):
        hypers = sample_hyperparameters(model)
        multiple_assign(model, hypers)
        lml = model.log_marginal_likelihood()
        if lml > best_lml:
            best_hypers = hypers
            best_lml = lml
    multiple_assign(model, best_hypers)


def sample_hyperparameters(model: gpflow.models.GPR) -> Dict[str, tf.Tensor]:
    """Take a single sample of the hyperparameters. Return as a
    parameter_dict.

    :param model: The model.
    :return: dictionary from ".submodule.parameter" to tensors.
    """
    params = select_dict_parameters_with_prior(model)
    for key, param in params.items():
        sample = param.prior.sample()
        params[key] = sample
    return params


def set_model_priors(model: gpflow.models.GPR):
    """Set priors for all model hyperparameters, and initialise the 
    likelihood hyperparameter.

    :param model: The model.
    """
    X, Y = model.data
    _set_lengthscales_prior(model, X)
    _set_variance_priors(model)


def _set_lengthscales_prior(model: gpflow.models.GPR, query_points: tf.Tensor):
    """Set lengthscale hyperparameters and their priors.
    
    :param model: The model.
    :param query_points: The input data, shape [N, D].
    """
    assert len(query_points.shape) == 2
    query_points = query_points.numpy()
    differences = np.abs(query_points[:, None, :] -query_points[None, ...])  # [N, N, D]
    triu_inds = np.triu_indices_from(differences[..., 0], k=1)
    triu_diffs = differences[triu_inds]
    log_triu_diffs = np.log(triu_diffs)
    log_max = np.max(log_triu_diffs, axis=0)
    log_min = np.min(log_triu_diffs, axis=0)
    log_median = np.median(log_triu_diffs, axis=0)
    scale = np.minimum(log_max - log_median, log_median - log_min) / 3
    log_median = tf.constant(log_median, dtype=default_float())
    scale = tf.constant(scale, dtype=default_float())
    model.kernel.lengthscales.assign(tf.math.exp(log_median))
    model.kernel.lengthscales.prior = tfp.distributions.LogNormal(log_median, scale)


def _set_variance_priors(model: gpflow.models.GPR, observations: tf.Tensor = None):
    """Set signal and noise variance hyperparameters and their priors.
    
    :param model: The model.
    :param observations: The observations.
    """
    LOG_SCALE = 1e-2
    SIGNAL_TO_NOISE_RATIO = math.e ** 2
    if observations is not None:
        variance = tf.math.reduce_variance(observations)
    else:
        variance = tf.constant(1, dtype=default_float())
    likelihood_var = variance / SIGNAL_TO_NOISE_RATIO
    model.kernel.variance.assign(variance)
    model.kernel.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(variance), LOG_SCALE
    )
    model.likelihood.variance.assign(likelihood_var)
    model.likelihood.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(likelihood_var), LOG_SCALE
    )
