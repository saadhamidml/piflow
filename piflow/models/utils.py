import gpflow
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.models.gpflow.builders import KERNEL_PRIOR_SCALE

# TODO: This needs to be rewritten to initialise by sampling.
def initialise_lengthscales(query_points: tf.Tensor):
    """Initialise lengthscales to the median distance along each dimension.
    
    :param query_points: The input data, shape [N, D].
    """
    assert len(query_points.shape) == 2
    differences = tf.abs(
        tf.expand_dims(query_points, axis=1) - tf.expand_dims(query_points, axis=0)
    )  # [N, N, D]
    medians = []
    for dim in range(query_points.shape[1]):
        diff_dim = tf.linalg.set_diag(
            differences[..., dim], tf.zeros(query_points.shape[0], dtype=gpflow.default_float())
        )
        diff_dim = tf.reshape(tf.linalg.band_part(diff_dim, 0, -1), (-1,))

    query_points = query_points.numpy()
    differences = np.abs(query_points[:, None, :] -query_points[None, ...])  # [N, N, D]
    triu_inds = np.triu_indices_from(differences[..., 0])
    triu_diffs = differences[triu_inds]
    lengthscales = np.median(triu_diffs, axis=0)
    return tf.constant(lengthscales, dtype=gpflow.default_float())


def set_model_priors(model: gpflow.models.GPR):
    """Set priors for all model hyperparameters, and initialise the 
    likelihood hyperparameter.

    :param model: The model.
    """
    model.kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(model.kernel.lengthscales), 1e-1
    )
    model.kernel.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(model.kernel.variance), 1e-1
    )
    model.likelihood.variance.assign(model.kernel.variance / 100)
    model.likelihood.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(model.likelihood.variance), 1e-2
    )
