import gpflow
from gpflow import default_float
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.models.gpflow.models import GaussianProcessRegression
import pytest
from piflow.models.transforms import DataTransformMixin, IdentityTransformer, MinMaxTransformer, StandardTransformer
from piflow.models.utils import initialise_hyperparameters, set_model_priors

from piflow.models.warped import MMLT_GPR, WSABI_L_GPR, WarpedGaussianProcessRegression
from piflow.probabilistic_integrator import IntegrandModel


@pytest.mark.parametrize('warping', [None, 'WSABI-L', 'MMLT'])
@pytest.mark.parametrize('prior_type', ['gaussian', 'uniform'])
def test_integrand_model(warping: str, prior_type: str):
    np.random.seed(0)
    tf.random.set_seed(0)
    # Generate data from a Gaussian.
    integrand = tfp.distributions.MultivariateNormalFullCovariance(
        loc=tf.constant([0.5, 0.5], dtype=tf.float64),
        covariance_matrix=tf.constant([[0.1, 0.], [0., 0.1]], dtype=tf.float64)
    )
    if prior_type == 'gaussian':
        prior = tfp.distributions.MultivariateNormalFullCovariance(
            loc=tf.constant([0., 0.], dtype=tf.float64),
            covariance_matrix=tf.constant([[1., 0.], [0., 1.]], dtype=tf.float64)
        )
        true_integral = tfp.distributions.MultivariateNormalFullCovariance(
            loc=prior.loc,
            covariance_matrix=prior.covariance() + integrand.covariance()
        ).prob(integrand.loc)
    elif prior_type == 'uniform':
        prior = tfp.distributions.Uniform(
            low=tf.constant([0., 0.], dtype=tf.float64),
            high=tf.constant([1., 1.], dtype=tf.float64),
        )
        def cdf(x):
            return 0.5 * (
                1 + tf.math.erf((x - integrand.loc) / tf.math.sqrt(2 * tf.linalg.diag_part(integrand.covariance())))
            )
        true_integral = tf.reduce_prod(cdf(prior.high) - cdf(prior.low))
    X = prior.sample(500)
    Y = tf.reshape(integrand.prob(X), (-1, 1))
    noise_std = tf.math.reduce_std(Y) / 10
    Y = Y + noise_std * tf.random.normal(Y.shape, dtype=default_float())
    kernel = gpflow.kernels.SquaredExponential(lengthscales=tf.ones(X.shape[1]))
    # Set up and train integrand model
    if warping is None:
        observation_transformer = StandardTransformer(Y)
        Y_ = observation_transformer.transform(Y)
        gpflow_model = gpflow.models.GPR((X, Y_), kernel)
        class PIFlowModel(DataTransformMixin, GaussianProcessRegression):
            pass
    elif warping == 'WSABI-L':
        if prior_type == 'uniform':
            old_prior = prior
            prior = tfp.distributions.MultivariateNormalFullCovariance(
                loc=tf.constant([0.5, 0.5], dtype=tf.float64),
                covariance_matrix=tf.constant([[1., 0.], [0., 1.]], dtype=tf.float64)
            )
            new_samples = 400
            X2 = prior.sample(new_samples)
            Y2 = tf.reshape(integrand.prob(X2), (-1, 1))
            Y2 = Y2 + noise_std * tf.random.normal(Y2.shape, dtype=default_float())
            X = tf.concat((X[:-new_samples], X2), axis=0)
            Y = tf.concat((Y[:-new_samples], Y2), axis=0)
            Y = Y * tf.reshape(tf.reduce_prod(old_prior.prob(X), axis=1), (-1, 1)) / tf.reshape(prior.prob(X), (-1, 1))
        observation_transformer = MinMaxTransformer(Y)
        Y_ = observation_transformer.transform(Y)
        gpflow_model = WSABI_L_GPR((X, Y_), kernel)
        PIFlowModel = WarpedGaussianProcessRegression
    elif warping == 'MMLT':
        observation_transformer = MinMaxTransformer(Y)
        Y_ = observation_transformer.transform(Y)
        gpflow_model = MMLT_GPR((X, Y_), kernel)
        PIFlowModel = WarpedGaussianProcessRegression
    set_model_priors(gpflow_model)
    initialise_hyperparameters(gpflow_model)
    model = PIFlowModel(gpflow_model, observation_transformer=observation_transformer)
    integrand_model = IntegrandModel(prior, model)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(gpflow_model.training_loss, gpflow_model.trainable_variables)
    gpflow.utilities.print_summary(gpflow_model)
    # Compute integral.
    int_mean, int_var = integrand_model.integral_posterior()
    # Assert approximation is close to true value.
    tf.debugging.assert_near(true_integral, int_mean, rtol=0.1)


# if __name__ == '__main__':
    # test_integrand_model(warping=None, prior_type='gaussian')
    # test_integrand_model(warping=None, prior_type='uniform')
<<<<<<< HEAD
    # test_integrand_model(warping='WSABI-L', prior_type='gaussian')
=======
    test_integrand_model(warping='WSABI-L', prior_type='gaussian')
>>>>>>> 2407d58813fd9118145379ac2e14c3fa81a90d4e
    # test_integrand_model(warping='WSABI-L', prior_type='uniform')
    # test_integrand_model(warping='MMLT', prior_type='gaussian')
    # test_integrand_model(warping='MMLT', prior_type='uniform')
