import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.models.gpflow.models import GaussianProcessRegression
import pytest

from piflow.models import WSABI_L_GPR
from piflow.probabilistic_integrator import IntegrandModel


# @pytest.mark.parametrize('warping', [None, 'WSABI-L'])
def test_integrand_model(warping: str = None):
    # Generate data from a Gaussian.
    integrand = tfp.distributions.MultivariateNormalFullCovariance(
        loc=tf.constant([1., 1.], dtype=tf.float64),
        covariance_matrix=tf.constant([[2., 0.], [0., 2.]], dtype=tf.float64)
    )
    prior = tfp.distributions.MultivariateNormalFullCovariance(
        loc=tf.constant([0., 0.], dtype=tf.float64),
        covariance_matrix=tf.constant([[1., 0.], [0., 1.]], dtype=tf.float64)
    )
    X = integrand.sample(500)
    Y = tf.reshape(integrand.prob(X), (-1, 1))
    # Set up and train integrand model
    if warping is None:
        gpflow_model = gpflow.models.GPR((X, Y), gpflow.kernels.SquaredExponential())
    elif warping == 'WSABI-L':
        gpflow_model = WSABI_L_GPR((X, Y), gpflow.kernels.SquaredExponential())
    model = GaussianProcessRegression(gpflow_model)
    integrand_model = IntegrandModel(prior, model)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(gpflow_model.training_loss, gpflow_model.trainable_variables)
    # Compute integral.
    int_mean, int_var = integrand_model.integral_posterior()
    # Assert approximation is close to true value.
    true_integral = tfp.distributions.MultivariateNormalFullCovariance(
        loc=prior.loc,
        covariance_matrix=prior.covariance() + integrand.covariance()
    ).prob(integrand.loc)
    tf.debugging.assert_near(true_integral, int_mean)
    

if __name__ == '__main__':
    #test_integrand_model()
    test_integrand_model(warping='WSABI-L')
