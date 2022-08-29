from copy import deepcopy
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

from piflow.probabilistic_integrator import IntegrandModel

def test_integrand_model():
    # Generate data from a Gaussian.
    integrand = tfp.distributions.MultivariateNormalFullCovariance(
        loc=tf.constant([0., 0.]),
        covariance_matrix=tf.constant([[1., 0.], [0., 1.]])
    )
    prior = deepcopy(integrand)
    X = integrand.sample(500)
    Y = integrand.prob(X)

    surrogate = gpflow.models.GPR((X, Y), gpflow.kernels.SquaredExponential())
    integrand_model = IntegrandModel(prior, surrogate)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(surrogate.training_loss, surrogate.trainable_variables)

    int_mean, int_var = integrand_model.integral_posterior()
    true_integral = tfp.distributions.MultivariateNormalFullCovariance(
        loc=prior.loc,
        covariance_matrix=prior.covariance_matrix + integrand.covariance_matrix
    ).prob(integrand.loc)

    tf.debugging.assert_near(true_integral, int_mean)


if __name__ == '__main__':
    test_integrand_model()