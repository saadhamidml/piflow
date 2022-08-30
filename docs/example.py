"""A simple example."""
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.space import Box
from trieste.objectives.utils import mk_observer
from trieste.models.gpflow.models import GaussianProcessRegression
from piflow.acquisition_functions import AcquisitionRule, UncertaintySampling

from piflow.models import WSABI_L_GPR
from piflow.probabilistic_integrator import ProbabilisticIntegrator

# Define problem
integrand = tfp.distributions.MultivariateNormalFullCovariance(
    loc=tf.constant([1., 1.], dtype=tf.float64),
    covariance_matrix=tf.constant([[2., 0.], [0., 2.]], dtype=tf.float64)
)
prior = tfp.distributions.MultivariateNormalFullCovariance(
    loc=tf.constant([0., 0.], dtype=tf.float64),
    covariance_matrix=tf.constant([[1., 0.], [0., 1.]], dtype=tf.float64)
)

# Wrap for trieste.
def eval_integrand(x: tf.Tensor) -> tf.Tensor:
    return tf.expand_dims(integrand.prob(x), -1)
observer = mk_observer(eval_integrand)
search_space = Box(lower=[-5., -5.], upper=[5., 5.])

# Sample initial datapoints.
num_initial_points = 5
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

# Set up the model.
gpflow_model = WSABI_L_GPR(initial_data.astuple(), kernel=gpflow.kernels.SquaredExponential())
model = GaussianProcessRegression(gpflow_model)

# Set up the acquisition function.
acquisition_rule = AcquisitionRule(builder=UncertaintySampling())

# Run the integrator.
num_steps = 15
integrator = ProbabilisticIntegrator(observer, prior, search_space)
result = integrator.integrate(
    num_steps,
    {'INTEGRAND': initial_data},
    {'INTEGRAND': model},
    acquisition_rule
)
integral_mean, integral_variance = result.integral_posterior()

print(f'Integral Mean: {integral_mean}')
print(f'Integral Variance: {integral_variance}')
