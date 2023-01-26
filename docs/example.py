"""A simple example."""
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.space import Box
from trieste.objectives.utils import mk_observer
from trieste.models.gpflow.models import GaussianProcessRegression
from piflow.acquisition_functions import AcquisitionRule, UncertaintySampling
from piflow.models.transforms import MinMaxTransformer

from piflow.models.warped import WSABI_L_GPR, MMLT_GPR, WarpedGaussianProcessRegression
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
X, Y = initial_data.astuple()
observation_transformer = MinMaxTransformer(Y)
Y_ = observation_transformer.transform(Y)
gpflow_model = WSABI_L_GPR((X, Y_), kernel=gpflow.kernels.SquaredExponential())
model = WarpedGaussianProcessRegression(
    gpflow_model, observation_transformer=observation_transformer
)

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
