"""A simple example."""
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.space import Box
from trieste.objectives.utils import mk_observer
#from trieste.models.gpflow.models import GaussianProcessRegression
from piflow.acquisition_functions import AcquisitionRule, UncertaintySampling
from piflow.beziers import BezierProcessRegression

#from piflow.models import WSABI_L_GPR
from gpmaniflow.models import LogBezierProcess
from piflow.probabilistic_integrator import ProbabilisticIntegrator

from piflow.objectives.genz import ContinuousFamily

# Define problem
#integrand = tfp.distributions.MultivariateNormalFullCovariance(
#    loc=tf.constant([1., 1.], dtype=tf.float64),
#    covariance_matrix=tf.constant([[2., 0.], [0., 2.]], dtype=tf.float64)
#)

# Wrap for trieste.
integrand = ContinuousFamily(seed = 0)
#def eval_integrand(x: tf.Tensor) -> tf.Tensor:
#    return tf.expand_dims(integrand.prob(x), -1)
observer = mk_observer(integrand)

search_space = integrand.domain#Box(lower=[0.], upper=[1.])
#print(search_space.dtype)
prior = tfp.distributions.Uniform(low = tf.cast(0.0, tf.float64), high = tf.cast(1.0, tf.float64))#, dtype = tf.float64)
# Sample initial datapoints.
num_initial_points = 2
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

# Set up the model.
#gpflow_model = WSABI_L_GPR(initial_data.astuple(), kernel=gpflow.kernels.SquaredExponential())
logbezier_model = LogBezierProcess(input_dim = 1, likelihood = gpflow.likelihoods.Gaussian(), num_data = num_initial_points)
model = BezierProcessRegression(logbezier_model)

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
