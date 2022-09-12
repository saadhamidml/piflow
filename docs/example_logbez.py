"""A simple example."""
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.space import Box
from trieste.objectives.utils import mk_observer
#from trieste.models.gpflow.models import GaussianProcessRegression
from piflow.acquisition_functions import AcquisitionRule, UncertaintySampling
from piflow.models.beziers import BezierProcessRegression
from piflow.models.transforms import MinMaxTransformer

#from piflow.models import WSABI_L_GPR
from gpmaniflow.models import LogBezierProcess
from piflow.probabilistic_integrator import ProbabilisticIntegrator

from piflow.objectives.genz import ContinuousFamily

# Set up and wrap for trieste.
integrand = ContinuousFamily(seed = 0)
observer = mk_observer(integrand)
# search_space is for acquisition optimisation only. We need to set the bounds slightly narrower
# than the integrand domain because the gradient becomes NaN if the acquisition function is
# evaluated at the corners, causing the acquisition function optimisation to fail.
# search_space = integrand.domain
search_space = Box(lower=[0 + 1e-6], upper=[1 - 1e-6])
prior = tfp.distributions.Uniform(low = tf.cast(0.0, tf.float64), high = tf.cast(1.0, tf.float64))
# Sample initial datapoints.
num_initial_points = 2
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

# Set up the model.
_, Y = initial_data.astuple()
observation_transformer = MinMaxTransformer(Y)
logbezier_model = LogBezierProcess(input_dim = 1, orders = 3, likelihood = gpflow.likelihoods.Gaussian(), num_data = num_initial_points)
model = BezierProcessRegression(logbezier_model, observation_transformer=observation_transformer)

# Set up the acquisition function.
acquisition_rule = AcquisitionRule(builder=UncertaintySampling())

# Run the integrator.
num_steps = 10
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
print("Actual integral:", integrand.integral_value)
