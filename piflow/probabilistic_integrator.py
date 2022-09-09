from distutils.dist import Distribution
import logging
from typing import Tuple, Mapping
import math
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import trieste

from .models import WSABI_L_GPR, MMLT_GPR
from gpmaniflow.models import LogBezierProcess

NoneType = type(None)

logger = logging.getLogger(__name__)

integral_mean = gpflow.utilities.Dispatcher('integral_mean')
integral_variance = gpflow.utilities.Dispatcher('integral_variance')


class IntegrandModel():
    """The object resulting from probabilistic integration."""

    def __init__(
        self,
        prior: tfp.distributions.Distribution,
        model: trieste.models.interfaces.TrainableProbabilisticModel
    ) -> None:
        self._prior = prior
        self._model = model
        self._integral_mean = None
        self._integral_variance = None

    def integral_posterior(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute and return the integral posterior.
        
        :return: Integral mean, scalar; integral standard deviation,
            scalar.
        """
        try:
            posterior = self._model._model.posterior()  # Populate caches.
        except AttributeError as e:
            # LogBezierProcess does not have kernel or caching.
            posterior = None
        if self._integral_mean is None:
            int_mean = integral_mean(
                self._prior,
                self._model._model,
                self._model.get_kernel(),  # Included in function signature for multiple dispatch.
                posterior=posterior
            )
            self._integral_mean = int_mean
        else:
            int_mean = self._integral_mean
        if self._integral_variance is None:
            int_variance = integral_variance(
                self._prior,
                self._model._model,
                self._model.get_kernel(),  # Included in function signature for multiple dispatch.
                posterior=posterior
            )
            self._integral_variance = int_variance
        else:
            int_variance = self._integral_variance
        return int_mean, int_variance


class ProbabilisticIntegrator():
    """Performs probabilistic integration."""

    def __init__(
        self,
        observer: trieste.observer.Observer,
        prior: tfp.distributions.Distribution,
        search_space: trieste.space.SearchSpace
    ) -> None:
        self._observer = observer
        self._prior = prior
        self._search_space = search_space

    def integrate(
        self,
        num_steps: int,
        datasets: Mapping[str, trieste.data.Dataset],
        models: Mapping[
            str, trieste.models.interfaces.TrainableProbabilisticModel
        ],
        acquistion_rule: trieste.acquisition.rule.AcquisitionRule,
        fit_initial_model: bool = True
    ):
        """Run the acquisition loop to collect samples for
        probabilistic integration.
        
        :param num_steps: Number of acquistion steps.
        :param dataset: The observations of the integrand.
        :param model: The model over the integrand.
        :param acquistion_rule: The acquisition rule.
        :return: IntegrandModel object.
        """
        if fit_initial_model:
            for tag, model in models.items():
                dataset = datasets[tag]
                model.update(dataset)
                model.optimize(dataset)
        for step in range(1, num_steps + 1):
            logger.info(f'PI acquisition {step}/{num_steps}')
            # Make an acquisition.
            query_points = acquistion_rule.acquire(
                self._search_space, models, datasets=datasets, prior=self._prior
            )
            observer_output = self._observer(query_points)
            tagged_output = (
                observer_output
                if isinstance(observer_output, Mapping)
                else {'INTEGRAND': observer_output}
            )
            datasets = {tag: datasets[tag] + tagged_output[tag] for tag in tagged_output}
            # Update and optimise the model.
            for tag, model in models.items():
                dataset = datasets[tag]
                model.update(dataset)
                model.optimize(dataset)
        integrand_model = IntegrandModel(self._prior, model)
        _ = integrand_model.integral_posterior()  # Populate cache.
        return  integrand_model
    

# Ordinary Bayesian Quadrature
def _bayesian_quadrature_mean(
    kernel_integral: tf.Tensor,
    model: gpflow.models.GPR,
    posterior: gpflow.posteriors.GPRPosterior = None
):
    kernel_integral = tf.reshape(kernel_integral, (1, -1))
    posterior = model.posterior() if posterior == None else posterior
    return tf.squeeze(kernel_integral @ posterior.cache[0] @ model.data[1])

def _bayesian_quadrature_var(
    kernel_integral: tf.Tensor,
    dbl_kernel_integral: tf.Tensor,
    model: gpflow.models.GPR,
    posterior: gpflow.posteriors.GPRPosterior = None
):
    kernel_integral = tf.reshape(kernel_integral, (1, -1))
    posterior = model.posterior() if posterior == None else posterior
    return (
        dbl_kernel_integral - kernel_integral @ posterior.cache[0] @ tf.transpose(kernel_integral)
    )


# Gaussian prior, RBF kernel, Ordinary GP
@integral_mean.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    gpflow.models.GPR,
    gpflow.kernels.RBF
)
def _gaussian_rbf_mean(
    prior: tfp.distributions.MultivariateNormalFullCovariance,
    model: gpflow.models.GPR,
    kernel: gpflow.kernels.RBF,
    posterior: gpflow.posteriors.GPRPosterior = None
):
    """Posterior mean over integral for a GP with an RBF kernel against
    a Gaussian prior.
    """
    kernel_integral = _gaussian_rbf_kernel_integral(prior, model)
    return _bayesian_quadrature_mean(kernel_integral, model, posterior)

@integral_variance.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    gpflow.models.GPR,
    gpflow.kernels.RBF
)
def _gaussian_rbf_var(
    prior: tfp.distributions.MultivariateNormalFullCovariance,
    model: gpflow.models.GPR,
    kernel: gpflow.kernels.RBF,
    posterior: gpflow.posteriors.GPRPosterior = None
):
    """Posterior variance over integral for a GP with an RBF kernel
    against a Gaussian prior.
    """
    dbl_kernel_integral = _gaussian_rbf_double_kernel_integral(prior, model)
    kernel_integral = _gaussian_rbf_kernel_integral(prior, model)
    return _bayesian_quadrature_var(kernel_integral, dbl_kernel_integral, model, posterior)

def _gaussian_rbf_kernel_integral(
    prior: tfp.distributions.MultivariateNormalFullCovariance,
    model: gpflow.models.GPR,
) -> tf.Tensor:
    X, _ = model.data  # [N, D]
    kernel = model.kernel
    num_dims = X.shape[1]
    lengthscales_sq = kernel.lengthscales ** 2 * tf.eye(
        num_dims, num_dims, dtype=tf.float64
    )  # [D, D]
    constant = kernel.variance * tf.sqrt(tf.linalg.det(2 * math.pi * lengthscales_sq))
    normal_probs = tfp.distributions.MultivariateNormalFullCovariance(
        loc=prior.loc,
        covariance_matrix=prior.covariance() + lengthscales_sq
    ).prob(X)  # [N]
    return constant * normal_probs  # [N]

def _gaussian_rbf_double_kernel_integral(
    prior: tfp.distributions.MultivariateNormalFullCovariance,
    model: gpflow.models.GPR,
) -> tf.Tensor:
    X, _ = model.data  # [N, D]
    kernel = model.kernel
    num_dims = X.shape[1]
    lengthscales_sq = kernel.lengthscales ** 2 * tf.eye(
        num_dims, num_dims, dtype=tf.float64
    )  # [D, D]
    constant = kernel.variance * tf.sqrt(tf.linalg.det(2 * math.pi * lengthscales_sq))
    combined_cov = prior.covariance() + 2 * lengthscales_sq  # [D, D]
    return constant / tf.sqrt(tf.linalg.det(combined_cov))  # Scalar


# Uniform prior, RBF kernel, Ordinary GP
@integral_mean.register(
    tfp.distributions.Uniform,
    gpflow.models.GPR,
    gpflow.kernels.RBF
)
def _uniform_rbf_mean(
    prior: tfp.distributions.Uniform,
    model: gpflow.models.GPR,
    kernel: gpflow.kernels.RBF,
    posterior: gpflow.posteriors.GPRPosterior = None
):
    kernel_integral = _uniform_rbf_kernel_integral(prior, model)
    return _bayesian_quadrature_mean(kernel_integral, model, posterior)

@integral_variance.register(
    tfp.distributions.Uniform,
    gpflow.models.GPR,
    gpflow.kernels.RBF
)
def _uniform_rbf_var(
    prior: tfp.distributions.Uniform,
    model: gpflow.models.GPR,
    kernel: gpflow.kernels.RBF,
    posterior: gpflow.posteriors.GPRPosterior = None
):
    dbl_kernel_integral = _uniform_rbf_double_kernel_integral(prior, model)
    kernel_integral = _uniform_rbf_kernel_integral(prior, model)
    return _bayesian_quadrature_var(kernel_integral, dbl_kernel_integral, model, posterior)

def _uniform_rbf_kernel_integral(
    prior: tfp.distributions.Uniform,
    model: gpflow.models.GPR
):
    X, _ = model.data
    kernel = model.kernel
    sqrt_tau = 1 / (math.sqrt(2) * kernel.lengthscales)
    constant = math.sqrt(math.pi) / (2 * sqrt_tau * (prior.high - prior.low))
    erf = tf.math.erf(sqrt_tau * (X - prior.low)) - tf.math.erf(sqrt_tau * (X - prior.high))
    return kernel.variance * tf.reduce_prod(constant * erf, axis=1)

def _uniform_rbf_double_kernel_integral(
    prior: tfp.distributions.Uniform,
    model: gpflow.models.GPR
):
    X, _ = model.data
    kernel = model.kernel
    tau = 1 / (2 * kernel.lengthscales ** 2)
    prior_diff = prior.high - prior.low
    constant = math.sqrt(math.pi) / (2 * tf.math.sqrt(tau) * (prior.high - prior.low) ** 2)
    t1 = -2 * prior_diff * tf.math.erf(-tf.math.sqrt(tau) * prior_diff)
    t2 = 2 * tf.math.expm1(-tau * prior_diff ** 2) / tf.math.sqrt(math.pi * tau)
    return kernel.variance * tf.reduce_prod(constant * (t1 + t2))


# WSABI-L, RBF kernel, Gaussian prior
@integral_mean.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    WSABI_L_GPR,
    gpflow.kernels.RBF
)
def _wsabi_mean(
        prior: tfp.distributions.MultivariateNormalFullCovariance,
        model: WSABI_L_GPR,
        kernel: gpflow.kernels.RBF,
        posterior: gpflow.posteriors.GPRPosterior = None
) -> tf.Tensor:
    """Mean of posterior over the integral for a WSABI-L GP with an RBF
    kernel and Gaussian prior.
    """
    posterior = model.posterior() if posterior == None else posterior
    K_inv_z = posterior.cache[0] @ model.data[1]  # [N, 1]
    dbl_kernel_integral = _wsabi_double_kernel_integral(prior, model)
    integral_mean = tf.squeeze(
        model.alpha + 0.5 * (tf.transpose(K_inv_z) @ dbl_kernel_integral @ K_inv_z)
    )
    assert integral_mean >= 0, f'Integral mean negative ({integral_mean})'
    return integral_mean


@integral_variance.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    WSABI_L_GPR,
    gpflow.kernels.RBF
)
def _wsabi_variance(
        prior: tfp.distributions.MultivariateNormalFullCovariance,
        model: WSABI_L_GPR,
        kernel: gpflow.kernels.RBF,
        posterior: gpflow.posteriors.GPRPosterior = None
) -> tf.Tensor:
    """Variance of posterior over the integral for a WSABI-L GP with an
    RBF kernel and Gaussian prior.
    """
    posterior = model.posterior() if posterior == None else posterior
    X, Z = model.data
    K_inv_z = posterior.cache[0] @ Z  # [N, 1]
    triple_integral = _wsabi_triple_kernel_integral(prior, model)
    double_integral = _wsabi_double_kernel_integral(prior, model)
    dbl_int_z = double_integral @ posterior.cache[0] @ Z
    prior_term = tf.transpose(K_inv_z) @ triple_integral @ K_inv_z
    correction_term = tf.transpose(dbl_int_z) @ posterior.cache[0] @ dbl_int_z
    integral_variance = tf.squeeze(prior_term - correction_term)
    assert integral_variance >= 0, f'Integral variance negative. Prior term: {prior_term}, correction Term: {correction_term}'
    return integral_variance


def _wsabi_double_kernel_integral(
        prior: tfp.distributions.MultivariateNormalFullCovariance,
        model: WSABI_L_GPR
) -> tf.Tensor:
    """Integral of k(x, a) * k(x, b) * pi(x) dx for RBF kernel and
    Gaussian prior.
    """
    num_dims = prior.loc.shape[0]
    X, Z = model.data
    lengthscales_sq = model.kernel.lengthscales ** 2 * tf.eye(
        num_dims, num_dims, dtype=tf.float64
    )  # [D, D]
    # D x D
    combined_covar = prior.covariance() + lengthscales_sq / 2
    K_DD = model.kernel.K(X, X)
    constant = model.kernel.variance * 2 ** (-num_dims / 2) * tf.sqrt(
        tf.linalg.det(lengthscales_sq) / tf.linalg.det(combined_covar)
    )
    # N x N
    exp_1 = tf.exp(tf.math.log(K_DD / tf.sqrt(model.kernel.variance)) / 2)
    X_avg = (tf.expand_dims(X, 1) + tf.expand_dims(X, 0)) / 2
    X_avg_offset = X_avg - prior.loc
    # N x N
    exp_2 = tf.exp(
        -0.5 * tf.einsum(
            'ijk,kl,ijl->ij',
            X_avg_offset,
            tf.linalg.inv(combined_covar),
            X_avg_offset
        )
    )
    # N x N
    integral = constant * exp_1 * exp_2
    # assert (integral >= 0).all()
    return integral


def _wsabi_triple_kernel_integral(
        prior: tfp.distributions.MultivariateNormalFullCovariance,
        model: WSABI_L_GPR
) -> tf.Tensor:
    """Integral of k(x, a) * k(x, x') * k(x', b) * pi(x) * pi(x') dx dx'
    for RBF kernel and Gaussian prior.
    """
    num_dims = prior.loc.shape[0]
    X, Z = model.data
    lengthscales_sq = model.kernel.lengthscales ** 2 * tf.eye(
        num_dims, num_dims, dtype=tf.float64
    )  # [D, D]
    # Preliminaries.
    covariance_sum = prior.covariance() + lengthscales_sq
    inv_covar = tf.linalg.inv(prior.covariance())
    inv_lengthscale = tf.linalg.inv(lengthscales_sq)
    inv_precision_sum = tf.linalg.inv(inv_covar + inv_lengthscale)
    combined_covariance = 2 * inv_precision_sum + lengthscales_sq
    # Compute components.
    constant = (
            model.kernel.variance ** (3 / 2)
            * (2 * math.pi) ** num_dims
            * tf.linalg.det(lengthscales_sq) ** (3 / 2)
            / tf.sqrt(tf.linalg.det(combined_covariance))
    )

    # N x D
    distribution = tfp.distributions.MultivariateNormalFullCovariance(
        loc=prior.loc, covariance_matrix=covariance_sum
    )
    # N
    probs = distribution.log_prob(X)
    # N x N
    probs_matrix = tf.exp(tf.expand_dims(probs, 1) + tf.expand_dims(probs, 0))

    # N x D
    location_transform = (
        X @ inv_lengthscale + tf.reshape(prior.loc, (1, -1)) @ inv_covar
    ) @ inv_precision_sum
    # N x N x D
    differences = tf.expand_dims(location_transform, 1) - tf.expand_dims(location_transform, 0)
    # N x N
    exp_term = tf.exp(
        -0.5 * tf.einsum(
            'ijk,kl,ijl->ij',
            differences,
            tf.linalg.inv(combined_covariance),
            differences
        )
    )
    # N x N
    integral = constant * probs_matrix * exp_term
    # assert (integral >= 0).all()
    return integral


# Generic Prior, MMLT
@integral_mean.register(
    tfp.distributions.Distribution,
    MMLT_GPR,
    gpflow.kernels.Kernel
)
def _mmlt_mean(
    prior: tfp.distributions.Distribution,
    model: MMLT_GPR,
    kernel: gpflow.kernels.Kernel,
    posterior: gpflow.posteriors.GPRPosterior = None,
    num_samples: int = None
):
    num_samples = 1000 * model.data[0].shape[1]
    samples = prior.sample(num_samples)  # [N, D]
    mean, _ = model.predict_f(samples, posterior=posterior)
    return tf.math.reduce_mean(mean)

@integral_variance.register(
    tfp.distributions.Distribution,
    MMLT_GPR,
    gpflow.kernels.Kernel
)
def _mmlt_var(
    prior: tfp.distributions.Distribution,
    model: MMLT_GPR,
    kernel: gpflow.kernels.Kernel,
    posterior: gpflow.posteriors.GPRPosterior = None,
    num_samples: int = None
):  
    X, Z = model.data
    posterior = model.posterior() if posterior == None else posterior
    num_samples = 1000 * X.shape[1]
    samples1 = prior.sample(num_samples)
    samples2 = prior.sample(num_samples)
    f_mean1, _ = model.predict_g(samples1, posterior=posterior)
    f_mean2, _ = model.predict_g(samples2, posterior=posterior)
    covar_factors = f_mean1 * f_mean2
    cross_cov = tf.linalg.diag_part(kernel.K(samples1, samples2)) - tf.linalg.diag_part(
        kernel.K(samples1, X) @ posterior.cache[0] @ kernel.K(X, samples2)
    )
    integral_var = tf.reduce_sum(covar_factors * tf.math.expm1(cross_cov)) / num_samples
    assert integral_var >= 0
    return integral_var


# Uniform Prior, Log-Bezier Process
@integral_mean.register(
    tfp.distributions.Uniform,
    LogBezierProcess,
    NoneType
)
def _LogBez_mean(
    prior: tfp.distributions.Uniform,
    model: LogBezierProcess,
    kernel: NoneType,
    posterior: gpflow.posteriors.AbstractPosterior = None
) -> tf.Tensor:
    #return model.BB.integral()
    return model.BB.integral_mean()

@integral_variance.register(
    tfp.distributions.Uniform,
    LogBezierProcess,
    NoneType
)
def _LogBez_variance(
    prior: tfp.distributions.Uniform,
    model: LogBezierProcess,
    kernel: NoneType,
    posterior: gpflow.posteriors.AbstractPosterior = None
) -> tf.Tensor:
    #import warnings
    #warnings.warn(f'Integral Variance not implemented for LogBezierProcess')
    #from numpy import nan
    #return tf.constant(nan)
    return model.BB.integral_variance()
