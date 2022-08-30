import logging
from typing import Tuple, Mapping
import math
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import trieste

from .models import WSABI_L_GPR


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
        posterior = self._model._model.posterior()  # Populate caches.
        if self._integral_mean is None:
            int_mean = integral_mean(
                self._prior,
                self._model._model,
                self._model._model.kernel,  # Included in function signature for multiple dispatch.
                posterior=posterior
            )
            self._integral_mean = int_mean
        else:
            int_mean = self._integral_mean
        if self._integral_variance is None:
            int_variance = integral_variance(
                self._prior,
                self._model._model,
                self._model._model.kernel,  # Included in function signature for multiple dispatch.
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


# RBF kernel, Gaussian prior
@integral_mean.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    gpflow.models.GPR,
    gpflow.kernels.RBF
)
def _rbf_gaussian_mean(
    prior: tfp.distributions.MultivariateNormalFullCovariance,
    model: gpflow.models.GPR,
    kernel: gpflow.kernels.RBF,
    posterior: gpflow.posteriors.GPRPosterior = None
):
    """Posterior mean over integral for a GP with an RBF kernel against
    a Gaussian prior.
    """
    posterior = model.posterior() if posterior == None else posterior
    kernel_integral = tf.reshape(_rbf_gaussian_kernel_integral(prior, model), (1, -1))
    return tf.squeeze(kernel_integral @ posterior.cache[0] @ model.data[1])

@integral_variance.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    gpflow.models.GPR,
    gpflow.kernels.RBF
)
def _rbf_gaussian_var(
    prior: tfp.distributions.MultivariateNormalFullCovariance,
    model: gpflow.models.GPR,
    kernel: gpflow.kernels.RBF,
    posterior: gpflow.posteriors.GPRPosterior = None
):
    """Posterior variance over integral for a GP with an RBF kernel
    against a Gaussian prior.
    """
    posterior = model.posterior() if posterior == None else posterior
    dbl_kernel_integral = _rbf_gaussian_double_kernel_integral(prior, model)
    kernel_integral = tf.reshape(_rbf_gaussian_kernel_integral(prior, model), (1, -1))
    return (
        dbl_kernel_integral - kernel_integral @ posterior.cache[0] @ tf.transpose(kernel_integral)
    )

def _rbf_gaussian_kernel_integral(
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

def _rbf_gaussian_double_kernel_integral(
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
    dbl_int_z = double_integral @ Z
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
