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
        if self._integral_mean is None:
            int_mean = integral_mean(self._prior, self._model)
            self._integral_mean = int_mean
        else:
            int_mean = self._integral_mean
        if self._integral_variance is None:
            int_variance = integral_variance(self._prior, self._model)
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
            # Make an acquisition.
            query_points = acquistion_rule.acquire(self._search_space, models, datasets=datasets)
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
        _ = integrand_model.integral_posterior()  # Populate results.

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
    kernel: gpflow.kernels.RBF
):
    """Posterior mean over integral for a GP with an RBF kernel against
    a Gaussian prior.
    """
    raise NotImplementedError

@integral_mean.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    gpflow.models.GPR,
    gpflow.kernels.RBF
)
def _rbf_gaussian_var(
    prior: tfp.distributions.MultivariateNormalFullCovariance,
    model: gpflow.models.GPR,
    kernel: gpflow.kernels.RBF
):
    """Posterior variance over integral for a GP with an RBF kernel
    against a Gaussian prior.
    """
    raise NotImplementedError


# WSABI-L, RBF kernel, Gaussian prior ( Warped GP) 
@integral_mean.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    WSABI_L_GPR,
    gpflow.kernels.RBF
)
def _wsabi_mean(
        prior: tfp.distributions.MultivariateNormal,
        model: WSABI_L_GPR,
        kernel: gpflow.kernels.RBF
) -> tf.Tensor:
    """Mean of posterior over the integral for a WSABI-L GP with an RBF
    kernel and Gaussian prior.
    """
    # N x 1
    try:
        K_DD_z0 = model.prediction_strategy.mean_cache
    except AttributeError:
        # Hack to get mean_cache populated.
        model.eval()
        mean = tf.tensor([])
        for p in prior:
            mean = tf.cat((mean, p.loc.view(-1)), dim=0)
        model(mean.unsqueeze(0))
        K_DD_z0 = model.prediction_strategy.mean_cache

    dbl_kernel_integral = _wsabi_double_kernel_integral(prior, model)

    integral_mean = (
            model.alpha + 0.5 * (K_DD_z0.T @ dbl_kernel_integral @ K_DD_z0)
    ).item()

    # assert integral_mean >= 0, f'Integral Mean: {integral_mean}'

    return integral_mean


@integral_mean.register(
    tfp.distributions.MultivariateNormalFullCovariance,
    WSABI_L_GPR,
    gpflow.kernels.RBF
)
def _wsabi_variance(
        prior: tfp.distributions.MultivariateNormal,
        model: WSABI_L_GPR,
        kernel: gpflow.kernels.RBF
) -> tf.Tensor:
    """Variance of posterior over the integral for a WSABI-L GP with an
    RBF kernel and Gaussian prior.
    """
    # N x 1
    try:
        K_DD_z0 = model.prediction_strategy.mean_cache
    except AttributeError:
        # Hack to get mean_cache populated.
        model.eval()
        mean = tf.tensor([])
        for p in prior:
            mean = tf.cat((mean, p.loc.view(-1)), dim=0)
        model(mean.unsqueeze(0))
        K_DD_z0 = model.prediction_strategy.mean_cache

    K_DD = model.covar_module(model.train_inputs[0])
    # from gpytf.lazy import DiagLazyTensor
    # K_DD += DiagLazyTensor(model.likelihood.noise.repeat(K_DD.size(0)))

    triple_integral = _wsabi_triple_kernel_integral(prior, model)
    double_integral = _wsabi_double_kernel_integral(prior, model)

    prior_term = K_DD_z0.T @ triple_integral @ K_DD_z0
    correction_term = (
        K_DD_z0.T
        @ K_DD.inv_matmul(
            right_tensor=double_integral, left_tensor=double_integral
        )
        @ K_DD_z0
    )

    integral_variance = (prior_term - correction_term).item()
    # assert integral_variance >= 0, f'Prior Term: {prior_term.item()}, Correction Term: {correction_term.item()}'

    return integral_variance


def _wsabi_kernel_mean(
        prior: tfp.distributions.MultivariateNormalFullCovariance,
        model: WSABI_L_GPR,
        kernel: gpflow.kernels.RBF,
        X: tf.Tensor
) -> tf.Tensor:
    # distribution parameters
    prior_mean = tf.tensor([])
    covar = tf.tensor([])
    for p in prior:
        prior_mean = tf.cat((prior_mean, p.loc.view(-1)), dim=0)
        covar = tf.cat((covar, p.scale.view(-1)), dim=0)
    covar = tf.diag(covar) ** 2
    # kernel parameters
    outputscale = model.covar_module.outputscale.squeeze()
    lengthscale = model.covar_module.base_kernel.lengthscale.squeeze()
    if lengthscale.numel() == 1:
        n_dims = prior_mean.size(0)
        lengthscale = tf.eye(n_dims) * lengthscale ** 2
    else:
        lengthscale = tf.diag_embed(lengthscale) ** 2

    # N x D
    train_inputs = model.train_inputs[0]
    # M
    warped_mean = model.warped_posterior(X).mean
    # N x 1
    try:
        K_DD_z0 = model.prediction_strategy.mean_cache
    except AttributeError:
        # Hack to get mean_cache populated.
        model.eval()
        model(prior_mean.unsqueeze(0))
        K_DD_z0 = model.prediction_strategy.mean_cache
    K_DD = model.covar_module(model.train_inputs[0])
    # from gpytorch.lazy import DiagLazyTensor
    # K_DD += DiagLazyTensor(model.likelihood.noise.repeat(K_DD.size(0)))

    # Partial_triple_integral
    constant = outputscale ** 2 * tf.det(lengthscale) / tf.sqrt(
        tf.det(0.5 * lengthscale + covar) * tf.det(2 * lengthscale)
    )
    # M x N x D
    differences = train_inputs.unsqueeze(0) - X.unsqueeze(1)
    averages = (train_inputs.unsqueeze(0) + X.unsqueeze(1)) / 2
    shifted_averages = averages - prior_mean
    # M x N
    exp_1 = tf.exp(
        -0.25 * tf.einsum(
            'ijk,kl,ijl->ij',
            differences,
            tf.inverse(lengthscale),
            differences
        )
    )
    exp_2 = tf.exp(
        -0.5 * tf.einsum(
            'ijk,kl,ijl->ij',
            shifted_averages,
            tf.inverse(lengthscale / 2 + covar),
            shifted_averages
        )
    )
    # M x N
    half_triple_integral = constant * exp_1 * exp_2
    assert (half_triple_integral >= 0).all()
    prior_term_right = half_triple_integral @ K_DD_z0
    prior_term = warped_mean * prior_term_right

    # N x N
    dbl_kernel_integral = _wsabi_double_kernel_integral(prior, model)
    right_vector = dbl_kernel_integral @ K_DD_z0
    inv_quad = K_DD.inv_matmul(
        right_tensor=right_vector,
        left_tensor=model.covar_module(X, train_inputs).evaluate()
    )
    correction_term = warped_mean * inv_quad

    # M
    integral = prior_term - correction_term

    # assert (integral >= 0).all()

    return integral


def _wsabi_double_kernel_integral(
        prior: tfp.distributions.MultivariateNormalFullCovariance,
        model: WSABI_L_GPR
) -> tf.Tensor:
    """Integral of k(x, a) * k(x, b) * pi(x) dx for RBF kernel and
    Gaussian prior.
    """
    # distribution parameters
    mean = tf.tensor([])
    covar = tf.tensor([])
    for p in prior:
        mean = tf.cat((mean, p.loc.view(-1)), dim=0)
        covar = tf.cat((covar, p.scale.view(-1)), dim=0)
    n_dims = mean.size(0)
    covar = tf.diag(covar) ** 2
    # # kernel parameters
    outputscale = model.covar_module.outputscale.squeeze()
    lengthscale = model.covar_module.base_kernel.lengthscale.squeeze()
    if lengthscale.numel() == 1:
        lengthscale = tf.eye(n_dims) * lengthscale ** 2 #/ 2
    else:
        lengthscale = tf.diag_embed(lengthscale) ** 2 #/ 2
    # D x D
    combined_covar = covar + lengthscale / 2
    K_DD = model.covar_module(model.train_inputs[0]).evaluate()
    constant = outputscale ** 2 * 2 ** (-n_dims / 2) * tf.sqrt(
        tf.det(lengthscale) / tf.det(combined_covar)
    )
    # N x N
    exp_1 = (K_DD / outputscale).log().div(2).exp()
    X_avg = (
                    model.train_inputs[0].unsqueeze(1)
                    + model.train_inputs[0].unsqueeze(0)
            ) / 2
    X_avg_offset = (X_avg - mean)
    # N x N
    exp_2 = tf.exp(
        -0.5 * tf.einsum(
            'ijk,kl,ijl->ij',
            X_avg_offset,
            tf.inverse(combined_covar),
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
    # distribution parameters
    mean = tf.tensor([])
    covar = tf.tensor([])
    for p in prior:
        mean = tf.cat((mean, p.loc.view(-1)), dim=0)
        covar = tf.cat((covar, p.scale.view(-1)), dim=0)
    n_dims = mean.size(0)
    covar = tf.diag(covar) ** 2
    # # kernel parameters
    outputscale = model.covar_module.outputscale.squeeze()
    lengthscale = model.covar_module.base_kernel.lengthscale.squeeze()
    if lengthscale.numel() == 1:
        lengthscale = tf.eye(n_dims) * lengthscale ** 2
    else:
        lengthscale = tf.diag_embed(lengthscale) ** 2

    covariance_sum = covar + lengthscale
    inv_covar = covar.inverse()
    inv_lengthscale = lengthscale.inverse()
    inv_precision_sum = tf.inverse(inv_covar + inv_lengthscale)
    combined_covariance = 2 * inv_precision_sum + lengthscale

    constant = (
            outputscale ** 3
            * (2 * math.pi) ** n_dims
            * tf.det(lengthscale) ** (3 / 2)
            / tf.sqrt(tf.det(combined_covariance))
    )

    # N x D
    data_locations = model.train_inputs[0]
    distribution = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean, covariance_matrix=covariance_sum
    )
    # N
    probs = distribution.log_prob(data_locations)
    # N x N
    probs_matrix = (probs.unsqueeze(1) + probs.unsqueeze(0)).exp()

    # N x D
    location_transform = (
        data_locations @ inv_lengthscale + mean @ inv_covar
    ) @ inv_precision_sum
    # N x N x D
    differences = (
            location_transform.unsqueeze(1) - location_transform.unsqueeze(0)
    )
    # N x N
    exp_term = tf.exp(
        -0.5 * tf.einsum(
            'ijk,kl,ijl->ij',
            differences,
            tf.inverse(combined_covariance),
            differences
        )
    )

    # N x N
    integral = constant * probs_matrix * exp_term
    # assert (integral >= 0).all()

    return integral
