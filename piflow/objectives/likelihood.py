"""Non-negative integrands -- synthetic or real ML likelihoods."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow import default_float


class GaussianMixtureSyntheticLikelihood():
    """Synthetic likelihood built using a GMM."""

    def __init__(self, dimension: int = 1, seed: int = None) -> None:
        """Initialise. self.integral_value assumes integration against
        uniform measure on unit hypercube.
        """
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        num_components = np.random.randint(5, 14)
        means = np.random.rand(num_components, dimension).tolist()
        scales = (np.random.randint(
            11, 19, size=(num_components, dimension)
        ) / 100).tolist()
        weights = np.random.dirichlet(np.ones(num_components)).tolist()
        self.weights = tf.constant(weights, dtype=default_float())
        self.means = tf.constant(means, dtype=default_float())
        self.scales = tf.constant(scales, dtype=default_float())
        # Weighted sum of the integrals for each Gaussian.
        dists = tfp.distributions.Normal(self.means, self.scales)
        cdf_diffs = dists.cdf(tf.ones_like(self.means)) - dists.cdf(tf.zeros_like(self.means))
        self.integral_value = tf.math.reduce_sum(self.weights * tf.math.reduce_prod(cdf_diffs, -1))
    
    def posterior_samples(self, num_samples: int, sample_factor: int = 16) -> tf.Tensor:
        indices = tfp.distributions.Categorical(probs=self.weights).sample(num_samples)
        indices = tf.sort(indices)  # So count order is correct
        _, _, counts = tf.unique_with_counts(indices)
        samples = []
        for i, (m, s) in enumerate(zip(self.means, self.scales)):
            samples.append(tfp.distributions.TruncatedNormal(
                m, s, tf.zeros_like(m), tf.ones_like(m)
            ).sample(counts[i]))
        return tf.concat(samples, 0)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Likelihood value at x.
        
        :param x: Query locations, shape [N, D].
        :return: The likelihood values at x, shape [N, 1].
        """
        dists = tfp.distributions.Normal(self.means, self.scales)
        weighted_log_probs = (
            tf.reduce_sum(dists.log_prob(tf.expand_dims(x, 1)), -1) + tf.math.log(self.weights)
        )
        return tf.exp(tf.math.reduce_logsumexp(weighted_log_probs, axis=-1, keepdims=True))


class BayesianLogisticRegressionLikelihood():
    """Likelihood for a Bayesian Logistic Regression model."""

    def __init__(
        self,
        prior: tfp.distributions.Distribution,
        train_inputs: tf.Tensor = None,
        train_targets: tf.Tensor = None,
        dimension: int = None,
        num_data: int = 1000,
        seed: int = None,
        num_mc_samples: int = 10000,
        integral_value: float = None
    ) -> None:
        """Initialise. self.integral_value assumes integration against
        a uniform measure on unit hypercube (shifted to be centred on the
        origin).
        """
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        # prior = tfp.distributions.Uniform(
        #     low=tf.cast([-0.5] * dimension, tf.float64),
        #     high=tf.cast([0.5] * dimension, tf.float64)
        # )
        if train_inputs is None:
            if dimension is None:
                raise TypeError('One of train_inputs or dimension must be specified.')
            self.weights = tf.expand_dims(prior.sample(), -1)  # [D, 1]
            self.train_inputs = tf.random.normal((num_data, dimension), dtype=default_float())  # [N, D]
            bernoulli_logits = self.train_inputs @ self.weights  # [N, 1]
            # [N, 1]
            self.train_targets = tfp.distributions.Bernoulli(logits=bernoulli_logits).sample() 
        else:
            self.train_inputs = train_inputs
            self.train_targets = train_targets
        if integral_value is None:
            for i in range(num_mc_samples):
                mc_samples = prior.sample(num_mc_samples)
            integral_value = self(mc_samples).mean().item()
        self.integral_value = integral_value
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Likelihood value at x.
        
        :param x: Query locations, shape [M, D].
        :return: The likelihood values at x, shape [M, 1].
        """
        logits = tf.squeeze(tf.linalg.matmul(
            tf.expand_dims(self.train_inputs, 0),  # [1, N, D]
            tf.expand_dims(x, -1)  # [M, D, 1]
        ), -1)  # [M, N]
        dist = tfp.distributions.Bernoulli(logits=logits)
        # [M]
        return tf.exp(
            tf.reduce_sum(dist.log_prob(self.train_targets[:, 0]), axis=-1, keepdims=True)
        )


class GaussianProcessRegressionLikelihood():
    """Likelihood for a Gaussian Process Regression model."""

    def __init__(
        self,
        prior: tfp.distributions.Distribution,
        train_inputs: tf.Tensor = None,
        train_targets: tf.Tensor = None,
        dimension: int = None,
        num_data: int = 100,
        seed: int = None,
        num_mc_samples: int = 10000,
        integral_value: float = None
    ) -> None:
        """Initialise. self.integral_value assumes integration against
        a uniform measure on the unit hypercube.
        """
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        # prior = tfp.distributions.Uniform(
        #     low=tf.cast([1.0, 0.01] + [1e-6] * dimension, tf.float64),
        #     high=tf.cast([10.0, 0.1] + [1.0] * dimension, tf.float64)
        # )
        kernel = gpflow.kernels.Matern12(lengthscales=tf.ones(dimension, dtype=default_float()))
        if train_inputs is None:
            if dimension is None:
                raise TypeError('One of train_inputs or dimension must be specified.')
            # [signal_variance, noise_variance, lengthscales]
            self.hyperparameters = self.prior.sample()
            # [N, D]
            self.train_inputs = tf.random.normal((num_data, dimension), dtype=default_float())
            kernel.variance.assign(self.hyperparameters[0])
            kernel.lengthscales.assign(self.hyperparameters[2:])
            covar = kernel(self.train_inputs) + self.hyperparameters[1] * tf.eye(
                num_data, dtype=default_float()
            )
            self.train_targets = tf.expand_dims(tfp.distributions.MultivariateNormalFullCovariance(
                loc=tf.zeros(num_data, dtype=default_float()), covariance_matrix=covar
            ).sample(), -1)
        else:
            self.hyperparameters = None
            self.train_inputs = train_inputs
            self.train_targets = train_targets
        self.model = gpflow.models.GPR((self.train_inputs, self.train_targets), kernel=kernel)
        if self.hyperparameters is not None:
            self.model.likelihood.variance.assign(self.hyperparameters[1])
        if integral_value is None:
            for i in range(num_mc_samples):
                mc_samples = prior.sample(num_mc_samples)
            integral_value = self(mc_samples).mean().item()
        self.integral_value = integral_value
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Likelihood value at x.
        
        :param x: Query locations, shape [N, D].
        :return: The likelihood values at x, shape [N, 1].
        """
        lmls = []
        for xi in x:
            self.model.kernel.variance.assign(xi[0])
            self.model.kernel.lengthscales.assign(xi[2:])
            self.model.likelihood.variance.assign(xi[1])
            lmls.append(self.model.log_marginal_likelihood())
        return tf.reshape(tf.exp(lmls), (-1, 1))
