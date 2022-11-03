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
        means = (100 * (np.random.rand(num_components, dimension) - 0.5)).tolist()
        scales = np.random.randint(
            11, 19, size=(num_components, dimension)
        ).tolist()
        weights = np.random.dirichlet(np.ones(num_components)).tolist()
        self.weights = tf.constant(weights, dtype=default_float())
        self.means = tf.constant(means, dtype=default_float())
        self.scales = tf.constant(scales, dtype=default_float())
        # Weighted sum of the integrals for each Gaussian.
        self.integral_value = np.nan
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Likelihood value at x.
        
        :param x: Query locations, shape [N, D].
        :return: The likelihood values at x, shape [N, 1].
        """
        dists = tfp.distributions.Normal(self.means, self.scales)
        weighted_log_probs = (
            tf.reduce_sum(dists.log_prob(tf.expand_dims(x, 1)), -1) + tf.math.log(self.weights)
        )
        return tf.math.reduce_logsumexp(weighted_log_probs, axis=-1, keepdims=True)


class BayesianLogisticRegressionLikelihood():
    """Likelihood for a Bayesian Logistic Regression model."""

    def __init__(
        self,
        train_inputs: tf.Tensor = None,
        train_targets: tf.Tensor = None,
        dimension: int = None,
        num_data: int = 1000,
        seed: int = None
    ) -> None:
        """Initialise. self.integral_value assumes integration against
        a uniform measure on unit hypercube (shifted to be centred on the
        origin).
        """
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        if train_inputs is None:
            if dimension is None:
                raise TypeError('One of train_inputs or dimension must be specified.')
            prior = tfp.distributions.Uniform(
                low=tf.cast([-0.5] * dimension, tf.float64),
                high=tf.cast([0.5] * dimension, tf.float64)
            )
            self.weights = tf.expand_dims(prior.sample(), -1)  # [D, 1]
            self.train_inputs = tf.random.normal((num_data, dimension), dtype=default_float())  # [N, D]
            bernoulli_logits = self.train_inputs @ self.weights  # [N, 1]
            # [N, 1]
            self.train_targets = tfp.distributions.Bernoulli(logits=bernoulli_logits).sample() 
        else:
            raise NotImplementedError
        self.integral_value = np.nan  # TODO: Use MC to establish.
        
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
        train_inputs: tf.Tensor = None,
        train_targets: tf.Tensor = None,
        dimension: int = None,
        num_data: int = 100,
        seed: int = None
    ) -> None:
        """Initialise. self.integral_value assumes integration against
        a uniform measure on the unit hypercube.
        """
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        kernel = gpflow.kernels.RBF(lengthscales=tf.ones(dimension, dtype=default_float()))
        if train_inputs is None:
            if dimension is None:
                raise TypeError('One of train_inputs or dimension must be specified.')
            prior = tfp.distributions.Uniform(
                low=tf.cast([1.0, 0.01] + [1e-6] * dimension, tf.float64),
                high=tf.cast([10.0, 0.1] + [1.0] * dimension, tf.float64)
            )
            # [signal_variance, noise_variance, lengthscales]
            self.hyperparameters = prior.sample()
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
        self.integral_value = np.nan  # TODO: Use MC to establish.
        
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
