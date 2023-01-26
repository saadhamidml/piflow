import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np
from math import pi

from gpflow.config import default_float

from gpmaniflow.utils import binomial_coef, factorial
from trieste.space import Box


class GenzFamily():
    def __init__(self, dimension = 1, seed = None, a = None):
        self._dimension = dimension
        self.domain = Box([0.] * self._dimension, [1.] * self._dimension)
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
    
    def posterior_samples(self, num_samples: int, sample_factor: int = 16) -> tf.Tensor:
        prior = tfp.distributions.Uniform(low=self.domain.lower, high=self.domain.upper)
        success = False
        while not success:
            samples_ = prior.sample(int(num_samples * sample_factor))
            prior_probs = prior.prob(samples_)
            if prior_probs.ndim > 1:
                prior_probs = tf.math.reduce_prod(prior_probs, axis=-1)
            secondary_samples = tfp.distributions.Uniform(
                low=tf.zeros_like(prior_probs), high=prior_probs
            ).sample()
            posterior_probs = tf.squeeze(self(samples_)) * prior_probs / self.integral_value
            mask = posterior_probs < secondary_samples
            try:
                samples = tf.concat((samples, samples_[mask]))
            except NameError as e:
                samples = samples_[mask]
            success = len(samples) >= num_samples
        return samples[:num_samples]

    def __call__(self, x: tf.Tensor, return_log: bool = False) -> tf.Tensor:
        raise NotImplementedError


class ContinuousFamily(GenzFamily):
    def __init__(self, dimension = 1, seed = None, a = None):
        super().__init__(dimension, seed)
        # if seed is None:
        #     self.u = tf.constant([0.5] * self._dimension, dtype = default_float())
        # else:
        self.u = tf.random.uniform(shape = [1, self._dimension], dtype = default_float()) 
        if a is None: 
            self.a = tf.constant([150/self._dimension ** 3] * self._dimension, dtype = default_float())
        else:
            self.a = a
        
        self.integral_value = tf.reduce_prod(2 / self.a * (1 - 0.5*tf.math.exp(-self.a * (1 - self.u)) - 0.5*tf.math.exp(-self.a*self.u)))
        
    def __call__(self, x: tf.Tensor, return_log: bool = False) -> tf.Tensor:
        f = self.a * tf.math.abs(x - self.u)
        f = tf.reduce_sum(f, axis = 1, keepdims = True)
        if return_log:
            return -f
        else:
            return tf.math.exp(-f)


class CornerPeakFamily(GenzFamily):
    def __init__(self, dimension = 1, seed = None, a = None):
        super().__init__(dimension, seed)
        # if seed is None:
        #     self.u = tf.constant([0.5] * self._dimension, dtype = default_float())
        # else:
        self.u = tf.random.uniform(shape = [1, self._dimension], dtype = default_float()) 
        if a is None: 
            self.a = tf.constant([600/self._dimension ** 3] * self._dimension, dtype = default_float())
        else:
            self.a = a
        if a is None:
            vec = range(self._dimension)
            bc = binomial_coef(np.array([self._dimension]), vec)
            new_vec = bc * (-1) ** (vec + np.array([self._dimension])) * (1 + (np.array([self._dimension]) - vec)*self.a[0]) ** (-1)
            self.integral_value = (1 / (factorial(self._dimension) * self.a[0] ** self._dimension)) * tf.reduce_sum(new_vec)
        else:
            print("Integral value not analytical for this choice of a")
        
    def __call__(self, x: tf.Tensor, return_log: bool = False) -> tf.Tensor:
        f = 1 + tf.reduce_sum(self.a * x, axis = 1, keepdims = True)
        if return_log:
            return (-self._dimension - 1) * tf.math.log(f)
        else:
            return f ** (-self._dimension - 1)


class GaussianPeakFamily(GenzFamily):
    def __init__(self, dimension = 1, seed = None, a = None):
        super().__init__(dimension, seed)
        # if seed is None:
        #     self.u = tf.constant([0.5] * self._dimension, dtype = default_float())
        # else:
        self.u = tf.random.uniform(shape = [1, self._dimension], dtype = default_float()) 
        if a is None: 
            self.a = tf.constant([100/self._dimension ** 2] * self._dimension, dtype = default_float())
        else:
            self.a = a
        G = tfp.distributions.Normal(loc = tf.cast(0.0, default_float()), scale = tf.cast(1.0, default_float()))
        vec = 1 / self.a * ( G.cdf(2 ** 0.5 * self.a * (1-self.u)) - G.cdf(- 2 ** 0.5 * self.a * self.u))
        self.integral_value = pi ** (self._dimension / 2) * tf.reduce_prod(vec)

    def __call__(self, x: tf.Tensor, return_log: bool = False) -> tf.Tensor:
        f = tf.reduce_sum( self.a ** 2 * (x - self.u) ** 2, axis = 1, keepdims = True)
        if return_log:
            return -f
        else:
            return tf.math.exp(-f)


if __name__ == '__main__':
    M = ContinuousFamily(dimension = 4)
    C = CornerPeakFamily(dimension = 2)
    C = CornerPeakFamily(dimension = 3)
    C = CornerPeakFamily(dimension = 10)
    G = GaussianPeakFamily(dimension = 3)
