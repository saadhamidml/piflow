import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np
from math import pi

from gpflow.config import default_float

from gpmaniflow.utils import binomial_coef, factorial
from trieste.space import Box

class ContinuousFamily():
    def __init__(self, dimension = 1, seed_u = None, a = None):
        self._dimension = dimension
        
        self.domain = Box([0.] * self._dimension, [1.] * self._dimension)
        if seed_u is None:
            self.u = tf.constant([0.5] * self._dimension, dtype = default_float())
        else:
            self.u = tf.random.uniform(shape = [1, self._dimension], dtype = default_float()) 
        if a is None: 
            self.a = tf.constant([150/self._dimension ** 3] * self._dimension, dtype = default_float())
        else:
            self.a = a
        
        self.integral_value = tf.reduce_prod(2 / self.a * (1 - 0.5*tf.math.exp(-self.a * (1 - self.u)) - 0.5*tf.math.exp(-self.a*self.u)))
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        f = self.a * tf.math.abs(x - self.u)
        f = tf.reduce_sum(f, axis = 1, keepdims = True)
        f = tf.math.exp(-f)
        return f

class CornerPeakFamily():
    def __init__(self, dimension = 1, seed_u = None, a = None):
        self._dimension = dimension
        self.domain = Box([0.] * self._dimension, [1.] * self._dimension)
        
        if seed_u is None:
            self.u = tf.constant([0.5] * self._dimension, dtype = default_float())
        else:
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
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        f = 1 + tf.reduce_sum(self.a * x, axis = 1, keepdims = True)
        f = f ** (-self._dimension - 1)
        return f

class GaussianPeakFamily():
    def __init__(self, dimension = 1, seed_u = None, a = None):
        self._dimension = dimension
        self.domain = Box([0.] * self._dimension, [1.] * self._dimension)
        
        if seed_u is None:
            self.u = tf.constant([0.5] * self._dimension, dtype = default_float())
        else:
            self.u = tf.random.uniform(shape = [1, self._dimension], dtype = default_float()) 
        if a is None: 
            self.a = tf.constant([100/self._dimension ** 2] * self._dimension, dtype = default_float())
        else:
            self.a = a
        G = tfp.distributions.Normal(loc = tf.cast(0.0, default_float()), scale = tf.cast(1.0, default_float()))
        vec = 1 / self.a * ( G.cdf(2 ** 0.5 * self.a * (1-self.u)) - G.cdf(- 2 ** 0.5 * self.a * self.u))
        self.integral_value = pi ** (self._dimension / 2) * tf.reduce_prod(vec)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        f = tf.reduce_sum( self.a ** 2 * (x - self.u) ** 2, axis = 1, keepdims = True)
        f = tf.math.exp(-f)
        return f

if __name__ == '__main__':
    M = ContinuousFamily(dimension = 4)
    C = CornerPeakFamily(dimension = 2)
    C = CornerPeakFamily(dimension = 3)
    C = CornerPeakFamily(dimension = 10)
    G = GaussianPeakFamily(dimension = 3)

