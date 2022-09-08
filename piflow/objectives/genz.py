import tensorflow as tf

from gpflow.config import default_float
from trieste.space import Box

class ContinuousFamily():
    def __init__(self, dimension = 1, seed = 0):
        self._dimension = dimension
        self._seed = tf.random.set_seed(seed)
        self.domain = Box([0.] * self._dimension, [1.] * self._dimension)
        
        self.a = tf.random.uniform(shape = [1, self._dimension], dtype = default_float())
        self.u = tf.random.uniform(shape = [1, self._dimension], dtype = default_float())
        self.integral_value = tf.reduce_prod(2 / self.a * (1 - 0.5*tf.math.exp(-self.a * (1 - self.u)) - tf.math.exp(-self.a*self.u)))
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        f = self.a * tf.math.abs(x - self.u)
        f = tf.reduce_sum(f, axis = 1, keepdims = True)
        f = tf.math.exp(-f)
        return f
""" 
class CornerPeakFamily():
    def __init__(self, dimension = 1, seed = 0):
        self._dimension = dimension
        self._seed = seed
        self.domain = Box([0.] * self._dimension, [1.] * self._dimension)
        self.integral_value = 
        
    def __call__(x: tf.Tensor) -> tf.Tensor:
        
        return f

class GaussianPeakFamily():
    def __init__(self, dimension = 1, seed = 0):
        self._dimension = dimension
        self._seed = seed
        self.domain = Box([0.] * self._dimension, [1.] * self._dimension)
        self.integral_value = 
        
    def __call__(x: tf.Tensor) -> tf.Tensor:
        
        return f
"""
