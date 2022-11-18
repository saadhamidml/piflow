from __future__ import annotations
from typing import Mapping

import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.utilities import is_variable
import trieste
from trieste.data import Dataset
from trieste.models.gpflow.interface import GPflowPredictor
from trieste.models.interfaces import TrainableProbabilisticModel
from trieste.models.optimizer import BatchOptimizer, Optimizer
from trieste.models.gpflow.utils import check_optimizer
from trieste.types import TensorType

from gpmaniflow.models.BezierProcess import LogBezierProcess

from piflow.models.transforms import DataTransformMixin

class _BezierProcessRegression(
        GPflowPredictor,
        TrainableProbabilisticModel
        ):

    def __init__(
            self,
            model: LogBezierProcess, #Generalise to any Bezier Process?
            optimizer: Optimizer | None = None,
            ):

        if optimizer is None:
            num_epochs = 10
            batch_size = 128
            optimizer = BatchOptimizer(
                tf.optimizers.Adam(learning_rate = 0.01),
                batch_size=batch_size,
                max_iter=num_epochs * model.num_data / batch_size,
                compile=True
            )

        super().__init__(optimizer)
        
        self._model = model
        self._ensure_variable_model_data()
        
        check_optimizer(self.optimizer)

    def _ensure_variable_model_data(self) -> None:
        if self._model.num_data is None:
            print("you must provide num_data")
        if not is_variable(self._model.num_data):
            print(self._model.num_data)
            self._model.num_data = tf.Variable(self._model.num_data, trainable = False)

    @property
    def model(self) -> LogBezierProcess:
        return self._model
    
    def get_kernel(self) -> gpflow.kernels.Kernel:
        return None

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        query_points = tf.reshape(query_points, (-1, self._model.input_dim))
        mean, cov = super().predict(query_points)
        return tf.expand_dims(mean, 1), tf.expand_dims(cov, 1)

    def update(self, dataset: Dataset) -> None:
        self._ensure_variable_model_data()
        num_data = dataset.query_points.shape[0]
        self.model.num_data.assign(num_data)

    def optimize(self, dataset: Dataset) -> None:
        super().optimize(dataset)
        elbo = self.model.maximum_log_likelihood_objective(dataset.astuple())
        try:
            parameters = self.model.get_parameters()
            self.model.increment_orders()
            super().optimise(dataset)
            new_elbo = self.model.maximum_log_likelihood_objective(dataset.astuple())
            if elbo > new_elbo:
                self.model.set_parameters(parameters)
        except:
            print('Implement get_parameters(), increment_orders() and set_parameters()')


class BezierProcessRegression(DataTransformMixin, _BezierProcessRegression):
    pass


class UncertaintySamplingSampler():
    def __call__(
        self,
        num_query_points: int,
        search_space: trieste.space.SearchSpace, 
        models: Mapping[str, trieste.models.interfaces.TrainableProbabilisticModel],
        datasets: Mapping[str, trieste.data.Dataset],
        prior: tfp.distributions.Distribution
    ) -> tf.Tensor:
        """Draw num_query_points samples from the posterior variance."""
        assert isinstance(prior, tfp.distributions.Uniform)
        model = models['INTEGRAND']
        dataset = datasets['INTEGRAND']
        print('Implement UncertaintySamplingSampler!')
        return search_space.sample(num_query_points)
