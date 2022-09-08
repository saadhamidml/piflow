from __future__ import annotations

import tensorflow as tf
from typing import Optional, Tuple, Union, cast

import gpflow
from gpflow.utilities import is_variable
from trieste.data import Dataset
from trieste.models.gpflow.interface import GPflowPredictor
from trieste.models.interfaces import TrainableProbabilisticModel
from trieste.models.optimizer import BatchOptimizer, Optimizer
from trieste.models.gpflow.utils import check_optimizer
from trieste.types import TensorType

from gpmaniflow.models.BezierProcess import LogBezierProcess

class BezierProcessRegression(
        GPflowPredictor,
        TrainableProbabilisticModel
        ):

    def __init__(
            self,
            model: LogBezierProcess, #Generalise to any Bezier Process?
            optimizer: Optimizer | None = None,
            ):

        if optimizer is None:
            optimizer = BatchOptimizer(tf.optimizers.Adam(), batch_size=100, compile=True)
            # FIX OPTIMIZER

        super().__init__(optimizer)

        check_optimizer(self.optimizer)

        self._model = model
        self._ensure_variable_model_data()

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
        return super().optimize(dataset)

