from abc import abstractmethod
from typing import Mapping, Union
import tensorflow as tf
import tensorflow_probability as tfp
import trieste
from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder as TAcquisitionFunctionBuilder,
    AcquisitionFunctionClass
)


class AcquisitionFunctionBuilder(TAcquisitionFunctionBuilder):
    """Builder for PI acquisition functions, which require a prior."""

    @abstractmethod
    def prepare_acquisition_function(
        self,
        model: trieste.models.interfaces.TrainableProbabilisticModel,
        dataset: trieste.data.Dataset,
        prior: tfp.distributions.Distribution
    ):
        """Prepare an acquisition function."""

    @abstractmethod
    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: trieste.models.interfaces.TrainableProbabilisticModel,
        dataset: trieste.data.Dataset,
        prior: tfp.distributions.Distribution
    ):
        """Update an acquisition function."""


class UncertaintySampling(AcquisitionFunctionBuilder):
    """Builder for the uncertainty sampling acquisition function, where
    the point chosen is the value of the integrand with the highest
    variance.
    """
    
    def prepare_acquisition_function(
        self,
        model: trieste.models.interfaces.TrainableProbabilisticModel,
        dataset: trieste.data.Dataset,
        prior: tfp.distributions.Distribution
    ):
        return uncertainty_sampling(model, prior)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: trieste.models.interfaces.TrainableProbabilisticModel,
        datasets: trieste.data.Dataset,
        prior: tfp.distributions.Distribution
    ):
        return function


class uncertainty_sampling(AcquisitionFunctionClass):
    def __init__(
        self,
        model: trieste.models.interfaces.TrainableProbabilisticModel,
        prior: tfp.distributions.Distribution
    ) -> None:
        self._model = model
        self._prior = prior
    
    def update(self):
        pass
    
    def __call__(self, x: Union[tf.Tensor, tf.Variable]):
        """Uncertainty sampling acquisition function."""
        _, var = self._model.predict(x)
        log_prior = self._prior.log_prob(x)
        return tf.log(var) + 2 * log_prior
