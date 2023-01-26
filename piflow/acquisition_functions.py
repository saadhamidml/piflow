from abc import abstractmethod
from typing import Mapping, Union
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import trieste
from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder as TAcquisitionFunctionBuilder,
    AcquisitionFunctionClass
)
from trieste.acquisition.rule import AcquisitionRule as TAcquisitionRule
from trieste.acquisition.optimizer import (
    automatic_optimizer_selector, generate_continuous_optimizer
)


logger = logging.getLogger(__name__)


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


class OptimizeAcquisition(TAcquisitionRule):
    def __init__(self, builder: AcquisitionFunctionBuilder) -> None:
        self._builder = builder
        self._optimizer = generate_continuous_optimizer(
                num_initial_samples=8192,
                num_optimization_runs=16,
            )
        self._num_query_points = 1
        self._acquisition_function = None
    
    def acquire(
        self,
        search_space: trieste.space.SearchSpace, 
        models: Mapping[str, trieste.models.interfaces.TrainableProbabilisticModel],
        datasets: Mapping[str, trieste.data.Dataset],
        prior: tfp.distributions.Distribution
    ) -> tf.Tensor:
        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                models['INTEGRAND'],
                datasets['INTEGRAND'],
                prior
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models['INTEGRAND'],
                datasets['INTEGRAND'],
                prior
            )
        return self._optimizer(search_space, self._acquisition_function)


class SampleAcquisition(TAcquisitionRule):
    def __init__(self, builder: AcquisitionFunctionBuilder, sampler, num_query_points: int) -> None:
        self._builder = builder
        self._optimizer = automatic_optimizer_selector
        self._sampler = sampler
        self._num_query_points = num_query_points
        self._acquisition_function = None

    def acquire(
        self,
        search_space: trieste.space.SearchSpace, 
        models: Mapping[str, trieste.models.interfaces.TrainableProbabilisticModel],
        datasets: Mapping[str, trieste.data.Dataset],
        prior: tfp.distributions.Distribution
    ) -> tf.Tensor:
        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                models['INTEGRAND'],
                datasets['INTEGRAND'],
                prior
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models['INTEGRAND'],
                datasets['INTEGRAND'],
                prior
            )
        optimum = self._optimizer(search_space, self._acquisition_function)
        logger.debug('Acquisiton function optimal value:', optimum)
        return self._sampler(self._num_query_points, search_space, optimum, models, datasets, prior)


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
        if len(log_prior.shape) == 3:
            log_prior = tf.reduce_sum(log_prior, axis=-1)
        return tf.reshape(tf.math.log(var), (-1, 1)) + 2 * log_prior
