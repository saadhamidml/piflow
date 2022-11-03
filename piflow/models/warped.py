from abc import abstractmethod
from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.models import GPR
from gpflow.posteriors import GPRPosterior
import tensorflow as tf
from trieste.data import Dataset
from trieste.models.gpflow.models import GaussianProcessRegression as TGaussianProcessRegression

from .transforms import DataTransformMixin
from .utils import set_model_priors, initialise_hyperparameters


class WarpedGPR(GPR):
    """GPR in a warped output space."""

    def __init__(self, data: RegressionData, *args, **kwargs):
        X_data, unwarped_Y_data = data
        self.unwarped_Y_data = unwarped_Y_data
        warped_data = X_data, self._warp(unwarped_Y_data)
        super().__init__(warped_data, *args, **kwargs)

    @abstractmethod
    def _warp(self, y: tf.Tensor) -> tf.Tensor:
        """Perform warping of Y values.
        
        :param y: The y values to be warped.
        :return: The warped Y values.
        """
        raise NotImplementedError
    
    def predict_g(self, *args, posterior: GPRPosterior = None, **kwargs):
        """Posterior over warped space."""
        if posterior is not None:
            return posterior.predict_f(*args, **kwargs)
        else:
            return super().predict_f(*args, **kwargs)
    
    @abstractmethod
    def predict_f(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False
    ) -> MeanAndVariance:
        """Posterior over unwarped function."""


class WSABI_L_GPR(WarpedGPR):
    """Linearised WSABI model."""

    @property
    def alpha(self):
        return 0.8 * tf.reduce_min(self.unwarped_Y_data)

    def _warp(self, y: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(2 * (y - self.alpha)) 

    def predict_f(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
        posterior: GPRPosterior = None
    ) -> MeanAndVariance:
        if full_output_cov:
            raise NotImplementedError
        g_mean, g_var = self.predict_g(
            Xnew, full_cov, full_output_cov, posterior=posterior
        )
        f_mean = self.alpha + g_mean ** 2 / 2
        if full_cov:
            f_var = g_var * tf.tensordot(g_mean, g_mean, axes=0)
        else:
            f_var = g_var * g_mean ** 2
        return f_mean, f_var


class MMLT_GPR(WarpedGPR):
    """Moment matched log transform model."""

    def _warp(self, y: tf.Tensor) -> tf.Tensor:
        return tf.math.log(y + 1e-9) 

    def predict_f(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
        posterior: GPRPosterior = None
    ) -> MeanAndVariance:
        if full_output_cov:
            raise NotImplementedError
        # Get warped prediction.
        g_mean, g_var = self.predict_g(Xnew, full_cov, full_output_cov, posterior=posterior)
        g_var_diag = tf.linalg.diag_part(g_var) if full_cov else g_var
        # Compute unwarping.
        f_mean = tf.exp(g_mean + 0.5 * g_var_diag)
        if full_cov:
            covar_factors = tf.linalg.matmul(
                f_mean, f_mean, transpose_a=True
            )
            f_cov = covar_factors * tf.math.expm1(g_var)
        else:
            f_cov = tf.math.expm1(g_var_diag) * f_mean ** 2 
        return f_mean, f_cov 


class GaussianProcessRegression(TGaussianProcessRegression):
    """GPR with automatic hyperparameter setting."""

    def update(self, dataset: Dataset) -> None:
        """Set the hyperparamter priors and initialise by sampling from
        these priors and selecting the best.
        """
        super().update(dataset)
        set_model_priors(self.model)
        initialise_hyperparameters(self.model)  # TODO: Allow configurable num_samples.


class _WarpedGaussianProcessRegression(GaussianProcessRegression):
    """Trieste Model for Warped GPR."""

    def update(self, dataset: Dataset) -> None:
        self.model.unwarped_Y_data = dataset.observations
        warped_dataset = Dataset(dataset.query_points, self.model._warp(dataset.observations))
        super().update(warped_dataset)


class WarpedGaussianProcessRegression(DataTransformMixin, _WarpedGaussianProcessRegression):
    pass
