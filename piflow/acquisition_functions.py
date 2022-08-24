from trieste.acquisition.interface import AcquisitionFunctionBuilder as TAcquisitionFunctionBuilder


class AcquisitionFunctionBuilder(TAcquisitionFunctionBuilder):
    """Builder for the uncertainty sampling acquisition function, where
    the point chosen is the value of the integrand with the highest
    variance.
    """

    def prepare_acquisition_function(
        self,
        models,
        datasets,
        prior
    ):
        """Prepare an acquisition function."""
        pass

    def update_acquisition_function(
        self,
        models,
        datasets,
        prior
    ):
        """Update an acquisition function."""
        pass


class UncertaintySampling(AcquisitionFunctionBuilder):
    pass


class uncertainty_sampling():
    pass
