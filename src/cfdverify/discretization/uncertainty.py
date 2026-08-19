from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .discretization_error import DiscretizationError

import numpy as np
import pandas as pd
from scipy.special import stdtrit

###############################################################################
# UncertaintyModel
###############################################################################
class UncertaintyModel(ABC):
    """Abstract base class for uncertainty models"""

    def __init__(self, parent: 'DiscretizationError'):
        """Class constructor

        Parameters
        ----------
        parent : DiscretizationError
            Parent discretization error class
        """
        self.parent = parent

    def __call__(self, *args, **kwargs):
        return self.uncertainty(*args, **kwargs)

    @abstractmethod
    def uncertainty(self,
                    key: str,
                    index: Union[int, None] = None,
                    **kwargs,
    ) -> Union[np.floating, pd.Series]:
        """Uncertainty method

        Parameters
        ----------
        key : str
            Key of system response quantity of interest
        index : int | None
            Index of discretization level of interest or None for all levels

        Returns
        -------
        : np.floating | pd.Series
            Uncertainty of requested values
        """
        pass

class GCI(UncertaintyModel):
    """Computes uncertainty using the Grid Convergence Index"""

    def uncertainty(self,
                    key: str,
                    index: Union[int, None] = None,
                    fs: Union[int, float] = 1.25,
                    normalize: bool = False,
    ) -> Union[np.floating, pd.Series]:
        """Compute Grid Convergence Index (GCI) for requested values

        The GCI method was proposed by Patrick Roache as a way to uniformly
        report discretization uncertainty in computational fluid dynamics
        simulation results in 1994. By default values are not normalized as
        suggested by Roache and the factor of safety is 1.25.
        Roache provided the equation

        .. math::
            GCI_1 = \\frac{Fs * |\\epsilon_{21}|}{r_{21}^p - 1},

        for estimating the uncertainty of the finer mesh for any two mesh pairs,
        and the equation

        .. math::
            GCI_2 = r_{21}^p * \\frac{Fs * |\\epsilon_{21}|}{r_{21}^p - 1},

        for the coarser mesh of any mesh pair. These equations use the absolute
        relative error measure :math:`|\\epsilon_{21}|` corrected for the
        distance to the infinitely fine mesh; this is equivalent to the absolute
        estimated error :math:`|\\epsilon_{\\mathrm{est}}|` for exact fits.
        However, for regression fits of data they are not equivalent; therefore,
        this code implements the GCI uncertainty measure as

        .. math::
            GCI = Fs * |\\epsilon_{\\mathrm{est}}|,

        so that it is valid for both exact and regression fits.

        Parameters
        ----------
        key : str
            Key of system response quantity of interest
        index : int | None
            Index of discretization level of interest or None for all levels
        fs : int | float
            Factor of safety for computation. Defaults to 1.25
        normalize : bool
            Whether output GCI value should be normalized or not

        Returns
        -------
        gci : np.floating | pd.Series
            GCI of requested values

        References
        ----------
        Patrick J. Roache, 1994, Perspective: A Method for Uniform Reporting of
        Grid Refinement Studies, Journal of Fluids Engineering,
        116(3): 405-413. https://doi.org/10.1115/1.2910291.
        """
        gci = fs * self.parent.abs_estimated_error(key, index)

        if normalize:
            if index is None:
                gci = gci / self.parent.data[key]
            else:
                gci = gci / self.parent.data[key][index]

        return gci

class StudentsTDistribution(UncertaintyModel):
    """Computes uncertainty using student's t distribution"""

    def uncertainty(self,
                    key: str,
                    index: Union[int, None] = None,
                    significance: float=0.05,
    ) -> Union[np.floating, pd.Series]:
        """Compute uncertainty using Student's t distribution

        Student's t distribution is a generalization of the normal probability
        distribution with fatter tails to account for low sample counts from a
        population.

        Parameters
        ----------
        key : str
            Key of system response quantity of interest
        index : int | None
            Not used for class but included for uniform interface
        significance : float
            Double-sided significance for Student's-t distribution

        Returns
        -------
        u : np.floating | pd.Series
            Uncertainty of requested values using Student's t distribution
        """
        data = self.parent.data[key]

        # Standard deviation of data. Use N-1 as samples from random process
        std_dev = data.std()

        # Compute student's t
        n = len(self.parent)
        df = n - 1
        v = stdtrit(df, 1 - significance/2)

        # Compute uncertainty based on error and significance
        u = v * std_dev / np.sqrt(n)

        # Return Series if no index selected
        if index is None:
            u = pd.Series(np.ones(len(self.parent))*u, name=key)

        return u

class FactorOfSafety(UncertaintyModel):
    """Computes uncertainty by a constant factor of safety"""

    def uncertainty(self,
                    key: str,
                    index: Union[int, None] = None,
                    factor: Union[int, float]=3,
    ) -> Union[np.floating, pd.Series]:
        """Compute uncertainty as a constant factor of the error estimate

        Parameters
        ----------
        key : str
            Key of system response quantity of interest
        index : int | None
            Index of discretization level of interest or None for all levels
        factor : int | float
            Factor of safety to apply to error estimate

        Returns
        -------
        : np.floating | pd.Series
            Uncertainty of requested values using supplied factor of safety
        """
        if index is None:
            error = self.parent.error(key)
        else:
            error = self.parent.error(key, index)

        return factor * abs(error)


class EçaHoekstra2014Uncertainty(UncertaintyModel):
    """Computes uncertainty for Eça and Hoekstra 2014 Procedure
    
    This method is only compatible with the EçaHoekstra2014Model class.
    """

    def uncertainty(self,
                    key: str,
                    index: Union[int, None] = None,
    ) -> Union[np.floating, pd.Series]:
        """Compute uncertainty as a constant factor of the error estimate

        Parameters
        ----------
        key : str
            Key of system response quantity of interest
        index : int | None
            Index of discretization level of interest or None for all levels

        Returns
        -------
        : np.floating | pd.Series
            Uncertainty of requested values using supplied factor of safety
        """
        # Compute fitting error
        if index is None:
            fs = self.parent.data[key]
            hs = self.parent.hs
        else:
            fs = self.parent.data[key].iloc[index]
            hs = self.parent.hs.iloc[index]
        error_fit = np.abs(fs - self.parent.model.model(key, hs))
        # Standard deviation of fit
        sigma = self.parent.model.std
        # Discretization error estimate
        error = np.abs(self.parent.error(key, index))

        # Appendix A: Step 2
        # Compute data range parameter, Eq. 19.
        all_fs = self.parent.data[key]
        data_range = (max(all_fs) - min(all_fs)) / (len(self.parent) - 1)
        
        # Appendix A: Step 3
        # Set factor of safety
        p = self.parent.model.p_fit
        p_formal = self.parent.model.p_formal
        if p >= 0.5 and p < p_formal*1.05 and sigma < data_range:
            Fs = 1.25
        else:
            Fs = 3
        
        # Appendix A: Step 4
        # Compute uncertainty
        if sigma < data_range:
            # Eq. 20
            u = Fs*error + sigma + error_fit 
        else:
            # Eq. 21, note difference in parenthesis
            u = Fs * (sigma / data_range) * (error + sigma + error_fit)

        if index is None:
            return pd.Series(u, name=key)
        else:
            return u
    