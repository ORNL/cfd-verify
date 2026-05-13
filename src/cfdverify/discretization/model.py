import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .discretization_error import DiscretizationError

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

###############################################################################
# DiscretizationModel
###############################################################################
class DiscretizationModel(ABC):
    """Abstract base class for discretization error models"""

    def __init__(self, parent : 'DiscretizationError') -> None:
        """Class constructor

        Parameters
        ----------
        parent : DiscretizationError
            Parent discretization error class
        """
        self.parent = parent
        self.parameters = pd.DataFrame(index=self.parameter_keys,
                                       columns=self.parent.keys)
        self.solve()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    @abstractmethod
    def parameter_keys(self):
        """List of parameter keys"""
        pass

    @abstractmethod
    def model(self,
              key: str,
              h: Union[int, float, np.ndarray]
    ) -> Union[int, float, np.ndarray]:
        """Estimate system response quantity at provided discretizations

        Parameters
        ----------
        key : str
            Key of system response quantity
        h : int | float | np.ndarray
            Discretization levels of interest

        Returns
        -------
        : int | float | np.ndarray
            System response quantity estimate
        """
        pass

    @abstractmethod
    def solve(self, *args, **kwargs):
        """Solve the discretization model"""
        pass

    @abstractmethod
    def f_est(self) -> pd.Series:
        """Return estimate of system response quantities

        Returns
        -------
        : pd.Series
            System response quantity estimates
        """
        pass

    @abstractmethod
    def order(self) -> pd.Series:
        """Return observed convergence orders of system response quantities

        Returns
        -------
        : pd.Series
            Observed convergence orders
        """
        pass

class SinglePower(DiscretizationModel):
    """Model discretization error with a single term power series"""

    #: Parameter keys for SinglePower
    parameter_keys = ["f_est", "alpha", "p"]

    def model(self,
              key: str,
              h: Union[int, float, np.ndarray]
    ) -> Union[int, float, np.ndarray]:
        """Estimate system response quantity at provided discretizations

        The discretization model for a single term power series expansion is

        .. math::
            f_h = f_0 + \\alpha h^{\\hat{p}},

        where :math:`f_h` is the system response quantity (SRQ) at a
        representative discretization size of :math:`h`, :math:`f_0` is the
        estimated SRQ with no discretization error, :math:`\\alpha` is the term
        coefficient, and :math:`\\hat{p}` is the observed order of convergence.

        Parameters
        ----------
        key : str
            Key of system response quantity
        h : int | float | np.ndarray
            Discretization levels of interest

        Returns
        -------
        : int | float | np.ndarray
            System response quantity estimate
        """
        parameters = self.parameters[key]
        return parameters.iloc[0] + parameters.iloc[1] * h**parameters.iloc[2]

    def solve(self, p_limits: Union[list, tuple] = [0,np.inf]):
        """Solve the model

        Parameters
        ----------
        p_limits : list | tuple
            Lower and upper limit for observed convergence order
        """
        # Define model to be solved by curve fitting method
        def model_p(hs, f_est, alpha, p):
            return f_est + alpha*hs**p

        # Validate inputs
        if len(p_limits) != 2 or (type(p_limits) is not list and type(p_limits) is not tuple):
            raise ValueError("p_limits must be a list or tuple with two elements!")

        # Normalize data for improved fitting
        hs = self.parent.hs / self.parent.hs[0]
        fs = self.parent.data / self.parent.data.iloc[0]

        # Iterate over each key
        for key in self.parent.keys:
            fs_key = fs[key]
            # Compute initial estimates for parameters
            f_est_0 = fs_key[0]
            p_0 = 1
            if p_0 < p_limits[0]:
                p_0 = p_limits[0]
            elif p_0 > p_limits[1]:
                p_0 = p_limits[1]
            alpha_0 = ((fs_key.iloc[-1] - fs_key.iloc[0])
                       / (hs.iloc[-1] - hs.iloc[0])**p_0)
            bnds = ([-np.inf, -np.inf, p_limits[0]],
                    [np.inf, np.inf, p_limits[1]])

            # Solve
            with warnings.catch_warnings():
                if len(self.parent) == 3:
                    warnings.filterwarnings("ignore", message="Covariance")
                try:
                    popt, pconv = curve_fit(model_p,
                                            hs,
                                            fs_key,
                                            [f_est_0, alpha_0, p_0],
                                            bounds=bnds,
                                            )
                except RuntimeError:
                    print(f"Solution not found for {fs_key}! Setting to NaN!")
                    popt = [np.nan, np.nan, np.nan]
            self.parameters.loc[self.parameter_keys[0], key] = popt[0] * self.parent.data[key][0]
            self.parameters.loc[self.parameter_keys[1], key] = popt[1] * self.parent.data[key][0] / self.parent.hs[0]**popt[2]
            self.parameters.loc[self.parameter_keys[2], key] = popt[2]

    def f_est(self) -> pd.Series:
        """Return estimate of system response quantities

        Returns
        -------
        : pd.Series
            System response quantity estimates
        """
        return self.parameters.loc[self.parameter_keys[0]]

    def order(self) -> pd.Series:
        """Return observed convergence orders of system response quantities

        Returns
        -------
        : pd.Series
            Observed convergence orders
        """
        return self.parameters.loc[self.parameter_keys[2]]

class FirstAndSecondOrder(DiscretizationModel):
    """Model discretization error with a 1st and 2nd order power series"""

    #: Parameter keys for SinglePower
    parameter_keys = ["f_est", "alpha_1", "alpha_2"]

    def model(self,
              key: str,
              h: Union[int, float, np.ndarray]
    ) -> Union[int, float, np.ndarray]:
        """Estimate system response quantity at provided discretizations

        The discretization model for a 1st and 2nd order power series expansion
        is

        .. math::
            f_h = f_0 + \\alpha_1 h^{1} + \\alpha_2 h^{2},

        where :math:`f_h` is the system response quantity (SRQ) at a
        representative discretization size of :math:`h`, :math:`f_0` is the
        estimated SRQ for an infinitely fine mesh, :math:`\\alpha_1` is the
        1st order term coefficient, and :math:`\\alpha_2` is the 2nd order term
        coefficient.

        Parameters
        ----------
        key : str
            Key of system response quantity
        h : int | float | np.ndarray
            Discretization levels of interest

        Returns
        -------
        : int | float | np.ndarray
            System response quantity estimate
        """
        parameters = self.parameters[key]
        return parameters.iloc[0] + parameters.iloc[1]*h + parameters.iloc[2]*h**2

    def solve(self):
        """Solve the model"""
        # Define model to be solved by curve fitting method
        def model_p(hs, f_est, alpha_1, alpha_2):
            return f_est + alpha_1*hs + alpha_2*hs**2

        # Normalize data for improved fitting
        hs = self.parent.hs / self.parent.hs[0]
        fs = self.parent.data / self.parent.data.iloc[0]

        # Iterate over each key
        for key in self.parent.keys:
            fs_key = fs[key]
            # Compute initial estimates for parameters
            f_est_0 = fs_key[0]
            alpha_1_0 = ((fs_key.iloc[-1] - fs_key.iloc[0])
                         / (hs.iloc[-1] - hs.iloc[0]))
            alpha_2_0 = ((fs_key.iloc[-1] - fs_key.iloc[0])
                         / (hs.iloc[-1] - hs.iloc[0])**2)
            bnds = ([-np.inf, -np.inf, -np.inf],
                    [np.inf, np.inf, np.inf])

            # Solve
            with warnings.catch_warnings():
                if len(self.parent) == 3:
                    warnings.filterwarnings("ignore", message="Covariance")
                popt, pconv = curve_fit(model_p,
                                        hs,
                                        fs_key,
                                        [f_est_0, alpha_1_0, alpha_2_0],
                                        bounds=bnds,
                                        )
            self.parameters.loc[self.parameter_keys[0], key] = popt[0] * self.parent.data[key][0]
            self.parameters.loc[self.parameter_keys[1], key] = popt[1] * self.parent.data[key][0] / self.parent.hs[0]**1
            self.parameters.loc[self.parameter_keys[2], key] = popt[2] * self.parent.data[key][0] / self.parent.hs[0]**2

    def f_est(self) -> pd.Series:
        """Return estimate of system response quantities

        Returns
        -------
        : pd.Series
            System response quantity estimates
        """
        return self.parameters.loc[self.parameter_keys[0]]

    def order(self) -> pd.Series:
        """Return observed convergence orders of system response quantities

        Returns
        -------
        : pd.Series
            Observed convergence orders
        """
        return pd.Series([1,2])

class AverageValue(DiscretizationModel):
    """Model discretization error as average of all values"""

    #: Parameter keys for AverageValue
    parameter_keys = ["mean", "std", "order"]

    def model(self,
              key: str,
              h: Union[int, float, np.ndarray]
    ) -> Union[int, float, np.ndarray]:
        """Estimate system response quantity at provided discretizations

        AverageValue uses the average of all system response quantities (SRQ)s
        as the estimated true value. This model is useful for cases with
        oscillatory data without a definitive trend.

        Parameters
        ----------
        key : str
            Key of system response quantity
        h : int | float | np.ndarray
            Discretization levels of interest

        Returns
        -------
        est : int | float | np.ndarray
            System response quantity estimate
        """
        if type(h) is int or type(h) is float:
            est = self.parameters.loc[self.parameter_keys[0], key]
        else:
            est = (np.ones(np.shape(h))
                   * self.parameters.loc[self.parameter_keys[0], key])

        return est

    def solve(self):
        """Solve the discretization model"""
        self.parameters.loc[self.parameter_keys[0]] = self.parent.data.mean()
        # Sample standard deviation
        self.parameters.loc[self.parameter_keys[1]] = self.parent.data.std()
        self.parameters.loc[self.parameter_keys[2]] = 0

    def f_est(self) -> pd.Series:
        """Return estimate of system response quantities

        Returns
        -------
        : pd.Series
            System response quantity estimates
        """
        return self.parameters.loc[self.parameter_keys[0]]

    def order(self) -> pd.Series:
        """Return observed convergence orders of system response quantities

        Returns
        -------
        : pd.Series
            Observed convergence orders
        """
        return self.parameters.loc[self.parameter_keys[2]]

class FinestValue(DiscretizationModel):
    """Model discretization error relative to finest response value"""

    #: Parameter keys for FinestValue
    parameter_keys = ["f_est", "order"]

    def model(self,
              key: str,
              h: Union[int, float, np.ndarray]
    ) -> Union[int, float, np.ndarray]:
        """Estimate system response quantity at provided discretizations

        FinestValue uses the system response quantity (SRQ) of the finest
        result as its estimate. This model is useful if only the finest mesh
        result is trusted.

        Parameters
        ----------
        key : str
            Key of system response quantity
        h : int | float | np.ndarray
            Discretization levels of interest

        Returns
        -------
        est: int | float | np.ndarray
            System response quantity estimate
        """
        if type(h) is int or type(h) is float:
            est = self.parameters.loc[self.parameter_keys[0], key]
        else:
            est = (np.ones(np.shape(h))
                   * self.parameters.loc[self.parameter_keys[0], key])

        return est

    def solve(self):
        """Solve the discretization model"""
        self.parameters.loc[self.parameter_keys[0]] = self.parent.data.iloc[0]
        self.parameters.loc[self.parameter_keys[1]] = 0

    def f_est(self) -> pd.Series:
        """Return estimate of system response quantities

        Returns
        -------
        : pd.Series
            System response quantity estimates
        """
        return self.parameters.loc[self.parameter_keys[0]]

    def order(self) -> pd.Series:
        """Return observed convergence orders of system response quantities

        Returns
        -------
        : pd.Series
            Observed convergence orders
        """
        return self.parameters.loc[self.parameter_keys[1]]

class MaximumValue(DiscretizationModel):
    """Model discretization error relative to maximum response value"""

    #: Parameter keys for MaximumValue
    parameter_keys = ["f_est", "order"]

    def model(self,
              key: str,
              h: Union[int, float, np.ndarray]
    ) -> Union[int, float, np.ndarray]:
        """Estimate system response quantity at provided discretizations

        MaximumValue uses the largest system response quantity (SRQ) of the
        results. This model is useful if expert knowledge indicates this is the
        only reliable result.

        Parameters
        ----------
        key : str
            Key of system response quantity
        h : int | float | np.ndarray
            Discretization levels of interest

        Returns
        -------
        est : int | float | np.ndarray
            System response quantity estimate
        """
        if type(h) is int or type(h) is float:
            est = self.parameters.loc[self.parameter_keys[0], key]
        else:
            est = (np.ones(np.shape(h))
                   * self.parameters.loc[self.parameter_keys[0], key])

        return est

    def solve(self):
        """Solve the discretization model"""
        self.parameters.loc[self.parameter_keys[0]] = self.parent.data.max()
        self.parameters.loc[self.parameter_keys[1]] = 0

    def f_est(self) -> pd.Series:
        """Return estimate of system response quantities

        Returns
        -------
        : pd.Series
            System response quantity estimates
        """
        return self.parameters.loc[self.parameter_keys[0]]

    def order(self) -> pd.Series:
        """Return observed convergence orders of system response quantities

        Returns
        -------
        : pd.Series
            Observed convergence orders
        """
        return self.parameters.loc[self.parameter_keys[1]]

class MinimumValue(DiscretizationModel):
    """Model discretization error relative to minimum response value"""

    #: Parameter keys for MinimumValue
    parameter_keys = ["f_est", "order"]

    def model(self,
              key: str,
              h: Union[int, float, np.ndarray]
    ) -> Union[int, float, np.ndarray]:
        """Estimate system response quantity at provided discretizations

        MinimumValue uses the smallest system response quantity (SRQ) of the
        results. This model is useful if expert knowledge indicates this is the
        only reliable result.

        Parameters
        ----------
        key : str
            Key of system response quantity
        h : int | float | np.ndarray
            Discretization levels of interest

        Returns
        -------
        est : int | float | np.ndarray
            System response quantity estimate
        """
        if type(h) is int or type(h) is float:
            est = self.parameters.loc[self.parameter_keys[0], key]
        else:
            est = (np.ones(np.shape(h))
                   * self.parameters.loc[self.parameter_keys[0], key])

        return est

    def solve(self):
        """Solve the discretization model"""
        self.parameters.loc[self.parameter_keys[0]] = self.parent.data.min()
        self.parameters.loc[self.parameter_keys[1]] = 0

    def f_est(self) -> pd.Series:
        """Return estimate of system response quantities

        Returns
        -------
        : pd.Series
            System response quantity estimates
        """
        return self.parameters.loc[self.parameter_keys[0]]

    def order(self) -> pd.Series:
        """Return observed convergence orders of system response quantities

        Returns
        -------
        : pd.Series
            Observed convergence orders
        """
        return self.parameters.loc[self.parameter_keys[1]]
