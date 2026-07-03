import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .discretization_error import DiscretizationError

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

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
    def order(self) -> Union[pd.Series, pd.DataFrame]:
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


class EçaHoekstra2014Model(DiscretizationModel):
    """Model discretization error following Eça and Hoekstra 2014"""

    # Parameters are a combination of all possible parameters
    parameter_keys = ["model", "f_est", "alpha_1", "p_1", "alpha_2", "p_2"]

    # Used for uncertainty fitting
    p_fit = None 
    p_formal = None 
    weights = None
    std = None

    # Define model options
    @staticmethod
    def model_p(hs, f_est, alpha, p):
        return f_est + alpha*hs**p
    
    @staticmethod
    def model_1(hs, f_est, alpha):
        return f_est + alpha*hs**1
    
    @staticmethod
    def model_2(hs, f_est, alpha):
        return f_est + alpha*hs**2
    
    @staticmethod
    def model_1and2(hs, f_est, alpha_1, alpha_2):
        return f_est + alpha_1*hs**1 + alpha_2*hs**2
    
    # Define residual functions for least squares fitting
    @staticmethod
    def residual_p(params, hs, fs):
        f_est, alpha, p = params
        return f_est + alpha*hs**p - fs

    @staticmethod
    def residual_1(params, hs, fs):
        f_est, alpha = params
        return f_est + alpha*hs - fs

    @staticmethod
    def residual_2(params, hs, fs):
        f_est, alpha = params
        return f_est + alpha*hs**2 - fs

    @staticmethod
    def residual_1and2(params, hs, fs):
        f_est, alpha_1, alpha_2 = params
        return f_est + alpha_1*hs + alpha_2*hs**2 - fs

    @staticmethod
    def residual_p_weighted(params, hs, fs, weights):
        f_est, alpha, p = params
        return weights * (f_est + alpha*hs**p - fs)

    @staticmethod
    def residual_1_weighted(params, hs, fs, weights):
        f_est, alpha = params
        return weights * (f_est + alpha*hs - fs)

    @staticmethod
    def residual_2_weighted(params, hs, fs, weights):
        f_est, alpha = params
        return weights * (f_est + alpha*hs**2 - fs)

    @staticmethod
    def residual_1and2_weighted(params, hs, fs, weights):
        f_est, alpha_1, alpha_2 = params
        return weights * (f_est + alpha_1*hs + alpha_2*hs**2 - fs)
    
    @staticmethod
    def fit_std_dev(residuals, weights, n, ddof):
        return np.sqrt(residuals @ weights @ residuals / (n - ddof))
    

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
        parameters = self.parameters[key]
        f_est = parameters.iloc[1]

        if parameters.loc["model"] == "model_p":
            alpha = parameters.iloc[2]
            p = parameters.iloc[3]
            fs = self.model_p(h, f_est, alpha, p)
        elif parameters.loc["model"] == "model_1":
            alpha = parameters.iloc[2]
            fs = self.model_1(h, f_est, alpha)
        elif parameters.loc["model"] == "model_2":
            alpha = parameters.iloc[2]
            fs = self.model_2(h, f_est, alpha)
        elif parameters.loc["model"] == "model_1and2":
            alpha_1 = parameters.iloc[2]
            alpha_2 = parameters.iloc[4]
            fs = self.model_1and2(h, f_est, alpha_1, alpha_2)
        else:
            raise ValueError("Invalid model representation!")
        
        return fs

    def solve(self, p_formal: int = 2):
        """Solve the model

        Parameters
        ----------
        p_formal : int
            Formal order of convergence of solver
        """
        # Error if less than four grids.  Can't compute standard deviation of fits.
        if len(self.parent) < 4:
            raise ValueError("This method requires at least four discretization levels!")
        
        self.p_formal = p_formal

        # Compute weights. Appendix B.1
        inverse_hs = 1 /self.parent.hs
        weights = inverse_hs / np.sum(inverse_hs)
        n = len(weights)

        # Get data
        # TODO, consider normalizing in the future
        hs = self.parent.hs #/ self.parent.hs[0]
        fs = self.parent.data #/ self.parent.data.iloc[0]

        # Iterate over each key
        for key in self.parent.keys:
            fs_key = fs[key]
            # Compute initial estimates for parameters
            f_est_0 = fs_key[0]
            p_0 = 1
            if p_0 > p_formal:
                p_0 = p_formal
            alpha_0 = ((fs_key.iloc[-1] - fs_key.iloc[0])
                       / (hs.iloc[-1] - hs.iloc[0])**p_0)

            # 1. Solve for unknown order first. 
            # 1a. Fit weighted and unweighted equations
            result_p = least_squares(self.residual_p, (f_est_0, alpha_0, p_0), args=(hs, fs_key))
            result_pw = least_squares(self.residual_p_weighted, (f_est_0, alpha_0, p_0), args=(hs, fs_key, weights))
            # 1b. Take result with smallest standard deviation
            std_p = self.fit_std_dev(result_p.fun, np.identity(n), n, 3)
            std_pw = self.fit_std_dev(result_pw.fun, np.diag(n*weights), n, 3)
            if std_p < std_pw:
                model_representation = "model_p"
                f_est = result_p.x[0]
                alphas = [result_p.x[1], np.nan]
                orders = [result_p.x[2], np.nan]
                self.std = std_p
            else:
                model_representation = "model_p"
                f_est = result_pw.x[0]
                alphas = [result_pw.x[1], np.nan]
                orders = [result_pw.x[2], np.nan]
                self.std = std_pw
            self.p_fit = orders[0]
            # 1c. If result is between p=0.5 and formal order, use fit
            if orders[0] >= 0.5 and orders[0] <= p_formal:
                pass
                            
            elif orders[0] > p_formal:
                # 2. Solve for 1st or 2nd order if exceeding formal order
                # 2a. Fit weighted and unweighted equations
                result_1 = least_squares(self.residual_1, (f_est_0, alpha_0), args=(hs, fs_key))
                result_1w = least_squares(self.residual_1_weighted, (f_est_0, alpha_0), args=(hs, fs_key, weights))
                result_2 = least_squares(self.residual_2, (f_est_0, alpha_0), args=(hs, fs_key))
                result_2w = least_squares(self.residual_2_weighted, (f_est_0, alpha_0), args=(hs, fs_key, weights))
                # 2b. Take result with smallest standard deviation
                std_1 = self.fit_std_dev(result_1.fun, np.identity(n), n, 2)
                std_1w = self.fit_std_dev(result_1w.fun, np.diag(n*weights), n, 2)
                std_2 = self.fit_std_dev(result_2.fun, np.identity(n), n, 2)
                std_2w = self.fit_std_dev(result_2w.fun, np.diag(n*weights), n, 2)
                sigmas = [std_1, std_1w, std_2, std_2w]
                index = sigmas.index(np.min(sigmas))
                if index == 0:
                    model_representation = "model_1"
                    f_est = result_1.x[0]
                    alphas = [result_1.x[1], np.nan]
                    orders = [1, np.nan]
                    self.std = std_1
                elif index == 1:
                    model_representation = "model_1"
                    f_est = result_1w.x[0]
                    alphas = [result_1w.x[1], np.nan]
                    orders = [1, np.nan]
                    self.std = std_1w
                elif index == 2:
                    model_representation = "model_2"
                    f_est = result_2.x[0]
                    alphas = [result_2.x[1], np.nan]
                    orders = [2, np.nan]
                    self.std = std_2
                else:
                    model_representation = "model_2"
                    f_est = result_2w.x[0]
                    alphas = [result_2w.x[1], np.nan]
                    orders = [2, np.nan]
                    self.std = std_2w

            else:
                # 3. Solve for 1st, 2nd, and mixed order if order is less than 0.5
                # 3a. Fit weighted and unweighted equations
                result_1 = least_squares(self.residual_1, (f_est_0, alpha_0), args=(hs, fs_key))
                result_1w = least_squares(self.residual_1_weighted, (f_est_0, alpha_0), args=(hs, fs_key, weights))
                result_2 = least_squares(self.residual_2, (f_est_0, alpha_0), args=(hs, fs_key))
                result_2w = least_squares(self.residual_2_weighted, (f_est_0, alpha_0), args=(hs, fs_key, weights))
                result_1and2 = least_squares(self.residual_1and2, (f_est_0, alpha_0, alpha_0), args=(hs, fs_key))
                result_1and2w = least_squares(self.residual_1and2_weighted, (f_est_0, alpha_0, alpha_0), args=(hs, fs_key, weights))
                # 3b. Take result with smallest standard deviation
                std_1 = self.fit_std_dev(result_1.fun, np.identity(n), n, 2)
                std_1w = self.fit_std_dev(result_1w.fun, np.diag(n*weights), n, 2)
                std_2 = self.fit_std_dev(result_2.fun, np.identity(n), n, 2)
                std_2w = self.fit_std_dev(result_2w.fun, np.diag(n*weights), n, 2)
                std_1and2 = self.fit_std_dev(result_1and2.fun, np.identity(n), n, 3)
                std_1and2w = self.fit_std_dev(result_1and2w.fun, np.diag(n*weights), n, 3)
                sigmas = [std_1, std_1w, std_2, std_2w, std_1and2, std_1and2w]
                index = sigmas.index(np.min(sigmas))
                if index == 0:
                    model_representation = "model_1"
                    f_est = result_1.x[0]
                    alphas = [result_1.x[1], np.nan]
                    orders = [1, np.nan]
                    self.std = std_1
                elif index == 1:
                    model_representation = "model_1"
                    f_est = result_1w.x[0]
                    alphas = [result_1w.x[1], np.nan]
                    orders = [1, np.nan]
                    self.std = std_1w
                elif index == 2:
                    model_representation = "model_2"
                    f_est = result_2.x[0]
                    alphas = [result_2.x[1], np.nan]
                    orders = [2, np.nan]
                    self.std = std_2
                elif index == 3:
                    model_representation = "model_2"
                    f_est = result_2w.x[0]
                    alphas = [result_2w.x[1], np.nan]
                    orders = [2, np.nan]
                    self.std = std_2w
                elif index == 4:
                    model_representation = "model_1and2"
                    f_est = result_1and2.x[0]
                    alphas = [result_1and2.x[1], result_1and2.x[2]]
                    orders = [1, 2]
                    self.std = std_1and2
                else:
                    model_representation = "model_1and2"
                    f_est = result_1and2w.x[0]
                    alphas = [result_1and2w.x[1], result_1and2w.x[2]]
                    orders = [1, 2]
                    self.std = std_1and2w

            # 4. Compute parameters from model
            self.parameters.loc[self.parameter_keys[0], key] = model_representation
            self.parameters.loc[self.parameter_keys[1], key] = f_est #* self.parent.data[key][0]
            self.parameters.loc[self.parameter_keys[2], key] = alphas[0] #* self.parent.data[key][0] / self.parent.hs[0]**2
            self.parameters.loc[self.parameter_keys[3], key] = orders[0]
            self.parameters.loc[self.parameter_keys[4], key] = alphas[1]
            self.parameters.loc[self.parameter_keys[5], key] = orders[1]


    def f_est(self) -> pd.Series:
        """Return estimate of system response quantities

        Returns
        -------
        : pd.Series
            System response quantity estimates
        """
        return self.parameters.loc["f_est"]

    def order(self) -> pd.DataFrame:
        """Return observed convergence orders of system response quantities

        Returns
        -------
        : pd.DataFrame
            Observed convergence orders
        """
        return self.parameters.loc[["p_1", "p_2"]]
       