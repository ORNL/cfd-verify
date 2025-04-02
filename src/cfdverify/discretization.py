# Standard Python packages
from abc import ABC, abstractmethod
import warnings
# Non-standard Python packages
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import stdtrit
import matplotlib.pyplot as plt

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
              h: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        """Estimate system response quantity at provided discertizations
        
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
    parameter_keys = ["f_est", "p", "alpha"]

    def model(self,
              key: str,
              h: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        """Estimate system response quantity at provided discertizations

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
        return parameters.iloc[0] + parameters.iloc[2] * h**parameters.iloc[1]
    
    def solve(self, p_limits: list | tuple = [0,np.inf]):
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
                popt, pconv = curve_fit(model_p,
                                        hs,
                                        fs_key,
                                        [f_est_0, alpha_0, p_0],
                                        bounds=bnds,
                                        )
            self.parameters.loc[self.parameter_keys[0], key] = popt[0] * self.parent.data[key][0]
            self.parameters.loc[self.parameter_keys[1], key] = popt[2]
            self.parameters.loc[self.parameter_keys[2], key] = popt[1] * self.parent.data[key][0] / self.parent.hs[0]**popt[2]
    
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
    
class AverageValue(DiscretizationModel):
    """Model discretization error as average of all values"""

    #: Parameter keys for AverageValue
    parameter_keys = ["mean", "std", "order"]

    def model(self,
              key: str,
              h: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        """Estimate system response quantity at provided discertizations

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
              h: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        """Estimate system response quantity at provided discertizations

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
              h: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        """Estimate system response quantity at provided discertizations

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
              h: int | float | np.ndarray
    ) -> int | float | np.ndarray:
        """Estimate system response quantity at provided discertizations

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

###############################################################################
# ErrorModel
###############################################################################
class ErrorModel(ABC):
    """Abstract base class for response error models"""

    def __init__(self, parent: 'DiscretizationError'):
        """Class constructor
        
        Parameters
        ----------
        parent : DiscretizationError
            Parent discretization error class
        """
        self.parent = parent

    def __call__(self, *args, **kwargs):
        return self.error(*args, **kwargs)

    @abstractmethod
    def error(self,
              key: str | None = None,
              index: int | None = None,
    ) -> np.floating | pd.Series | pd.DataFrame:
        """Error method
        
        Parameters
        ----------
        key : str | None
            Key of system response quantity of interest or None for all SRQs
        index : int | None
            Index of discretization level of interest or None for all levels
            
        Returns
        -------
        : np.floating | pd.Series | pd.DataFrame
            Error of requested values
        """
        pass

    def get_data(self, key:str | None) -> pd.Series | pd.DataFrame:
        """Return either all discretization data or key data
        
        Parameters
        ----------
        key: str | None
            Key for system response quantity or None for all data

        Returns
        -------
        data : pd.Series | pd.DataFrame
            DataFrame of system response quantities of interest
        """
        if key is None:
            data = self.parent.data
        else:
            data = self.parent.data[key]
        return data

class EstimatedError(ErrorModel):
    """Compute errors relative to estimated response value"""

    def error(self,
              key: str | None = None,
              index: int | None = None,
    ) -> np.floating | pd.Series | pd.DataFrame:
        """Compute error relative to estimated zero discretization error value

        .. math::
            \\epsilon_i = f_i - f_0.
        
        Parameters
        ----------
        key : str | None
            Key of system response quantity of interest or None for all SRQs
        index : int | None
            Index of discretization level of interest or None for all levels
            
        Returns
        -------
        err : np.floating | pd.Series | pd.DataFrame
            Estimated error of requested values
        """
        data = self.get_data(key)
        if key is None:
            f_est = self.parent.f_est
        else:
            f_est = self.parent.f_est[key]

        if index is None:
            err = data - f_est
        else:
            err = data.iloc[index] - f_est
        
        return err
    
class RelativeError(ErrorModel):
    """Compute errors relative to coarser response value"""

    def error(self,
              key: str | None = None,
              index: int | None = None,
    ) -> np.floating | pd.Series | pd.DataFrame:
        """Compute error relative to coarser discretization level

        Errors for all but the coarsest level are computed as

        .. math::
            \\epsilon_i = f_i - f_{i+1},

        while the error for the coarsest level is computed as

        .. math::
            \\epsilon_i = f_{i-1} - f_{i}.
        
        Parameters
        ----------
        key : str | None
            Key of system response quantity of interest or None for all SRQs
        index : int | None
            Index of discretization level of interest or None for all levels
            
        Returns
        -------
        rel_err : np.floating | pd.Series | pd.DataFrame
            Relative error of requested values
        """
        data = self.get_data(key)

        if index is None:
            rel_err = data.diff(-1)
            # Define relative error for last mesh as same as previous mesh
            rel_err.iloc[-1] = rel_err.iloc[-2]
        elif index != len(self.parent) - 1:
            rel_err = data[index:index+2].diff(-1).iloc[0]
        else:
            # Define relative error for last mesh as same as previous mesh
            rel_err = data[index-1:index+1].diff(-1).iloc[0]

        return rel_err
            
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
                    index: int | None = None,
                    **kwargs,
    ) -> np.floating | pd.Series:
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
                    index: int | None = None,
                    fs: int | float = 1.25,
                    normalize: bool = False,
    ) -> np.floating | pd.Series:
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
        estimated error :math:`|\\epsilon_{\mathrm{est}}|` for exact fits. 
        However, for regression fits of data they are not equivalent; therefore,
        this code implements the GCI uncertainty measure as

        .. math::
            GCI = Fs * |\\epsilon_{\mathrm{est}}|,

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
        P. J. Roache, "Perspective: A Method for Uniform Reporting of Grid
        Refinement Studies," Journal of Fluids Engineering, 116:3 (1994).
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
                    index: int | None = None,
                    significance: float=0.05,
    ) -> np.floating | pd.Series:
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
                    index: int | None = None,
                    factor: int | float=3,
    ) -> np.floating | pd.Series:
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

###############################################################################
# DiscretizationError
###############################################################################
class DiscretizationError(ABC):
    """Abstract factory for discretization error classes"""

    def __init__(self,
                 arg1: list | tuple | np.ndarray | pd.Series | dict | pd.DataFrame,
                 arg2: list | tuple | np.ndarray | pd.Series | dict | str | None = None,
                **kwargs,
    ) -> None:
        """Class constructor
        
        Parameters
        ----------
        arg1 : list | tuple | np.ndarray | pd.Series | dict | pd.DataFrame
            Discretization levels (list | tuple) or data (dict | pd.DataFrame)
        arg2 : list | tuple | np.ndarray | pd.Series | dict | str | None
            System reponse quantities (list | tuple), mesh key (str), or None
        """
        # Create class data from arguments
        self._assign_data(arg1, arg2)

        # Define models for class instance
        self.model = self.create_model()
        self.error = self.create_error()
        self.uncertainty = self.create_uncertainty()
        self.u = self.uncertainty # Alias for user experience

        # Define solved state attributes
        self.f_est = self.model.f_est()
        self.order = self.model.order()
        self.parameters = self.model.parameters

    @abstractmethod
    def create_model(self) -> DiscretizationModel:
        """Create discretization model for analysis
        
        Returns
        -------
        : DiscretizationModel
            Discretization model for the data
        """
        pass

    @abstractmethod
    def create_error(self) -> ErrorModel:
        """Create error model for analysis
        
        Returns
        -------
        : ErrorModel
            Error model for the data
        """
        pass
    
    @abstractmethod
    def create_uncertainty(self) -> UncertaintyModel:
        """Create uncertainty model for analysis
        
        Returns
        -------
        : UncertaintyModel
            Uncertainty model for the data
        """
        pass

    # Special methods
    def __len__(self) -> int:
        """Return number of discretization levels
        
        Returns
        -------
        : int
            Number of discretization levels
        """
        return len(self.hs)
        
    # Private methods #########################################################
    def _assign_data(self,
                     arg1: list | tuple | np.ndarray | pd.Series | dict | pd.DataFrame,
                     arg2: list | tuple | np.ndarray | pd.Series | dict | str | None = None,
    ) -> None:
        """Assign class data attributes based on input types
        
        Parameters
        ----------
        arg1 : list | tuple | np.ndarray | pd.Series | dict | pd.DataFrame
            Discretization levels (list | tuple) or data (dict | pd.DataFrame)
        arg2 : list | tuple | np.ndarray | pd.Series | dict | str | None
            System reponse quantities (list | tuple), mesh key (str), or None
        """
        if type(arg1) in [list, tuple, np.ndarray, pd.Series]:
            # Assign mesh sizes based on data type
            if type(arg1) in [list, tuple]:
                self.hs_key ="hs"

            elif type(arg1) is np.ndarray:
                if len(np.squeeze(arg1).shape) != 1:
                    raise ValueError("Numpy array of discretization sizes must be 1 dimensional!")
                self.hs_key ="hs"

            elif type(arg1) is pd.Series:
                if arg1.name is None:
                    self.hs_key = "hs"
                else:
                    self.hs_key = str(arg1.name)

            else:
                raise TypeError("Invalid type for first argument!")
                
            self.hs = pd.Series(arg1, name=self.hs_key)

            # Assign response data based on type
            if type(arg2) in [list, tuple, np.ndarray]:
                self.keys = ("System Response Quantity",)
                self.data = pd.DataFrame({self.keys[0]: arg2})

            elif type(arg2) is dict:
                self.keys = tuple(arg2.keys())
                self.data = pd.DataFrame(arg2)

            elif type(arg2) is pd.Series:
                if arg2.name is None:
                    self.keys = ("System Response Quantity",)
                else:
                    self.keys = (str(arg2.name),)
                self.data = pd.DataFrame({self.keys[0]: arg2})
            else:
                raise TypeError("Second argument must be a list, tuple, numpy.ndarray, pandas.Series, or dict when first argument is a list, tuple, numpy.ndarray, pandas.Series, or dict!")

        elif type(arg1) is dict and len(arg1.keys()) == 1:
            self.hs_key = list(arg1.keys())[0]
            self.hs = pd.Series(arg1[self.hs_key], name=self.hs_key)

            # Assign response data based on type
            if type(arg2) in [list, tuple, np.ndarray]:
                self.keys = ("System Response Quantity",)
                self.data = pd.DataFrame({self.keys[0]: arg2})

            elif type(arg2) is dict:
                self.keys = tuple(arg2.keys())
                self.data = pd.DataFrame(arg2)

            elif type(arg2) is pd.Series:
                if arg2.name is None:
                    self.keys = ("System Response Quantity",)
                else:
                    self.keys = (str(arg2.name),)
                self.data = pd.DataFrame({self.keys[0]: arg2})
            else:
                raise TypeError("Second argument must be a list, tuple, numpy.ndarray, pandas.Series, or dict when first argument is singular dictionary!")

        elif type(arg1) is dict:
            if arg2 is None:
                mesh_key = "hs"
            elif type(arg2) is str:
                mesh_key = arg2
            else:
                raise TypeError("Second argument must be a string if first argument is a dict!")
            if mesh_key in arg1.keys():
                data_dict = arg1.copy()
                self.hs_key = mesh_key
                self.hs = pd.Series(data_dict.pop(self.hs_key),
                                    name=self.hs_key)
                self.keys = tuple(data_dict.keys())
                self.data = pd.DataFrame(data_dict)
            else:
                raise ValueError(f"{mesh_key} key not found in dict for discretization levels!")
            
        elif type(arg1) is pd.DataFrame:
            if arg2 is None:
                mesh_key = "hs"
            elif type(arg2) is str:
                mesh_key = arg2
            else:
                raise TypeError("Second argument must be a string if first argument is a dict!")
            if mesh_key in arg1.keys():
                data_dict = arg1.copy()
                self.hs_key = mesh_key
                self.hs = pd.Series(data_dict.pop(self.hs_key),
                                    name=self.hs_key)
                self.keys = tuple(data_dict.keys())
                self.data = pd.DataFrame(data_dict)
            else:
                raise ValueError(f"{mesh_key} key not found in dict for discretization levels!")

        else:
            raise TypeError("Invalid type for first argument. Valid types are list, tuple, np.ndarray, pd.Series, dict, or Pandas.DataFrame") 

        # Sort data from smallest discretization size to largest
        self._sort()
        # Define common attributes
        self._compute_refinement_ratios()

    def _compute_refinement_ratios(self):
        """Compute refinement ratios of data

        The refinement ratio is defined as the coarse mesh size divided by the
        fine mesh size, or

        .. math::
            r_i = \\frac{h_{i+1}}{h_i},

        for all discretization levels except the coarsest. For the coarsest 
        mesh (n), the refinement ratio of the next finer mesh is used as

        .. math::
            r_n = \\frac{h_n}{h_{n-1}}

        """
        rrs = []
        for idx in range(0, len(self)-1):
            rrs.append(self.hs[idx+1] / self.hs[idx])
        self.refinement_ratios = tuple(rrs)

    def _sort(self):
        """Sort discretization data from smallest to largest size"""
        idx = self.hs.sort_values().index
        self.hs = self.hs.loc[idx].reset_index(drop=True)
        self.data = self.data.loc[idx].reset_index(drop=True)

    # Data methods ############################################################
    def estimated_error(self,
                        key: str | None = None,
                        index: int | None = None,
    ) -> np.floating | pd.Series | pd.DataFrame:
        """Compute estimated error for data

        Parameters
        ----------
        key : str, optional
            Key for system response quantity of interest, by default None
        index : int, optional
            Index for level of interest, by default None

        Returns
        -------
        np.floating | pd.Series | pd.DataFrame
            Estimated error of quantities of interest
        """
        if key is None:
            data = self.data
            f_est = self.f_est
        else:
            data = self.data[key]
            f_est = self.f_est[key]

        if index is None:
            est_err = data - f_est
        else:
            est_err = data.iloc[index] - f_est

        return est_err
    
    def abs_estimated_error(self,
                       key: str=None,
                       index: int=None,
    ) -> np.floating | pd.Series | pd.DataFrame:
        """Compute absolute estimated error for data

        Parameters
        ----------
        key : str, optional
            Key for system response quantity of interest, by default None
        index : int, optional
            Index for level of interest, by default None

        Returns
        -------
        np.floating | pd.Series | pd.DataFrame
            Absolute estimated error of quantities of interest
        """
        
        return abs(self.estimated_error(key, index))
    
    def relative_error(self,
                       key: str | None = None,
                       index: int | None = None,
    ) -> np.floating | pd.Series | pd.DataFrame:
        """Compute error relative to coarser discretization level

        Errors for all but the coarsest level are computed as

        .. math::
            \\epsilon_i = f_i - f_{i+1},

        while the error for the coarsest level is computed as

        .. math::
            \\epsilon_i = f_{i-1} - f_{i}.

        Parameters
        ----------
        key : str | None
            Key for system response quantity of interest or None for all
        index : int | None
            Index for level of interest or None for all

        Returns
        -------
        rel_err : np.floating | pd.Series | pd.DataFrame
            Relative error of quantities of interest
        """
        if key is None:
            data = self.data
        else:
            data = self.data[key]

        if index is None:
            rel_err = data.diff(-1)
            # Define relative error for last mesh as same as previous mesh
            rel_err.iloc[-1] = rel_err.iloc[-2]
        elif index != len(self) - 1:
            rel_err = data[index:index+2].diff(-1).iloc[0]
        else:
            # Define relative error for last mesh as same as previous mesh
            rel_err = data[index-1:index+1].diff(-1).iloc[0]

        return rel_err
    
    def abs_relative_error(self,
                       key: str=None,
                       index: int=None,
    ) -> np.floating | pd.Series | pd.DataFrame:
        """Compute absolute error relative to coarser discretization level

        Errors for all but the coarsest level are computed as

        .. math::
            \\epsilon_i = |f_i - f_{i+1}|,

        while the error for the coarsest level is computed as

        .. math::
            \\epsilon_i = |f_{i-1} - f_{i}|.

        Parameters
        ----------
        key : str | None
            Key for system response quantity of interest or None for all
        index : int | None
            Index for level of interest or None for all

        Returns
        -------
        rel_err : np.floating | pd.Series | pd.DataFrame
            Absolute relative error of quantities of interest
        """
        return abs(self.relative_error(key, index))

    # Output methods ##########################################################
    def plot(self,
             key: str | None = None,
             index : int = 0,
             filename: str="DiscretizationError.png",
             *,
             title: str=None,
             xlabel: str=None,
             ylabel: str=None,
             error: bool=True,
             uncertainty: bool=True,
    ) -> None:
        """Plot system response quantity data and model and save figure

        If the key is not provided, the first key in the data is used
        
        Parameters
        ----------
        key : str
            Key of system response quantity of interest
        index : int
            Index of interest in the study
        filename : str
            Name of file to save figure to
        title : str
            (Optional) Title of plot
        xlabel : str
            (Optional) X-axis label
        ylabel : str
            (Optional) Y-axis label
        error : bool
            (Optional) Plot error bar
        uncertainty : bool
            (Optional) Plot uncertainty bar
        """
        if key is None:
            key = self.keys[0]

        fig, ax = plt.subplots()
        # Plot data
        ax.plot(self.hs, self.data[key], 'o', color="k", label="Data")
        # Plot estimate with its associated model
        ax.plot(0, self.f_est[key], "o", color="#1f77b4", label="Estimate")
        hs = np.linspace(0, self.hs.values[-1])
        ax.plot(hs, self.model.model(key, hs), "--", color="#1f77b4",
                label="Model")
        # Plot error
        if error or uncertainty:
            fill_hs = np.array([0, self.hs.values[index]])
            err = abs(self.error(key, index)) * np.ones(fill_hs.shape)
            val = self.data.loc[index, key] * np.ones(fill_hs.shape)
            err_low = val - err
            err_high = val + err
        if error:
            ax.fill_between(fill_hs, err_low, err_high, color="#ff7f0e",
                            alpha=0.25, edgecolor=None, label="Error")
        # Plot uncertainty
        if uncertainty:
            unc = self.uncertainty(key, index) * np.ones(fill_hs.shape)
            val = self.data.loc[index, key] * np.ones(fill_hs.shape)
            unc_low = val - unc
            unc_high = val + unc
            ax.fill_between(fill_hs, err_high, unc_high, color="#ffe119",
                            alpha=0.25, edgecolor=None)
            ax.fill_between(fill_hs, unc_low, err_low, color="#ffe119",
                            alpha=0.25, edgecolor=None, label="Uncertainty")
            
        # Annotate and save
        ax.set_xlim(left=0)
        if xlabel is None:
            ax.set_xlabel("Discretization Size")
        else:
            ax.set_xlabel(xlabel)
        if ylabel is None:
            ax.set_ylabel("System Response Quantity")
        else:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.legend()
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    
    def summarize(self, key: str | None = None) -> None:
        """Summarize the solution verification data

        If no key is provided, the first key in the data is used
        
        Parameters
        ----------
        key : str
            Key of system response quantity of interest
        """
        if key is None:
            key = self.keys[0]

        print(f"Mesh Size \t {key}")
        print("--------- \t ---------")
        for h, f in zip(self.hs, self.data[key]):
            print(f"{h:9.4g} \t {f:9.4g}")
        print(f"Extrapolated Value: {self.f_est[key]:.6g}")
        print(f"Fine mesh error: {self.error(key, 0):.6g}")
        print(f"Fine mesh uncertainty: {self.u(key, 0):.6g}")

    def export(self, filename: str="DiscretizationData.csv") -> None:
        """Export data for later processing

        Parameters
        ----------
        filename : str
            Name of file to export data to
        """
        export_data = pd.concat([self.data, self.parameters])
        export_data.to_csv(filename, index_label="Index")

###############################################################################
# Concrete classes
###############################################################################
class CustomDiscretizationError(DiscretizationError):
    """Discretization error class for custom implementations"""

    def __init__(self,
                 arg1: list | tuple | np.ndarray | pd.Series | dict | pd.DataFrame,
                 arg2: list | tuple | np.ndarray | pd.Series | dict | str | None = None,
                 model: DiscretizationModel=SinglePower,
                 error: ErrorModel=EstimatedError,
                 uncertainty: UncertaintyModel=GCI,
    ) -> None:
        """Class constructor
        
        Parameters
        ----------
        arg1 : list | tuple | np.ndarray | pd.Series | dict | pd.DataFrame
            Discretization levels (list | tuple) or data (dict | pd.DataFrame)
        arg2 : list | tuple | np.ndarray | pd.Series | dict | str | None
            System reponse quantities (list | tuple), mesh key (str), or None
        model : DiscretizationModel
            Discretization model class to use for analysis
        error : ErrorModel
            Error model class to use for analysis
        uncertainty : UncertaintyModel
            Uncertainty model class to use for analysis
        """
        # Create class data from arguments
        self._assign_data(arg1, arg2)

        # Define models for class instance
        self.model = self.create_model(model)
        self.error = self.create_error(error)
        self.uncertainty = self.create_uncertainty(uncertainty)
        self.u = self.uncertainty # Alias for easier use

        # Define solved state attributes
        self.f_est = self.model.f_est()
        self.order = self.model.order()
        self.parameters = self.model.parameters

    def create_model(self, model: DiscretizationModel) -> DiscretizationModel:
        """Create discretization model for analysis

        Parameters
        ----------
        model : DiscretizationModel
            Discretization model to use for analysis
        
        Returns
        -------
        : DiscretizationModel
            Discretization model for the data
        """
        return model(self)
    
    def create_error(self, model: ErrorModel) -> ErrorModel:
        """Create error model for analysis

        Parameters
        ----------
        model : ErrorModel
            Error model to use for analysis
        
        Returns
        -------
        : ErrorModel
            Error model for the data
        """
        return model(self)
    
    def create_uncertainty(self, model: UncertaintyModel) -> UncertaintyModel:
        """Create uncertainty model for analysis

        Parameters
        ----------
        model : UncertaintyModel
            Uncertainty model to use for analysis
        
        Returns
        -------
        : UncertaintyModel
            Uncertainty model for the data
        """
        return model(self)

class Classic(DiscretizationError):
    """Discretization error class consistent with ASME V&V 20 standard"""

    def create_model(self) -> DiscretizationModel:
        """Create SinglePower discretization model for analysis
        
        Returns
        -------
        : DiscretizationModel
            Discretization model for the data
        """
        return SinglePower(self)
    
    def create_error(self) -> ErrorModel:
        """Create EstimatedError error model for analysis
        
        Returns
        -------
        : ErrorModel
            Error model for the data
        """
        return EstimatedError(self)
    
    def create_uncertainty(self) -> UncertaintyModel:
        """Create GCI uncertainty model for analysis
        
        Returns
        -------
        : UncertaintyModel
            Uncertainty model for the data
        """
        return GCI(self)
    
class Average(DiscretizationError):
    """Discretization error class using average value of responses"""

    def create_model(self) -> DiscretizationModel:
        """Create AverageValue discretization model for analysis
        
        Returns
        -------
        : DiscretizationModel
            Discretization model for the data
        """
        return AverageValue(self)
    
    def create_error(self) -> ErrorModel:
        """Create EstimatedError error model for analysis
        
        Returns
        -------
        : ErrorModel
            Error model for the data
        """
        return EstimatedError(self)
    
    def create_uncertainty(self) -> UncertaintyModel:
        """Create StudentsTDistribution uncertainty model for analysis
        
        Returns
        -------
        : UncertaintyModel
            Uncertainty model for the data
        """
        return StudentsTDistribution(self)