from abc import ABC, abstractmethod
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .error import ErrorModel, EstimatedError
from .model import AverageValue, DiscretizationModel, SinglePower
from .uncertainty import GCI, StudentsTDistribution, UncertaintyModel

###############################################################################
# DiscretizationError
###############################################################################
class DiscretizationError(ABC):
    """Abstract factory for discretization error classes"""

    def __init__(self,
                 arg1: Union[list, tuple, np.ndarray, pd.Series, dict, pd.DataFrame],
                 arg2: Union[list, tuple, np.ndarray, pd.Series, dict, str, None] = None,
                **kwargs,
    ) -> None:
        """Class constructor

        Parameters
        ----------
        arg1 : list | tuple | np.ndarray | pd.Series | dict | pd.DataFrame
            Discretization levels (list | tuple) or data (dict | pd.DataFrame)
        arg2 : list | tuple | np.ndarray | pd.Series | dict | str | None
            System response quantities (list | tuple), mesh key (str), or None
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
                     arg1: Union[list, tuple, np.ndarray, pd.Series, dict, pd.DataFrame],
                     arg2: Union[list, tuple, np.ndarray, pd.Series, dict, str, None] = None,
    ) -> None:
        """Assign class data attributes based on input types

        Parameters
        ----------
        arg1 : list | tuple | np.ndarray | pd.Series | dict | pd.DataFrame
            Discretization levels (list | tuple) or data (dict | pd.DataFrame)
        arg2 : list | tuple | np.ndarray | pd.Series | dict | str | None
            System response quantities (list | tuple), mesh key (str), or None
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
                        key: Union[str, None] = None,
                        index: Union[int, None] = None,
    ) -> Union[np.floating, pd.Series, pd.DataFrame]:
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
    ) -> Union[np.floating, pd.Series, pd.DataFrame]:
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
                       key: Union[str, None] = None,
                       index: Union[int, None] = None,
    ) -> Union[np.floating, pd.Series, pd.DataFrame]:
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
    ) -> Union[np.floating, pd.Series, pd.DataFrame]:
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
             key: Union[str, None] = None,
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
            val = self.data.loc[index, key]
            err_min = min(val, self.f_est[key])
            err_max = max(val, self.f_est[key])
        if error:
            ax.fill_between(fill_hs, err_min, err_max, color="#ff7f0e",
                            alpha=0.25, edgecolor=None, label="Error")
        # Plot uncertainty
        if uncertainty:
            unc = self.uncertainty(key, index) * np.ones(fill_hs.shape)
            val = self.data.loc[index, key] * np.ones(fill_hs.shape)
            unc_low = val - unc
            unc_high = val + unc
            ax.fill_between(fill_hs, err_max, unc_high, color="#ffe119",
                            alpha=0.25, edgecolor=None)
            ax.fill_between(fill_hs, unc_low, err_min, color="#ffe119",
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

    def summarize(self, key: Union[str, None] = None) -> None:
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
                 arg1: Union[list, tuple, np.ndarray, pd.Series, dict, pd.DataFrame],
                 arg2: Union[list, tuple, np.ndarray, pd.Series, dict, str, None] = None,
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
            System response quantities (list | tuple), mesh key (str), or None
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
