from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .discretization_error import DiscretizationError

import numpy as np
import pandas as pd

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
              key: Union[str, None] = None,
              index: Union[int, None] = None,
    ) -> Union[np.floating, pd.Series, pd.DataFrame]:
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

    def get_data(self, key: Union[str, None]) -> Union[pd.Series, pd.DataFrame]:
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
              key: Union[str, None] = None,
              index: Union[int, None] = None,
    ) -> Union[np.floating, pd.Series, pd.DataFrame]:
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
