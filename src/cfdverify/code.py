from collections.abc import Iterable
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot options
# Grid
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "grey"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["axes.axisbelow"] = True


class OrderOfAccuracy:
    """Code verification data as well as analysis and output methods"""


    def __init__(self, data: pd.DataFrame, relative_error: bool=False) -> None:
        """Default constructor
        
        Parameters
        ----------
        data : pd.DataFrame
            Code convergence data in a pandas DataFrame. Index values are the 
            discretization size of the data and column keys are the responses 
            to analyze.
        """
        # Format data
        self.data = data.sort_index(ascending=False)
        self.keys = self.data.keys().to_list()
        # Label index if it isn't already named
        if self.data.index.name is None:
            self.data.index.name = '\u0394'
        # If using a benchmark (gold) solution, convert values into errors
        if relative_error:
            self.reference_result = self.data.iloc[-1]
            self.reference_size = self.data.index[-1]
            self.data = (self.data - self.reference_result).drop(self.data.index[-1])
        
        # Compute derived quantities
        self.r = self.compute_refinement_ratios()
        self.p_hat = self.compute_orders()

        # Create a single dataframe for export
        # Rename columns as multi-index
        error_label = ["Error"] * len(self.keys)
        error_multi_index = pd.MultiIndex.from_arrays([self.keys, error_label],
                                                      names=["Error Measures", "Parameters"])
        data_to_combine = self.data.copy()
        data_to_combine.columns = error_multi_index

        order_label = ["Order"] * len(self.keys)
        order_multi_index = pd.MultiIndex.from_arrays([self.keys, order_label],
                                                      names=["Error Measures", "Parameters"])
        order_to_combine = self.p_hat.copy()
        order_to_combine.columns = order_multi_index
        # Combine dataframes
        self.combined = pd.concat([data_to_combine, order_to_combine], axis=1)
        self.combined.sort_index(axis=1, inplace=True)

    
    def compute_refinement_ratios(self) -> pd.Series:
        """Computes the refinement ratios of the data
        
        Returns
        -------
        pd.Series
            Refinement ratios of error pairs. Coarsest result is undefined.
        """
        ratios = (self.data.index[:-1] / self.data.index[1:]).to_list()
        ratios.insert(0, np.nan)

        return pd.Series(ratios, index=self.data.index, name="Refinement Ratio")


    def compute_orders(self) -> pd.DataFrame:
        """Computes the observed order of the data
        
        Returns
        -------
        pd.DataFrame
            Observed convergence orders for error pairs. Coarsest result is undefined.
        """
        orders = {}
        for key in self.data:
            coarse = self.data[key].iloc[:-1]
            fine = self.data[key].iloc[1:]
            p_hat = self._observed_order(coarse.values, fine.values, self.r.iloc[1:].values)
            orders[self._order_key(key)] = np.insert(p_hat, 0, np.nan)

        return pd.DataFrame(orders, index=self.data.index)
    

    def _observed_order(
            self,
            coarse_error: float | np.ndarray,
            fine_error: float | np.ndarray,
            refinement_ratio: float | np.ndarray,
    ) -> float | np.ndarray:
        """Computes observed order of the data
        
        See Eq. 5.22 in Oberkampf and Roy 2010

        Parameters
        ----------
        coarse_error : float | np.ndarray
            The error of the coarse mesh pair or array of pairs.
        fine_error : : float | np.ndarray
            The error of the fine mesh pair or array of pairs.
        refinement_ratio : float | np.ndarray
            The refinement ratio of the error pairs or array of pairs.

        Returns
        -------
        float
            Observed convergence order for error pairs
        """
        return np.log(coarse_error / fine_error) / np.log(refinement_ratio)
    

    def _order_key(self, key: str) -> str:
        """Name observed order for key
        
        Parameters
        ----------
        key : str
            Key of column in pd.DataFrame

        Returns
        -------
        str
            Name for observed order of key
        """
        return str(key)+" Order"
    

    # Generic methods #########################################################
    def get_average_orders(self) -> pd.Series:
        """Get the mean order of accuracy for each response
        
        Returns
        -------
        pd.Series
            Mean order of accuracy for each response. Doesn't include NaNs.
        """
        return self.p_hat.mean()


    # Plotting methods ########################################################
    def plot_response(
            self, key: str,
            plot_theoretical_orders: bool=True,
            xlabel: str | None=None,
            ylabel: str | None=None,
            title: str | None=None,
            save_figure: str | os.PathLike | bool | None=None,
    ) -> None:
        """Plot specified key to visualize error convergence
        
        Parameters
        ----------
        key : str
            Key of response to plot
        plot_theoretical_orders : bool
            Plot theoretical order lines for comparison. Defaults to true.
        xlabel : str | None
            Label for x axis. Defaults to index name.
        ylabel : str | None
            Label for y axis. Defaults to response key.
        title : str | None
            Title for plot. Defaults to None.
        save_figure: str | os.PathLike | bool | None
            Save figure to file if true. Defaults to None, which doesn't save.
            If a string or os.PathLike, saves to specified name. If True, saves
            to "{key}_Convergence.png"
        """
        order_key = self._order_key(key)
        mean_p_hat = self.p_hat[order_key].mean()

        fig, ax = plt.subplots()
        ax.loglog(self.data.index,
                  self.data[key],
                  marker='o',
                  linestyle='dashed',
                  label=rf"$\overline{{\hat{{p}}}}={mean_p_hat:.4n}$")

        if plot_theoretical_orders:
            floor = math.floor(mean_p_hat)
            ceil = math.ceil(mean_p_hat)
            if floor == ceil:
                floor = ceil - 1
            x_lims = ax.get_xlim()
            xs = np.linspace(x_lims[0], x_lims[1])
            # Compute theoretical order responses using power law with offset
            y_floor = (self.data[key].iloc[0]*1.15) * xs**floor
            y_ceil = (self.data[key].iloc[0]*0.85) * xs**ceil
            # Ensure the solid line is the closest order to the data
            if abs(floor-mean_p_hat) <= abs(ceil-mean_p_hat):
                ax.loglog(xs, y_floor, color="black", linestyle="solid", label=rf"$\hat{{p}}={floor}$")
                ax.loglog(xs, y_ceil, color="black", linestyle="dashdot", label=rf"$\hat{{p}}={ceil}$")
            else:
                ax.loglog(xs, y_floor, color="black", linestyle="dashdot", label=rf"$\hat{{p}}={floor}$")
                ax.loglog(xs, y_ceil, color="black", linestyle="solid", label=rf"$\hat{{p}}={ceil}$")

        # Apply plot annotations
        if xlabel:
            ax.set_xlabel(xlabel)
        elif self.data.index.name is not None:
            ax.set_xlabel(str(self.data.index.name))

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(key)

        if title:
            ax.set_title(title)

        ax.legend()

        # Save figure, if desired
        if isinstance(save_figure, (str, os.PathLike)):
            fig.savefig(save_figure, bbox_inches="tight", dpi=300)
        elif save_figure:
            figure_title = f"{key}_Convergence.png"
            fig.savefig(figure_title, bbox_inches="tight", dpi=300)


    def plot_responses(
            self,
            keys : Iterable | None=None,
            plot_theoretical_orders=True,
            xlabel: str | None=None,
            ylabel: str | None=None,
            title: str | None=None,
            save_figure: str | os.PathLike | bool | None=None,
    ) -> None:
        """Plot specified or all keys to visualize error convergence
        
        Parameters
        ----------
        keys : Iterable | None
            Keys of responses to plot. Defaults to all responses.
        plot_theoretical_orders : bool
            Plot theoretical order lines for comparison. Defaults to true.
        xlabel : str | None
            Label for x axis. Defaults to index name.
        ylabel : str | None
            Label for y axis. Defaults to response key.
        title : str | None
            Title for plot. Defaults to None.
        save_figure: str | os.PathLike | bool | None
            Save figure to file if true. Defaults to None, which doesn't save.
            If a string or os.PathLike, saves to specified name. If True, saves
            to "Convergences.png"
        """
        if keys is None:
            keys = self.keys
            
        mean_p_hats = {}
        for key in keys:
            order_key = self._order_key(key)
            mean_p_hats[key] = self.p_hat[order_key].mean()

        fig, ax = plt.subplots()
        for key in keys:
            ax.loglog(self.data.index,
                      self.data[key],
                      marker='o',
                      linestyle='dashed',
                      label=rf"{key}: $\overline{{\hat{{p}}}}={mean_p_hats[key]:.4n}$")

        if plot_theoretical_orders:
            # Set orders as closest to mean and 1 less
            mean_p_hat = sum(mean_p_hats.values()) / len(mean_p_hats)
            ceil = round(mean_p_hat)
            floor = ceil - 1           
            x_lims = ax.get_xlim()
            xs = np.linspace(x_lims[0], x_lims[1])
            max_value = self.data.iloc[0].max()
            # Compute theoretical order responses using power law with offset
            y_floor = (max_value*1.15) * xs**floor
            y_ceil = (max_value*1.15) * xs**ceil
            ax.loglog(xs, y_floor, color="black", linestyle="dashdot", label=rf"$\hat{{p}}={floor}$")
            ax.loglog(xs, y_ceil, color="black", linestyle="solid", label=rf"$\hat{{p}}={ceil}$")

        # Apply plot annotations
        if xlabel:
            ax.set_xlabel(xlabel)
        elif self.data.index.name is not None:
            ax.set_xlabel(str(self.data.index.name))

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("Log(X)")

        if title:
            ax.set_title(title)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        # Save figure, if desired
        if isinstance(save_figure, (str, os.PathLike)):
            fig.savefig(save_figure, bbox_inches="tight", dpi=300)
        elif save_figure:
            figure_title = f"Convergences.png"
            fig.savefig(figure_title, bbox_inches="tight", dpi=300)


    def plot_order(
            self,
            key,
            xlabel: str | None=None,
            ylabel: str | None=None,
            title: str | None=None,
            save_figure: str | os.PathLike | bool | None=None,
    ) -> None:
        """Plot specified key to visualize order of convergence
        
        Parameters
        ----------
        key : str
            Key of response to plot
        xlabel : str | None
            Label for x axis. Defaults to index name.
        ylabel : str | None
            Label for y axis. Defaults to response key.
        title : str | None
            Title for plot. Defaults to None.
        save_figure: str | os.PathLike | bool | None
            Save figure to file if true. Defaults to None, which doesn't save.
            If a string or os.PathLike, saves to specified name. If True, saves
            to "{key}_OrderOfConvergence.png"
        """
        order_key = self._order_key(key)
        mean_p_hat = self.p_hat[order_key].mean()

        fig, ax = plt.subplots()
        ax.semilogx(self.data.index,
                    self.p_hat[order_key],
                    marker='o',
                    linestyle='dashed',
                    label=rf"$\overline{{\hat{{p}}}}={mean_p_hat:.4n}$")

        # Apply plot annotations
        if xlabel:
            ax.set_xlabel(xlabel)
        elif self.data.index.name is not None:
            ax.set_xlabel(str(self.data.index.name))

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(r"Observed Order of Accuracy ($\hat{p}$)")

        if title:
            ax.set_title(title)

        ax.legend()

        # Save figure, if desired
        if isinstance(save_figure, (str, os.PathLike)):
            fig.savefig(save_figure, bbox_inches="tight", dpi=300)
        elif save_figure:
            figure_title = f"{key}_OrderOfConvergence.png"
            fig.savefig(figure_title, bbox_inches="tight", dpi=300)


    def plot_orders(
            self,
            keys : Iterable | None=None,
            xlabel: str | None=None,
            ylabel: str | None=None,
            title: str | None=None,
            save_figure: str | os.PathLike | bool | None=None,
    ) -> None:
        """Plot specified or all keys to visualize order of convergence
        
        Parameters
        ----------
        keys : Iterable | None
            Keys of responses to plot. Defaults to all responses.
        xlabel : str | None
            Label for x axis. Defaults to index name.
        ylabel : str | None
            Label for y axis. Defaults to response key.
        title : str | None
            Title for plot. Defaults to None.
        save_figure: str | os.PathLike | bool | None
            Save figure to file if true. Defaults to None, which doesn't save.
            If a string or os.PathLike, saves to specified name. If True, saves
            to "OrdersOfConvergence.png"
        """
        if keys is None:
            keys = self.keys

        mean_p_hats = {}
        for key in keys:
            order_key = self._order_key(key)
            mean_p_hats[key] = self.p_hat[order_key].mean()

        fig, ax = plt.subplots()
        for key in keys:
            ax.semilogx(self.data.index,
                        self.p_hat[self._order_key(key)],
                        marker='o',
                        linestyle='dashed',
                        label=rf"{key}: $\overline{{\hat{{p}}}}={mean_p_hats[key]:.4n}$")

        # Apply plot annotations
        if xlabel:
            ax.set_xlabel(xlabel)
        elif self.data.index.name is not None:
            ax.set_xlabel(str(self.data.index.name))

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(r"Observed Order of Accuracy ($\hat{p}$)")

        if title:
            ax.set_title(title)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3) 

        # Save figure, if desired
        if isinstance(save_figure, (str, os.PathLike)):
            fig.savefig(save_figure, bbox_inches="tight", dpi=300)
        elif save_figure:
            figure_title = f"OrdersOfConvergence.png"
            fig.savefig(figure_title, bbox_inches="tight", dpi=300)


    # Output methods ##########################################################
    def print(self) -> None:
        """Print results to console for command line interface"""
        print(self.combined)


    def export(self, filename: str | os.PathLike, type: str="csv") -> os.PathLike:
        """Export order of accuracy study results to a file in tabular format

        Parameters
        ----------
        filename : str | os.PathLike
            Name of file to write data to
        type : str
            Type of file to write to. Currently supported types are comma 
            separated variable (csv) and latex.

        Returns
        -------
        filename : os.PathLike
            Path object data was written to
        """
        # Hold valid types
        VALID_TYPES = ["csv", "latex"]
        # Ensure filename is a Path object
        filename = Path(filename)

        if type == "csv":
            self.combined.to_csv(filename)
        elif type == "latex":
            self.combined.to_latex(filename)
        else:
            msg = "Unsupported type!\n"
            msg += f"Supported types are: {VALID_TYPES}\n"
            msg += "Look at pandas.DataFrame API for manual options."
            raise ValueError(msg)

        return filename
    