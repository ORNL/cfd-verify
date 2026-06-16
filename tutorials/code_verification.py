# %% Import modules
import pandas as pd

from cfdverify.code import OrderOfAccuracy


# %% Create data to analyze
# First, you need the size of the discretizations considered. Here, the average
# cell size.
mesh_sizes = [0.001, 0.0005, 0.00025, 0.000125] # [m]

## Next, collate the simulation responses
# Global measures, like the L2 norm of the error, should be used in all studies
l2_norm = [1e-4, 2.525e-5, 6.35e-6, 1.609375e-6]
# Local measures of specific quantities of interest, like the average pressure 
# on a surface, should also be considered when appropriate
avg_pressure = [0.11, 0.026_2, 0.006_312_5, 0.001_531_25]

# Then, create your DataFrame using your favorite method
data = pd.DataFrame({"L2 Norm": l2_norm,
                     "Pressure": avg_pressure},
                     mesh_sizes)
data.index.name = '\u0394h (m)' # Labeling your index can make prettier plots

# %% Create order of accuracy object and report results
ooa = OrderOfAccuracy(data)
ooa.print() # Print results to console for command line interfaces
ooa.plot_responses() # Plot log-log error easily
ooa.plot_order("Pressure") # Plot orders to directly visualize convergence
file = ooa.export("MyConvergenceData.tex", "latex") # And, export results directly
print(f"Results saved to: {file.resolve()}")
