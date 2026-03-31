"""
This script explores the common attributes and methods of the discretization
model that a user may want to interact with.
"""
# %%
## Import modules and create model
from cfdverify.discretization import Classic
mesh_sizes = [0.00292402, 0.00414913, 0.00573555] # m
pressure_drop = [100, 98, 95] # kPa
max_vel = [8, 8.5, 9.2] # m/s
model = Classic(mesh_sizes, {"DP": pressure_drop, "Vel": max_vel})

# %%
## Mesh data
# The key (name) for the discretization data
print(f"Mesh key is: {model.hs_key}")
# The discretization values are stored as a Pandas Series
print(f"Mesh sizes are: \n{model.hs}")

# %%
## System response quantity data
# The keys (names) for the responses are stored in a tuple
print(f"Response keys are: {model.keys}")
# The response data is stored in a Pandas DataFrame
print(f"Response data are: \n{model.data}")

# %%
## Data manipulation methods
# Get the number of discretization levels being investigated
print(f"Number of meshes: {len(model)}")
# Get the relative error of simulation responses
print(f"Relative error of DP: \n{model.relative_error('Vel')}")
# Or you can get the absolute relative error
print(f"Absolute relative error of DP: \n{model.abs_relative_error('Vel')}")

# %%
## Data interpretation
# Summaries of data can be generated per response key
model.summarize("Vel")
# Plots can be generated per response key and customized
model.plot(key="Vel",
           filename="ExamplePlot.png",
           title="My Velocity Solution Verification Results",
           xlabel="Mesh Size (m)",
           ylabel="Velocity (m/s)")
# Data can be exported for use in tables in CSV format
model.export("ModelData.csv")
# %%
