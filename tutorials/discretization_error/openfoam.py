"""
This script can be used with the OpenFOAM discretization error tutorial. When
properly modified for your system, it will extract the relevant files from the
completed cases, load them into a Pandas DataFrame, and then estimate the 
discretization error using CFDverify.

Lines that should be modified are commented with `# MODIFY`, which you can 
search in a text editor for easy modification.

Sections are commented with `# %%`, so this script can be run interactively as
Jupyter code blocks if using VS Code.
"""
# %% Import modules
from pathlib import Path

import pandas as pd

import cfdverify.utils as utils
from cfdverify.discretization import Classic


# %% Read data from each mesh case
base_dir = Path("/home/justin/cfd/bfs") # MODIFY
cases = ["bfs1", "bfs2", "bfs4"]

cp_data = {}
for case in cases:
    cp_file = base_dir / case / "postProcessing" / "sampleCp" / "0" / "cp"
    data = pd.read_csv(cp_file, comment="#", sep=r"\s+", names=["Time", "Cp"])
    cp_data[case] = data.set_index("Time")["Cp"]

cp = pd.DataFrame(cp_data)
print(cp)


# %% Estimate discretization error at the last time step
# Cell counts taken from each case's log.blockMesh, ordered finest to coarsest
cell_counts = [328640, 82160, 20540] # bfs4, bfs2, bfs1
mesh_sizes = utils.mesh_size(domain=1, count=cell_counts, dim=2)

last_time = 2000
cp_final = cp.loc[last_time, ["bfs4", "bfs2", "bfs1"]].to_numpy()

model = Classic(mesh_sizes, {"Cp": cp_final})
model.summarize()
model.plot()
