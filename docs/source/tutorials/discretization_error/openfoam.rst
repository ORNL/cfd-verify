OpenFOAM
========

OpenFOAM is an open-source finite-volume solver which is a popular choice for CFD solvers.
This tutorial will walk you through conducting a discretization error evaluation using OpenFOAM.
For this case, we will use the 2D incompressible backwards-facing step case in the OpenFOAM tutorials.
You will need OpenFOAM installed to run the cases for this tutorial.

Steps
^^^^^

#. Create a directory on your analysis machine called `bfs` where we will run the OpenFOAM cases.
#. Copy the tutorial case to a subdirectory called `bfs1`. The tutorial is located at `[path-to-OpenFOAM-installation]/tutorials/incompressible/simpleFoam/backwardFacingStep2D`.
#. Copy the `bfs1` subdirectory to another subdirectory called `bfs2`.
#. Copy the `bfs2` subdirectory to another subdirectory called `bfs4`.



Next Steps
^^^^^^^^^^

The above tutorial explains the basic process of evaluating discretization error using Richardson extrapolation