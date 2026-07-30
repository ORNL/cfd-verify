OpenFOAM
========

OpenFOAM is an open-source finite-volume solver that is widely used for CFD.
This tutorial will walk you through conducting a discretization error evaluation using OpenFOAM.
For this case, we will recreate the backwards-facing step (BFS) case published by Celik and Karatekin in 1997 [Celik1997]_.
*Note*, you will access to an OpenFOAM installation for this tutorial. 
If you do not have access and are not familiar with installing research software, it is recommended you start with the mfoil tutorial instead.

The BFS is a canonical case in CFD that has been extensively used for turbulence modeling development.
The step simulates a 2D channel flow (i.e., infinite in the spanwise direction) where the channel suddenly expands.
This expansion causes flow separation at the step, and with sufficient velocity, extensive turbulent mixing behind it.
The flow separation forms a clockwise vortex behind the step where the velocity near the wall is opposite of the primary streamwise flow.
The chief quantity of interest in this flow is the reattachment length (i.e., the distance behind the step where the average velocity just off the wall returns to streamwise).
The reattachment length can be measured by identify the location on the wall where the static pressure reaches a local maximum.

Celik and Karatekin showed...

In this tutorial, we will investigate the discretization error of the reattachment length, the mean post-step wall pressure, and local velocity components in the free shear layer.
Stuff on why ...


Steps
^^^^^

#. Create a directory on your analysis machine called `bfs` where we will run the OpenFOAM cases.
#. Copy the tutorial case to a subdirectory called `bfs1`. The tutorial is located at `[path-to-OpenFOAM-installation]/tutorials/incompressible/simpleFoam/backwardFacingStep2D`.
#. Copy the `bfs1` subdirectory to another subdirectory called `bfs2`.
#. Copy the `bfs2` subdirectory to another subdirectory called `bfs4`.



Next Steps
^^^^^^^^^^

The above tutorial explains the basic process of evaluating discretization error using Richardson extrapolation

References
==========

.. [Celik1997] I. Ceilk and O. Karatekin, *Grid Convergence Studies on Nonuniform Grids*, 1997.