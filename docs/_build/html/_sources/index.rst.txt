.. ECMC for pure gauge LQCD documentation master file, created by
   sphinx-quickstart on Tue Mar  4 10:38:21 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ECMC for pure gauge LQCD documentation
======================================

Implementation of the Event-Chain algorithm for SU(3) pure gauge theory on the lattice.
Uses the Wilson action and 1+1 dimensional lattice.

The code is separated in 4 modules :
 - gauge_su3.py which contains SU(3)/gauge configurations functions
 - analytical_reject.py which contains all the routines necessary to generate rejects following analytical results (only possible for lambda_2,lambda_3 and lambda_5)
 - numerical_reject.py which contains all the routines necessary to generate rejects following numerical procedure, only for lambda_8
 - ECMC.py which is the principal module that uses all the others, implementing the Event-Chain algorithm

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules