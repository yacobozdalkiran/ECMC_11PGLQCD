# Event-Chain Monte Carlo on 1+1d Pure Gauge Lattice QCD

Python implementation of Event-Chain Monte Carlo for pure gauge 1+1 dimensions lattice QCD.

The implementation consists of 4 python files located in src/ : 

* **gauge_su3.py** which contains SU(3)/gauge configurations functions
* **analytical_reject.py** which contains all the routines necessary to generate rejects following analytical results (only possible for $\lambda_2$, $\lambda_3$ and $\lambda_5$)
* **numerical_reject.py** which contains all the routines necessary to generate rejects following numerical procedure, only for $\lambda_8$
* **ECMC.py** which is the principal module that uses all the others, implementing the Event-Chain algorithm

A documentation on every function constituting those four modules is available here : https://yacobozdalkiran.github.io/ECMC_11PGLQCD/

To use the modules, you have to install numpy and scipy :
```bash
pip install numpy
pip install scipy
```
Then you can clone the repo and create a python script test.py in the src folder and use the functions :

```python
from ECMC import *

#Test of an ECMC step + computation of actions before/after :
c = init_conf(30,30,cold=True)
beta = 2.5
angle_l = 120
param_lambda_l = 15
param_pos_l = 17
print("Action before ECMC step : " +str(calculate_action(c, beta)))
ECMC_step_l(c, 1, 1, 0, beta, angle_l, param_lambda_l, param_pos_l)
print("Action after ECMC step : " +str(calculate_action(c, beta)))


#Test of sample generation :
sample = ECMC_samples(L=3, T=3, cold = True, beta=1.55, angle_l=120, param_lambda_l=15, param_pos_l=30, n=3)

```
