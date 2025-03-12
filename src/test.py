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
