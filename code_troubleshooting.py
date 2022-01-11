#!/usr/bin/env python3

import sys
sys.path.insert(1, 
       '/Users/jeanettemumford/Dropbox/Research/Projects/RT_sims/Code')
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from functions import *
from scipy.stats import gamma, exponnorm
import time 
from nilearn.glm.first_level.design_matrix import _cosine_drift

file_path = './outputfile.txt'
sys.stdout = open(file_path, "w")

print('getting started')
sys.stdout.flush()
rt_grinband_shift = gamma.rvs(a=1.7, loc=.5, scale=.49, size=10000)
shape_grinband_shift, mu_grinband_shift, sigma_grinband_shift =\
     exponnorm.fit(rt_grinband_shift*1000)
inv_lambda_grinband_shift = np.multiply(sigma_grinband_shift,
                                        shape_grinband_shift)
n_trials = 30
scan_length = 225
repetition_time = 1

mu_expnorm = mu_grinband_shift
lam_expnorm = 1 / inv_lambda_grinband_shift
sigma_expnorm = sigma_grinband_shift

max_rt = 8000
min_rt = 50
event_duration = .5
beta_scales_yes = 5
beta_scales_no = 30
nsim = 100
center_rt = True

win_sub_noise_sd_range_scales_yes = [1.2, 1.75,  3, 4, 10]
win_sub_noise_sd_range_scales_no = [.25,  .35, .5,  .75, 3]

nsim_pow = 5000
ISI_min_max_vec = [(1, 3), (2, 5), (3, 6), (4, 7)]

mu_expnorm = mu_grinband_shift
lam_expnorm = 1 / inv_lambda_grinband_shift
sigma_expnorm = sigma_grinband_shift

hp_filter = True




nsim_check = 200000
ISI_min = 3
ISI_max = 6
sd_ind = 1
start_time = time.time()
for i in range(nsim_check):
    if i%1000 == 0:
        print(i)
        print(start_time - time.time())
        start_time = time.time()
        sys.stdout.flush()
    regressors, _ = make_regressors_one_trial_type(n_trials, scan_length, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, 
                                   ISI_max, center_rt)