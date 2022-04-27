
import numpy as np

nsub = 100
n_trials = 80
repetition_time = 1
#Stroop settings
mu_expnorm = 530
lam_expnorm = 1 / 160
sigma_expnorm = 77
max_rt = 8000
min_rt = 50
event_duration = .1  
center_rt=False
hp_filter = True


win_sub_noise_sd={'dv_scales_yes': .8, 'dv_scales_no': .09}
btwn_sub_noise_sd={'dv_scales_yes': .65, 'dv_scales_no': .6}
beta_scales_yes_type1err = np.array([1.05, 1.05])
beta_scales_no_type1err = np.array([.85, .85])

beta_scales_yes_power={'beta1': 1.05, 'beta2': [1.05, 1.1, 1.15, 1.2,  1.25, 1.3, 1.35, 1.4]}
beta_scales_no_power={'beta1': .85, 'beta2': [.85, .9, .95, 1, 1.05, 1.1, 1.15, 1.2]}

rt_diff_s_vec = [0, .025, 0.05, 0.1, .15]
