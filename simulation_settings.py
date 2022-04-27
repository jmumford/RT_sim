
import numpy as np

nsub = 100
n_trials = 80
repetition_time = 1
#Stroop settings
#mu_expnorm = 530
#lam_expnorm = 1 / 160
#sigma_expnorm = 77
mu_grinband_shift = 638
inv_lambda_grinband_shift = 699
sigma_grinband_shift = 103
mu_expnorm = mu_grinband_shift
lam_expnorm = 1 / inv_lambda_grinband_shift
sigma_expnorm = sigma_grinband_shift
max_rt = 8000
min_rt = 50
event_duration = .1  
center_rt=False
hp_filter = True
ISI_min = 3
ISI_max = 6

win_sub_noise_sd={'dv_scales_yes': 1.15, 'dv_scales_no': .09}
btwn_sub_noise_sd={'dv_scales_yes': .65, 'dv_scales_no': .75}

