#!/usr/bin/env python3

from functions import group_power_range_btwn_sd, power_plot_group, est_win_sub_mod_sd, calc_group_eff_size
import numpy as np
import sys


figure_directory = '/home/users/jmumford/RT_sims/Output'
display_plots = False

textout = '/home/users/jmumford/RT_sims/Output/text_out.txt'
sys.stdout = open(textout, "w")

nsub = 30
n_trials = 30
scan_length = 225
repetition_time = 1
mu_grinband_shift = 638
inv_lambda_grinband_shift = 699
sigma_grinband_shift = 103
mu_expnorm = mu_grinband_shift
lam_expnorm = 1 / inv_lambda_grinband_shift
sigma_expnorm = sigma_grinband_shift
max_rt = 8000
min_rt = 50
event_duration = .5  
center_rt=True
hp_filter = True
ISI_min = 3
ISI_max = 6
nsim_sd_est = 100
nsim_power = 500


beta = {'dv_scales_yes': 3, 'dv_scales_no': 15}
win_sub_noise_sd = {'dv_scales_yes': 3, 'dv_scales_no': .5}

win_sub_mod_sd = est_win_sub_mod_sd(n_trials, scan_length, 
              repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, center_rt,
              hp_filter, nsim_sd_est)

btwn_sub_noise_sd_vec = {'dv_scales_yes': np.array([0, 3, 6, 7, 15, 20]),
                         'dv_scales_no': np.array([0, 20, 40, 50, 80, 110])}
all_sd_params = {'win_sub_noise_sd': win_sub_noise_sd,
                 'win_sub_mod_sd': win_sub_mod_sd,
                 'btwn_noise_sd_vec': btwn_sub_noise_sd_vec}

eff_sizes = calc_group_eff_size(all_sd_params, beta)

print(eff_sizes)
sys.stdout.flush()

output_forced_choice = group_power_range_btwn_sd(n_trials, scan_length, repetition_time, 
              mu_expnorm, lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, all_sd_params, 
              nsub, nsim_power, center_rt, beta, hp_filter, figure_directory, display_plots)

print('Power estimated, starting plot')
sys.stdout.flush()

power_plot_group(output_forced_choice, figure_directory, 
                     zoom=False, show_rt_mod=False, display_plots=False)

power_plot_group(output_forced_choice, figure_directory, 
                     zoom=False, show_rt_mod=True, display_plots=False)

print('Starting Stroop')
sys.stdout.flush()
#Stroop settings
mu_expnorm = 530
lam_expnorm = 1 / 160
sigma_expnorm = 77

beta = {'dv_scales_yes': 1.7, 'dv_scales_no': 8}
win_sub_noise_sd = {'dv_scales_yes': 1.25, 'dv_scales_no': .5}
win_sub_mod_sd = est_win_sub_mod_sd(n_trials, scan_length, 
              repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, center_rt,
              hp_filter, nsim_sd_est)
btwn_sub_noise_sd_vec = {'dv_scales_yes': np.array([3.5]),
                         'dv_scales_no': np.array([16])}
all_sd_params = {'win_sub_noise_sd': win_sub_noise_sd,
                 'win_sub_mod_sd': win_sub_mod_sd,
                 'btwn_noise_sd_vec': btwn_sub_noise_sd_vec}


eff_sizes = calc_group_eff_size(all_sd_params, beta)
print(eff_sizes)

nsub = 30
print(f'Results with {nsub} subjects')
output_stroop = group_power_range_btwn_sd(n_trials, scan_length, repetition_time, 
              mu_expnorm, lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, all_sd_params, 
              nsub, nsim_power, center_rt, beta, hp_filter, figure_directory, display_plots)

nsub = 50
print(f'Results with {nsub} subjects')
output_stroop = group_power_range_btwn_sd(n_trials, scan_length, repetition_time, 
              mu_expnorm, lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, all_sd_params, 
              nsub, nsim_power, center_rt, beta, hp_filter, figure_directory, display_plots)

sys.stdout.flush()
