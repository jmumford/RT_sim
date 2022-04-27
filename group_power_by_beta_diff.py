#!/usr/bin/env python3

root_dir = '/home/users/jmumford/RT_sims/'

import sys
import pandas as pd
import numpy as np
sys.path.insert(1, root_dir + '/Code')
from simulation_settings import *
from functions import group_2stim_beta2_vec
import json

nsim = 2500
rt_diff_s = 0
beta_scales_yes={'beta1': .75, 'beta2': [.75, .8, .95, 1,  1.05, 1.1]}
beta_scales_no={'beta1': .85, 'beta2': [.85, .9, 1.1, 1.15, 1.2, 1.25]}

out_results_name = root_dir + f'/Output/group_power_output_nsim{nsim}_nsub{nsub}_mu{mu_expnorm}_btwn_noise{btwn_sub_noise_sd["dv_scales_yes"]}{btwn_sub_noise_sd["dv_scales_no"]}.csv'
out_settings_name = root_dir + f'/Output/group_power_output_nsim{nsim}_nsub{nsub}_mu{mu_expnorm}_settings_btwn_noise{btwn_sub_noise_sd["dv_scales_yes"]}{btwn_sub_noise_sd["dv_scales_no"]}.json'

beta_diff_vec = np.array(beta_scales_yes['beta2']) - np.array(beta_scales_yes['beta1'])
output_power = group_2stim_beta2_vec(n_trials, scan_length, repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, 
              win_sub_noise_sd,
              btwn_sub_noise_sd,
              center_rt, beta_scales_yes,
              beta_scales_no, 
              rt_diff_s, nsub, nsim)


data_type = ['blocked', 'random']
models = ['Two stimulus types, no RT', 'Two stimulus types, RT mod',\
          'Two stimulus types, 2 RT dur only']
scale_type = ['dv_scales_yes', 'dv_scales_no']
beta_con_type = ['beta_diff_est', 'beta2_est']

data_type_long = []
models_long = []
scale_type_long = []
beta_con_type_long = []
rej_rate_beta_con = []
beta_diff_long = []

for cur_data_type in data_type:
    for cur_model in models:
        for cur_scale_type in scale_type:
            for cur_beta_con_type in beta_con_type:
                rej_rate_loop = \
                    output_power['group_rej_rate'][cur_data_type][cur_model]\
                                            [cur_scale_type][cur_beta_con_type]
                nvals = len(rej_rate_loop)
                data_type_long.extend([cur_data_type] * nvals)
                models_long.extend([cur_model] * nvals)
                scale_type_long.extend([cur_scale_type] * nvals)
                beta_con_type_long.extend([cur_beta_con_type] * nvals) 
                rej_rate_beta_con.extend(rej_rate_loop)
                beta_diff_long.extend(beta_diff_vec)


data_long = pd.DataFrame(list(zip(beta_diff_long, data_type_long, models_long, \
                    scale_type_long, beta_con_type_long, rej_rate_beta_con)),
                    columns = ['Beta diff','Data Type', 'Model', 'Scale Type', \
                   'Beta Contrast', 'Rejection Rate'])


data_long.to_csv(out_results_name)


all_settings = {'nsub': nsub,
             'mu_expnorm':mu_expnorm,
             'lam_expnorm': lam_expnorm,
             'sigma_expnorm': sigma_expnorm,
             'max_rt': max_rt,
             'min_rt': min_rt,
             'event_duration': event_duration,
             'center_rt': center_rt,
             'hp_filter': hp_filter,
             'ISI_min': ISI_min,
             'ISI_max': ISI_max,
             'win_sub_noise_sd': win_sub_noise_sd,
             'btwn_sub_noise_sd': btwn_sub_noise_sd,
             'beta_scales_yes_beta1': beta_scales_yes['beta1'],
             'beta_scales_yes_beta2': list(beta_scales_yes['beta2']),
             'beta_scale_no_beta1': beta_scales_no['beta1'],
             'beta_scale_no_beta2': list(beta_scales_no['beta2']),
             'nsim': nsim}
with open(out_settings_name, "w") as outfile:
    json.dump(all_settings, outfile, indent=4)

