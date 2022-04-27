#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
root_dir = '/home/users/jmumford/RT_sims/'
sys.path.insert(1, root_dir + '/Code')
from simulation_settings import *
from functions import group_2stim_rt_diff_vec
import json

beta_scales_yes = np.array([.75, .75])
beta_scales_no = np.array([.85, .85])
nsim = 2500

out_results_name = root_dir + f'/Output/group_type1_err_corr_by_rtdiff_output_nsim{nsim}_nsub{nsub}_mu{mu_expnorm}_btwn_noise{btwn_sub_noise_sd["dv_scales_yes"]}{btwn_sub_noise_sd["dv_scales_no"]}.csv'
out_settings_name = root_dir + f'/Output/group_type1_err_corr_by_rtdiff_output_nsim{nsim}_nsub{nsub}_mu{mu_expnorm}_btwn_noise{btwn_sub_noise_sd["dv_scales_yes"]}{btwn_sub_noise_sd["dv_scales_no"]}.json'

rt_diff_s_vec = [0, 0.05, 0.1, .2, .3]

output = group_2stim_rt_diff_vec(n_trials, scan_length, repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, 
              win_sub_noise_sd,
              btwn_sub_noise_sd, 
              center_rt, beta_scales_yes, beta_scales_no,  
              rt_diff_s_vec, nsub, nsim)


data_type = ['blocked', 'random']
models = ['Two stimulus types, no RT', 'Two stimulus types, RT mod',\
          'Two stimulus types, 2 RT dur only']
scale_type = ['dv_scales_yes', 'dv_scales_no']


data_type_long = []
models_long = []
scale_type_long = []
rej_rate_beta_diff = []
group_cor_betadiff_rtdiff = []
group_cor_betadiff_rtmn = []
rt_diff_long = []

for cur_data_type in data_type:
    for cur_model in models:
        for cur_scale_type in scale_type:
            rej_rate_loop = output['group_rej_rate'][cur_data_type][cur_model]\
                                            [cur_scale_type]['beta_diff_est']
            nvals = len(rej_rate_loop)
            data_type_long.extend([cur_data_type] * nvals)
            models_long.extend([cur_model] * nvals)
            scale_type_long.extend([cur_scale_type] * nvals)
            rej_rate_beta_diff.extend(rej_rate_loop)
            group_cor_betadiff_rtdiff.extend(output['group_rtdiff_cor_avg'][cur_data_type][cur_model]\
                                            [cur_scale_type]['beta_diff_est'])
            group_cor_betadiff_rtmn.extend(output['group_rtmn_cor_avg'][cur_data_type][cur_model]\
                                            [cur_scale_type]['beta_diff_est'])
            rt_diff_long.extend(output['rt_diff'] * nvals)


data_long = pd.DataFrame(list(zip(rt_diff_long, data_type_long, models_long, \
                            scale_type_long,rej_rate_beta_diff,
                            group_cor_betadiff_rtdiff, group_cor_betadiff_rtmn)),
            columns = ['RT diff','Data Type', 'Model', 'Scale Type', \
                 'Rejection Rate', 'Correlation (beta diff with rt diff)',
                 'Correlation (beta diff with rt mn)'])
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
             'beta_scales_yes': list(beta_scales_yes),
             'beta_scale_no': list(beta_scales_no),
             'nsim': nsim}
with open(out_settings_name, "w") as outfile:
    json.dump(all_settings, outfile, indent=4)

