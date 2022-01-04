#!/usr/bin/env python3

# import sys
# sys.path.insert(1, 
#       '/Users/jeanettemumford/Dropbox/Research/Projects/RT_sims/Code')
#import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from functions import calc_win_sub_pow_range
from scipy.stats import gamma, exponnorm

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
hp_filter = False
ISI_min_max_vec = [(1, 3), (2, 5), (3, 6), (4, 7)]

mu_expnorm = mu_grinband_shift
lam_expnorm = 1 / inv_lambda_grinband_shift
sigma_expnorm = sigma_grinband_shift

hp_filter = True

output_unmod_beta_power_grin_hp_yes, output_rtmod_beta_power_grin_hp_yes = \
       calc_win_sub_pow_range(n_trials, scan_length, repetition_time,
                              mu_expnorm, lam_expnorm, sigma_expnorm,
                              max_rt, min_rt, event_duration, ISI_min_max_vec,
                              win_sub_noise_sd_range_scales_yes,
                              win_sub_noise_sd_range_scales_no, center_rt,
                              beta_scales_yes, beta_scales_no, hp_filter,
                              nsim_pow)

print('Grinband Done')

# Adjust noise for stroop settings
mu_expnorm = 525
lam_expnorm = 1 / 166
sigma_expnorm = 77
win_sub_noise_sd_range_scales_yes = [.5, .75, 1.25, 2.75, 7]
win_sub_noise_sd_range_scales_no = [.25, 0.35, 0.5, 0.8, 3]

output_unmod_beta_power_stroop_hp_yes, output_rtmod_beta_power_stroop_hp_yes =\
       calc_win_sub_pow_range(n_trials, scan_length, repetition_time,
                              mu_expnorm, lam_expnorm, sigma_expnorm,
                              max_rt, min_rt, event_duration, ISI_min_max_vec,
                              win_sub_noise_sd_range_scales_yes,
                              win_sub_noise_sd_range_scales_no, center_rt,
                              beta_scales_yes, beta_scales_no, hp_filter,
                              nsim_pow)

output_unmod_beta_power_grin = output_unmod_beta_power_grin_hp_yes
output_rtmod_beta_power_grin = output_rtmod_beta_power_grin_hp_yes
output_unmod_beta_power_stroop = output_unmod_beta_power_stroop_hp_yes
output_rtmod_beta_power_stroop = output_rtmod_beta_power_stroop_hp_yes

isi_labels = list(output_unmod_beta_power_grin.keys())
isi_labels = isi_labels[0:3]
nrows_plot = 4
ncols_plot = len(isi_labels)
fig, axs = plt.subplots(nrows_plot+1, ncols_plot, sharex=True, sharey=True,
                        figsize=(9.5, 9),
                        gridspec_kw={"height_ratios": [0.02, 1, 1, 1, 1]})
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                left=False, right=False)
plt.xlabel("Effect size (correlation)", fontsize=20)
plt.ylabel("Power (within-subject)", fontsize=20)
plt.setp(axs, xlim=(0.05, .3))
out_types = [('grin', 'scales_yes', 'Forced Choice\n Scales with RT'),
             ('grin', 'scales_no', "Forced Choice\n Doesn't scale with RT"),
             ('stroop', 'scales_yes', 'Stroop\n Scales with RT'),
             ('stroop', 'scales_no', "Stroop\n Doesn't scale with RT")]
for row in range(len(out_types)):
    for col in range(len(isi_labels)):
        rt_settings = out_types[row][0]
        scale_settings = out_types[row][1]
        plot_label = out_types[row][2]
        isi_settings = isi_labels[col]
        out_name_rtmod = f'output_rtmod_beta_power_{rt_settings}'
        out_name_unmod = f'output_unmod_beta_power_{rt_settings}'
        rtmod_data = vars()[out_name_rtmod]
        unmod_data = vars()[out_name_unmod]
        sim_type = f'dv_{scale_settings}'
        if scale_settings == 'scales_yes':
            correlation =\
              rtmod_data[isi_settings]['cor_est_scales_yes_filt_yes']
        else:
            correlation = \
              unmod_data[isi_settings]['cor_est_scales_no_filt_yes']
        line1, = axs[row+1, col].plot(
                     correlation,
                     rtmod_data[isi_settings][sim_type]['RT Duration only'],
                     'tab:blue', label='RT duration')
        line2, = axs[row+1, col].plot(
                    correlation,
                    unmod_data[isi_settings][sim_type]['Impulse Duration'],
                    'tab:green', label='Const (Impulse*)')
        line3,  = axs[row+1, col].plot(
                    correlation,
                    unmod_data[isi_settings][sim_type]['Stimulus Duration'],
                    color='tab:purple', label='Const (Stimulus Duration*)')
        line4,  = axs[row+1, col].plot(
                    correlation,
                    unmod_data[isi_settings][sim_type]['Mean RT Duration'],
                    color='tab:red', label='Const (Mean RT Duration*)')
        line5,  = axs[row+1, col].plot(
                    correlation,
                    unmod_data[isi_settings][sim_type]['No RT effect'],
                    color='tab:olive', label='Const (Stimulus Duration)')
        axs[row+1, col].set_title(plot_label)
for i, ax in enumerate(axs.flatten()[:3]):
    ax.axis("off")
    ax.set_title(f'ISI=U{isi_labels[i]}', fontweight='bold', fontsize=15)
fig.subplots_adjust(hspace=.5, bottom=0.1)
fig.tight_layout()
plt.legend(handles=[line1, line2, line3, line4, line5],
           loc='center right', bbox_to_anchor=(1.32, .5),
           ncol=1)
figpath = "/Users/jeanettemumford/sherlock_home/RT_sims/Output/single_sub_power.pdf"
plt.savefig(figpath, format='pdf', transparent=True, pad_inches=.1,
            bbox_inches='tight')

