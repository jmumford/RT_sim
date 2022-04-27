from scipy.stats import exponnorm, gamma
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#test that I understand how it works.

K=10
loc=154
scale=75
#shape_expnorm = 1 / (sigma_expnorm * lam_expnorm)
#subject_specific_mu_expnorm = exponnorm.rvs(shape_expnorm, mu_expnorm,
#                                                sigma_expnorm, 1) - \
#                                                1 / lam_expnorm

rand_data = exponnorm.rvs(K, loc, scale, 1000)

k_est, loc_est, scale_est  = exponnorm.fit(rand_data)
print(k_est)
print(loc_est)
print(scale_est)
# Okay, that seems about right, now onto the real data

basedir = Path('/Users/jeanettemumford/Dropbox/Research/Projects/RTGroup/Data/')
tseries_paths = [i for i in basedir.glob('TimeSeries_3mm_ROI/sub-*_roi_-1.92_-50.02_48.11_dat_fit_avg.csv')]

shape_est = list()
mu_est = list()
sigma_est = list()

shape_est_cong = list()
mu_est_cong = list()
sigma_est_cong = list()

shape_est_incong = list()
mu_est_incong = list()
sigma_est_incong = list()

mn_rt = list()

mn_rt_cong = np.full(len(tseries_paths), np.nan)
mn_rt_incong = np.full(len(tseries_paths), np.nan)

std_rt_cong = np.full(len(tseries_paths), np.nan)
std_rt_incong = np.full(len(tseries_paths), np.nan)

for idx, tseries_file in enumerate(tseries_paths):
    subid = tseries_file.parts[-1][4:7]
    event_file = basedir.glob(f"Stroop_event_files/sub-s{subid}_ses-*_task-stroop_run-1_events.tsv")
    event_file = [i for i in event_file][0]
    event_data = pd.read_csv(event_file, sep='\t')
    rt_means_by_category = event_data.groupby('trial_type')['response_time'].mean()
    rt_sd_by_category = event_data.groupby('trial_type')['response_time'].std()
    mn_rt_cong[idx] = rt_means_by_category['congruent']*1000
    mn_rt_incong[idx] = rt_means_by_category['incongruent']*1000
    std_rt_cong[idx] = rt_sd_by_category['congruent']*1000
    std_rt_incong[idx] = rt_sd_by_category['incongruent']*1000
    reaction_times = event_data.loc[event_data.junk == False, 'response_time']*1000
    reaction_times_cong = event_data.loc[((event_data.junk == False) & (event_data.trial_type == 'congruent')), 'response_time']*1000
    reaction_times_incong = event_data.loc[((event_data.junk == False) & (event_data.trial_type == 'incongruent')), 'response_time']*1000
    shape_loop, mu_loop, sigma_loop  = exponnorm.fit(reaction_times)
    shape_est.append(shape_loop)
    mu_est.append(mu_loop)
    sigma_est.append(sigma_loop)
    shape_loop_cong, mu_loop_cong, sigma_loop_cong  = exponnorm.fit(reaction_times_cong)
    shape_est_cong.append(shape_loop_cong)
    mu_est_cong.append(mu_loop_cong)
    sigma_est_cong.append(sigma_loop_cong)
    shape_loop_incong, mu_loop_incong, sigma_loop_incong  = exponnorm.fit(reaction_times_incong)
    shape_est_incong.append(shape_loop_incong)
    mu_est_incong.append(mu_loop_incong)
    sigma_est_incong.append(sigma_loop_incong)
    mn_rt.append(reaction_times.mean())


print([np.mean(shape_est),  1/np.mean(np.multiply(sigma_est, shape_est)), np.mean(mu_est), np.mean(sigma_est)])
print([np.mean(shape_est_cong),  1/np.mean(np.multiply(sigma_est_cong, shape_est_cong)), np.mean(mu_est_cong), np.mean(sigma_est_cong)])
print([np.mean(shape_est_incong),  1/np.mean(np.multiply(sigma_est_incong, shape_est_incong)), np.mean(mu_est_incong), np.mean(sigma_est_incong)])

print(np.mean(mn_rt_cong))
print(np.mean(std_rt_cong))
print(np.mean(mn_rt_incong))
print(np.mean(std_rt_incong))


plt.plot(mn_rt_cong, mn_rt_incong, 'b.')
plt.show()

plt.plot(mn_rt_incong - mn_rt_cong, 'b.')
plt.show()

print(np.std(mn_rt_cong))
print(np.std(mn_rt_incong))


mu_est = np.array(mu_est)
sigma_est = np.array(sigma_est)
shape_est = np.array(shape_est)
mn_rt = np.array(mn_rt)
print(mu_est[mn_rt>1000])
print(shape_est[mn_rt>1000])
print(sigma_est[mn_rt>1000])

from scipy.stats import norm
params_norm = norm.fit(mu_est)

rand_samp = np.random.normal(525, 90, 1000)

fig, axs = plt.subplots(3)
axs[0].plot(mn_rt, mu_est, '.')
axs[1].plot(mn_rt, shape_est, '.')
axs[2].plot(mn_rt, sigma_est, '.')
plt.show()

#Notes about expnorm:
#    theoretical mean = mu_expnorm + 1/lam_expnorm
#    theoretical variance = sigma_expnorm**2 + 1/lam_expnorm**2

np.mean(mn_rt_incong - mn_rt_cong)

th_mn = mu_est + np.multiply(shape_est, sigma_est)

#shape_expnorm = 1 / (sigma_expnorm * lam_expnorm)
inv_lambda_est =  np.multiply(sigma_est, shape_est)
th_sd = np.sqrt(np.array(sigma_est)**2 + np.array(inv_lambda_est)**2)
print(np.mean(th_mn))
print(np.mean(th_sd))

print(np.mean(shape_est))
print(np.mean(inv_lambda_est))
print(np.mean(mu_est))
print(np.mean(sigma_est))

print(np.mean(mn_rt))


shape_grp_cong, mu_grp_cong, sigma_grp_cong  = exponnorm.fit(np.array(mn_rt_cong)/1000)
inv_lambda_grp_cong = np.multiply(sigma_grp_cong, shape_grp_cong)

shape_grp_incong, mu_grp_incong, sigma_grp_incong  = exponnorm.fit(np.array(mn_rt_incong)/1000)
inv_lambda_grp_incong = np.multiply(sigma_grp_incong, shape_grp_incong)

shape_grp, mu_grp, sigma_grp  = exponnorm.fit(np.array(mn_rt)/1000)
inv_lambda_grp = np.multiply(sigma_grp, shape_grp)

print([shape_grp, inv_lambda_grp, mu_grp, sigma_grp])
print([shape_grp_cong, inv_lambda_grp_cong, mu_grp_cong, sigma_grp_cong])
print([shape_grp_incong, inv_lambda_grp_incong, mu_grp_incong, sigma_grp_incong])

np.mean(mn_rt_incong - mn_rt_cong)/1000

x_dist = np.linspace(0, 2, 100)
expnorm_fit_all = exponnorm.pdf(x=x_dist, K=shape_grp, loc=mu_grp, scale=sigma_grp)
expnorm_fit_cong = exponnorm.pdf(x=x_dist, K=shape_grp_cong, loc=mu_grp_cong, scale=sigma_grp_cong)
expnorm_fit_incong = exponnorm.pdf(x=x_dist, K=shape_grp_incong, loc=mu_grp_incong, scale=sigma_grp_incong)
test_incong = exponnorm.pdf(x=x_dist, K=shape_grp_cong, loc=mu_grp_cong+.09, scale=sigma_grp_cong)

plt.plot(x_dist, expnorm_fit_all, label='all')
plt.plot(x_dist, expnorm_fit_cong, label='cong')
plt.plot(x_dist, expnorm_fit_incong, label='incong')
plt.plot(x_dist, test_incong, label = 'test_incong')
plt.legend()
plt.show()




shape_grp, mu_grp, sigma_grp  = exponnorm.fit(np.array(mn_rt)/1000)
inv_lambda_grp = np.multiply(sigma_grp, shape_grp)
print([shape_grp, inv_lambda_grp, mu_grp, sigma_grp])

g_shape, g_loc, g_scale = gamma.fit(np.array(mn_rt)/1000)
# shape is alpha, scale = 1/beta (beta often called rate)
print([g_shape, g_loc, 1/g_scale])

#compare original data to the two distribution fits
mn_rt_s = np.array(mn_rt)/1000

x_dist = np.linspace(0, 2, 100)
gam_fit = gamma.pdf(x=x_dist,loc=g_loc, a=g_shape, scale=g_scale)
expnorm_fit = exponnorm.pdf(x=x_dist, K=shape_grp, loc=mu_grp, scale=sigma_grp)

fig, ax = plt.subplots(1, 1)
ax.hist(mn_rt_s, 50,  density=True, alpha = .5)
ax.plot(x_dist, gam_fit)
ax.plot(x_dist, expnorm_fit, 'r')
plt.show()

# Trying to recreate parameters from Grinband, but for exponnorm
# I cannot match their mean/sd reported, but this seems to look similar
# to their histogram in the supplement
fake_data = gamma.rvs(a = 1.7, loc = 0.5, scale = .4, size = 10000)
#fake_data=fake_data[fake_data>.45]
#goal: mean=.84, sd=.64
print([fake_data.mean(), fake_data.std()])
plt.hist(fake_data, 100)
plt.show()

shape_grin, mu_grin, sigma_grin  = exponnorm.fit(fake_data*1000)
inv_lambda_grin = np.multiply(sigma_grin, shape_grin)

shape_grp, mu_grp, sigma_grp  = exponnorm.fit(np.array(mn_rt))
inv_lambda_grp = np.multiply(sigma_grp, shape_grp)

print([shape_grin, inv_lambda_grin, mu_grin, sigma_grin])
print([shape_grp, inv_lambda_grp, mu_grp, sigma_grp])