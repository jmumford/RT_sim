from scipy.stats import exponnorm, t
import scipy.stats
import numpy as np
from nilearn.glm.first_level import hemodynamic_models
from nilearn.glm.first_level.design_matrix import _cosine_drift
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
import seaborn as sns
import pandas as pd
from collections import defaultdict

def make_3column_onsets(onsets, durations, amplitudes):
    """Utility function to generate 3 column onset structure
    """
    return(np.transpose(np.c_[onsets, durations, amplitudes]))


def make_regressors_one_trial_type(n_trials, scan_length,
                                   repetition_time=1, mu_expnorm=600,
                                   lam_expnorm=1 / 100, sigma_expnorm=75,
                                   max_rt=2000, min_rt=0, event_duration=2, 
                                   ISI_min=2, ISI_max = 5, center_rt=True):
    """Generate regressors for one trial type samping response times from
    ex-Gaussian distribution.  7 regressors, total, are generated based on
    different popular modeling options used for trials that vary in duration by
    RT across trials

    Args:
        mu_expnorm (Union[int, float])      : Mean exgaussian RT 
                                              parameter mu(ms)
        n_trials (int)                      : number of trials
        scan_length (Union[int, float])     : length of scan in seconds
        repetition_time (Union[int, float]) : repetition time for scan
                                             (in seconds). Defaults to 1
        lam_expnorm (float, optional)       : rate parameter for exponential.
                                              Defaults to 1/100.
        sigma_expnorm (int, optional)       : variance of Gaussian RT.
                                              Defaults to 75.
        max_rt (int, optional)              : Maximum RT value in msec.
                                              Defaults to 2000.
        min_rt (int, optional)              : Minimum RT value in msec.
                                              Defaults to 0.
        event_duration (int, optional)      : Duration of events in seconds.
                                              Defaults to 2 secs.
        ISI_min (Union[int, float], optional): Minimum of interstimulus
                                              interval (s). Defaults to 2 secs.
        ISI_max (Union[int, float], optional): Maximum of interstimulus
                                              interval (s). Defaults to 5 secs.
        center_rt                            : Whether or not modulated RT is 
                                               centered. Default is True.
    Returns:
        regressors: Nested dictionary containing regressors with specific
                    durations (first level) and modulations (second level)
                    Durations, fixed_zero: Delta function (0 duration)
                    Durations, fixed_event_duration: duration = max_rt value
                    Durations, rt: duration = reaction time for that trial
                    Modulations, modulated: modulation is mean centered
                                            reaction time
                    Modulations, unmodulated: modulation 1 for all trials
    Notes about expnorm:
    theoretical mean = mu_expnorm + 1/lam_expnorm
    theoretical variance = sigma_expnorm**2 + 1/lam_expnorm**2
    Defaults gleaned from fig 1 of "Analysis of response time data" (heathcote)
    mus: 600-700, sigma: 50-100, lam = 1/tau where tau = 100

    """
    # 2 stages to generate RT (subject mean RT -> trial RTs).
    # NOTES:
    # - this assumes sigma_subject and sigma_trial are equal, probably wrong
    # - because samples may sometimes fall outside of min/max, we generate
    # twice as many as we need, filter for min/max, and then take the number
    # we actually need   
    shape_expnorm = 1 / (sigma_expnorm * lam_expnorm)
    subject_specific_mu_expnorm = exponnorm.rvs(shape_expnorm, mu_expnorm,
                                                sigma_expnorm, 1) - \
                                                1 / lam_expnorm
    #subject_specific_mu_expnorm = mu_expnorm
    sim_num_trials = 0
    while sim_num_trials < n_trials:
        rt_trials_twice_what_needed = exponnorm.rvs(shape_expnorm,
                                                subject_specific_mu_expnorm,
                                                sigma_expnorm, n_trials * 2) 
        rt_trials_filtered = rt_trials_twice_what_needed[
            np.where((rt_trials_twice_what_needed < max_rt) &
                 (rt_trials_twice_what_needed > min_rt))]
        sim_num_trials = rt_trials_filtered.shape[0]
    rt_trials = rt_trials_filtered[:n_trials] / 1000
    rt_trial_mean_s = np.mean(rt_trials)
    ISI = np.random.uniform(low = ISI_min, high = ISI_max, size = n_trials - 1)
    onsets = np.cumsum(np.append([5], rt_trials[0:(n_trials-1)]+ISI))
    frame_times = np.arange(0, scan_length*repetition_time, repetition_time)
    if center_rt == True:
        amplitudes = {'modulated': (rt_trials - rt_trial_mean_s),
                      'unmodulated': np.ones(onsets.shape)}
    else:
        amplitudes = {'modulated': rt_trials,
                      'unmodulated': np.ones(onsets.shape)}

    durations = {'fixed_zero': np.zeros(onsets.shape),
                 'fixed_event_duration': np.zeros(onsets.shape) +
                 event_duration,
                 'fixed_mean_rt': np.zeros(onsets.shape) + 
                rt_trial_mean_s,
                 'rt': rt_trials,
                 'rt_orth': rt_trials}
    
    regressors = {key: {key2: {} for key2 in amplitudes} for key in durations}

    for duration_type, duration in durations.items():
        for modulation_type, amplitude in amplitudes.items():
            if duration_type == 'rt_orth' and modulation_type == 'unmodulated':
                onsets_3col = make_3column_onsets(onsets, duration, amplitude)
                non_orth_dur_rt, _ = hemodynamic_models.compute_regressor(
                      onsets_3col, 'spm', frame_times, oversampling=16)
                contrasts = np.array([[1]])
                dur_stim_unmod = regressors['fixed_event_duration']['unmodulated']
                _, _, _, predy, _, _ = runreg(non_orth_dur_rt, 
                                    dur_stim_unmod, 
                                    contrasts, hp_filter=False, 
                                    compute_stats=False)
                regressors[duration_type][modulation_type] = non_orth_dur_rt - predy 
                # Alternative way (that kills power)
                #onsets_3col = make_3column_onsets(onsets + event_duration, 
                #                                  duration - event_duration, amplitude)
                #
                #regressors[duration_type][modulation_type], _ = \
                #      hemodynamic_models.compute_regressor(
                #      onsets_3col, 'spm', frame_times, oversampling=16)
            else:
                onsets_3col = make_3column_onsets(onsets, duration, amplitude)
                regressors[duration_type][modulation_type], _ = \
                      hemodynamic_models.compute_regressor(
                      onsets_3col, 'spm', frame_times, oversampling=16)

    return regressors, rt_trial_mean_s


def runreg(y, x, contrasts, hp_filter=True, compute_stats=True):
    """ Regression estimation function that estimates parameter estimates
    and contrast estimates for each row of a contrast matrix, contrasts.  Can
    either output betas only or betas and t-stats/p-values.

    Args:
        y (column array, length T): Dependent variable in regression
        x (array, T x nregressors): design matrix
        contrasts (ncontrasts x nregressors): Contrast matrix containing
                                              ncontrasts contrasts to compute
                                              ncontrasts estimates and t-stats
        compute_stats (logical, default=TRUE): If FALSE only contrast estimate
                                               is provided.  If TRUE the
                                               t-stats/p-values are returned
    expects contrasts as a ncontrasts X nregressors matrix
    but if only one is passed it will just be a (nregressors, ) matrix
    so reshape into a 1 X nregressors matrix
    """
    if len(contrasts.shape) == 1:
        contrasts = contrasts.reshape(1, contrasts.shape[0])

    if hp_filter == True:
        n_time_points = len(y)
        dct_basis = _cosine_drift(.01, np.arange(n_time_points))
        dct_basis = np.delete(dct_basis, -1, axis = 1)
        x = np.concatenate((x, dct_basis), axis = 1)
        n_contrasts = contrasts.shape[0]
        n_basis = dct_basis.shape[1]
        contrasts = np.concatenate((contrasts, np.zeros((n_contrasts, n_basis))),
                                    axis = 1)

    if hp_filter == False:
        n_basis = 0
    
    inv_xt_x = np.linalg.inv(x.T.dot(x))
    beta_hat = inv_xt_x.dot(x.T).dot(y)
    con_est = contrasts.dot(beta_hat)

    pred_y = x.dot(beta_hat)
    r_squared = np.corrcoef(np.array(pred_y).T, np.array(y).T)[0,1]
    con_t = []
    con_p = []
    if compute_stats:
        residual = y - pred_y
        df = x.shape[0] - x.shape[1]
        sigma2 = sum(residual**2) / df
        cov_beta_hat = sigma2*inv_xt_x
        con_std_err = np.sqrt(np.diagonal(
                                contrasts.dot(cov_beta_hat).dot(contrasts.T)))
        # Looking for a better way to fix this than the following.
        con_std_err = np.expand_dims(con_std_err, axis=1)
        con_t = np.divide(con_est, con_std_err)
        con_p = scipy.stats.t.sf(abs(con_t), df) * 2
    
    return np.array(con_est).flatten(), np.array(con_t).flatten(), \
        np.array(con_p).flatten(), pred_y, n_basis, r_squared 


def calc_cor_over_noise_range(nsim, n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max = 5, win_sub_noise_sd_range = .1, center_rt=True,
              beta_scales_yes = 1, beta_scales_no = 1):
    cor_est_scales_no_filt = []
    #cor_est_unfilt = []
    cor_est_scales_yes_filt = []
    for win_sub_noise_sd_loop in win_sub_noise_sd_range:
        #print(win_sub_noise_sd_loop)
        eff_size_out = sim_avg_eff_size(nsim, n_trials, scan_length, repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, win_sub_noise_sd_loop, center_rt,
              beta_scales_yes, beta_scales_no)
        cor_est_scales_yes_filt.append(eff_size_out['sim_effect_scales_yes_hp_filt'])
        cor_est_scales_no_filt.append(eff_size_out['sim_effect_scales_no_hp_filt'])
    output = {'cor_est_scales_yes_filt': cor_est_scales_yes_filt, 
               'cor_est_scales_no_filt': cor_est_scales_no_filt}
    return output



def sim_avg_eff_size(nsim, n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max = 5, win_sub_noise_sd=0.1, center_rt=True,
              beta_scales_yes = 1, beta_scales_no = 1):
    """STUFF
    """
    eff_size_scales_yes_filt_no_sim = []
    eff_size_scales_yes_filt_yes_sim = []
    eff_size_scales_no_filt_yes_sim = []
    eff_size_scales_yes_filt_yes_sim = []

    for sim in range(0, nsim):
        regressors, _ = make_regressors_one_trial_type(n_trials, scan_length, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, ISI_max, center_rt)
        dv_scales_yes = beta_scales_yes*regressors['rt']['unmodulated'] + \
                     np.random.normal(0, win_sub_noise_sd, (scan_length, 1))
        dv_scales_no = beta_scales_no*regressors['fixed_zero']['unmodulated']+ \
                     np.random.normal(0, win_sub_noise_sd, (scan_length, 1))                              
        eff_size_dv_scales_yes_filtered_no = np.corrcoef(regressors['rt']['unmodulated'].T, dv_scales_yes.T)[1,0]
        eff_size_dv_scales_no_filtered_no = np.corrcoef(regressors['fixed_zero']['unmodulated'].T, dv_scales_no.T)[1,0]
        x_duration_rt_only = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['rt']['unmodulated']), axis=1)
        contrasts = np.array([[0, 1]])
        _, t_dv_scales_yes, _, _, n_basis, _ = runreg(dv_scales_yes, x_duration_rt_only, contrasts, hp_filter=True, compute_stats=True)
        eff_size_dv_scales_yes_filtered = t_dv_scales_yes/(scan_length - (2 + n_basis) + t_dv_scales_yes ** 2) ** .5
        eff_size_scales_yes_filt_yes_sim.append(eff_size_dv_scales_yes_filtered)
        x_cons_dur_only = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['fixed_zero']['unmodulated']), axis=1)
        _, t_dv_scales_no, _, _, n_basis, _ = runreg(dv_scales_no, x_cons_dur_only, contrasts, hp_filter=True, compute_stats=True)
        eff_size_dv_scales_no_filtered = t_dv_scales_no/(scan_length - (2 + n_basis) + t_dv_scales_no ** 2) ** .5
        eff_size_scales_no_filt_yes_sim.append(eff_size_dv_scales_no_filtered)
        output = {'sim_effect_scales_yes_hp_filt' : np.mean(eff_size_scales_yes_filt_yes_sim),
                  'sim_effect_scales_no_hp_filt' : np.mean(eff_size_scales_no_filt_yes_sim)}
    return  output

def make_empty_dict_model_keys():
    models = {'Impulse Duration',
              'Fixed/RT Duration (orth)',
              'Stimulus Duration',
              'Mean RT Duration',
              'RT Duration only',
              'No RT effect'}
    dependent_variables = {'dv_scales_yes',
            'dv_scales_no'}
    empty_model_dict = {key: {key2: [] for key2 in models} for key in 
        dependent_variables}
    return empty_model_dict

def est_win_sub_mod_sd(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, center_rt=True,
              hp_filter=True, nsim=100):
    rt_dur_des_sd = []
    zero_dur_des_sd = []
    for i in range(0, nsim):
        regressors, _ = make_regressors_one_trial_type(n_trials, scan_length, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, ISI_max, center_rt)
        reg_shape = (scan_length, 1)
        x_duration_0 = np.concatenate((np.ones(reg_shape),
                    regressors['fixed_zero']['unmodulated'],
                    regressors['fixed_zero']['modulated']), axis=1)
        x_duration_event_duration = np.concatenate((np.ones(reg_shape),
                    regressors['fixed_event_duration']['unmodulated'],
                    regressors['fixed_event_duration']['modulated']), axis=1)
        x_duration_mean_rt = np.concatenate((np.ones(reg_shape),
                    regressors['fixed_mean_rt']['unmodulated'],
                    regressors['fixed_mean_rt']['modulated']), axis=1)
        x_duration_rt_only = np.concatenate((np.ones(reg_shape),
                    regressors['rt']['unmodulated']), axis=1)
        x_duration_event_only = np.concatenate((np.ones(reg_shape),
                    regressors['fixed_event_duration']['unmodulated']), axis=1)
        x_duration_event_duration_rt_orth = np.concatenate((np.ones(reg_shape),
                    regressors['fixed_event_duration']['unmodulated'],
                    regressors['rt_orth']['unmodulated']), axis=1)

        models = {'Impulse Duration': x_duration_0,
              'Fixed/RT Duration (orth)': x_duration_event_duration_rt_orth,
              'Stimulus Duration': x_duration_event_duration,
              'Mean RT Duration': x_duration_mean_rt,
              'RT Duration only': x_duration_rt_only,
              'No RT effect': x_duration_event_only}
        win_sub_var_est =  {key: {} for key in models}
        for model_name, model_mtx in models.items():
            if hp_filter == True:
                n_time_points = model_mtx.shape[0]
                dct_basis = _cosine_drift(.01, np.arange(n_time_points))
                dct_basis = np.delete(dct_basis, -1, axis = 1)
                model_mtx = np.concatenate((model_mtx, dct_basis), axis = 1)
            win_sub_var_est[model_name] = np.linalg.inv(model_mtx.T.dot(model_mtx))[1, 1]
        rt_dur_des_sd.append(np.sqrt(win_sub_var_est['RT Duration only']))
        zero_dur_des_sd.append(np.sqrt(win_sub_var_est['Impulse Duration']))
        output = {'des_sd_rt_dur': np.mean(rt_dur_des_sd), 
                  'des_sd_zero_dur': np.mean(zero_dur_des_sd)}
    return(output)



def sim_fit_sub(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, win_sub_noise_sd=0.1, center_rt=True,
              beta_scales_yes = 1, beta_scales_no = 1, hp_filter=True):
    """Runs 3 models with two different dependent variables and estimates
    unmodulated parameter estimate.  For 
    first model, dependent variable is BOLD signal that scales with reaction
    time (duration is reaction time) while dependent variable for second model
    doesn't scale with reaction time (duration is max_rt).  

    The three models each include 2 regressors and unmodulated regressor and
    one modulated by mean centered reaction time.  The difference is the 
    durations of the two regressors which are set to 0s, event duration (max_rt)
    and mean of the reaction time across trials for each of the 3 models.

    """
    regressors, mean_rt = make_regressors_one_trial_type(n_trials, scan_length, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, ISI_max, center_rt)
    dv_scales_yes = beta_scales_yes*regressors['rt']['unmodulated'] + \
                     np.random.normal(0, win_sub_noise_sd, (scan_length, 1))
    dv_scales_no = beta_scales_no*regressors['fixed_zero']['unmodulated']+ \
                     np.random.normal(0, win_sub_noise_sd, (scan_length, 1))
    
    x_duration_0 = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['fixed_zero']['unmodulated'],
                    regressors['fixed_zero']['modulated']), axis=1)
    x_duration_event_duration = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['fixed_event_duration']['unmodulated'],
                    regressors['fixed_event_duration']['modulated']), axis=1)
    x_duration_mean_rt = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['fixed_mean_rt']['unmodulated'],
                    regressors['fixed_mean_rt']['modulated']), axis=1)
    x_duration_rt_only = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['rt']['unmodulated']), axis=1)
    x_duration_event_only = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['fixed_event_duration']['unmodulated']), axis=1)
    x_duration_event_duration_rt_orth = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['fixed_event_duration']['unmodulated'],
                    regressors['rt_orth']['unmodulated']), axis=1)
    
    models = {'Impulse Duration': x_duration_0,
              'Fixed/RT Duration (orth)': x_duration_event_duration_rt_orth,
              'Stimulus Duration': x_duration_event_duration,
              'Mean RT Duration': x_duration_mean_rt,
              'RT Duration only': x_duration_rt_only,
              'No RT effect': x_duration_event_only}
    dependent_variables = {'dv_scales_yes': dv_scales_yes,
            'dv_scales_no': dv_scales_no}
    
    unmod_beta_est = {key: {key2: {} for key2 in models} for key in 
        dependent_variables}
    rtmod_beta_est = {key: {key2: {} for key2 in models} for key in 
        dependent_variables}
    unmod_beta_p = {key: {key2: {} for key2 in models} for key in 
        dependent_variables}
    rtmod_beta_p = {key: {key2: {} for key2 in models} for key in 
        dependent_variables}
    model_r2 = {key: {key2: {} for key2 in models} for key in 
        dependent_variables}
    for model_name, model_mtx in models.items():
        for dependent_variable_name, dv in dependent_variables.items():
            if model_mtx.shape[1] == 2:
                contrasts = np.array([[0, 1]])
            if model_mtx.shape[1] == 3:
                contrasts = np.array([[0, 1, 0], [0, 0, 1]])
            con_estimates, _, p_values, _, _, r2  = runreg(dv, model_mtx, contrasts, 
            hp_filter, compute_stats=True)
            model_r2[dependent_variable_name][model_name] = r2
            if (model_mtx.shape[1] == 2) & (model_name == 'RT Duration only'):
                unmod_beta_est[dependent_variable_name][model_name] = np.nan
                unmod_beta_p[dependent_variable_name][model_name] = np.nan
                rtmod_beta_est[dependent_variable_name][model_name] = con_estimates[0]
                rtmod_beta_p[dependent_variable_name][model_name] = p_values[0]
            elif (model_mtx.shape[1] == 2) & (model_name == 'No RT effect'):
                unmod_beta_est[dependent_variable_name][model_name] = con_estimates[0]
                unmod_beta_p[dependent_variable_name][model_name] = p_values[0]
                rtmod_beta_est[dependent_variable_name][model_name] = np.nan
                rtmod_beta_p[dependent_variable_name][model_name] = np.nan
            else:
                unmod_beta_est[dependent_variable_name][model_name] = con_estimates[0]
                unmod_beta_p[dependent_variable_name][model_name] = p_values[0]
                rtmod_beta_est[dependent_variable_name][model_name] = con_estimates[1]
                rtmod_beta_p[dependent_variable_name][model_name] = p_values[1]
                
    return unmod_beta_est, rtmod_beta_est, unmod_beta_p, rtmod_beta_p, mean_rt, model_r2


def lev1_many_subs(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, win_sub_noise_sd=0.1, btwn_sub_noise_sd=1, nsub = 50,
              center_rt=True, beta_scales_yes = 1, beta_scales_no = 1, hp_filter=True):
        
    """
    Stuff
    """
    unmod_beta_est_all_list_dict = list()
    rtmod_beta_est_all_list_dict = list()
    mean_rt_all = list()
    r2_all = list()

    for sub in range(0, nsub):
        beta_scales_yes_sub = beta_scales_yes + np.random.normal(0, btwn_sub_noise_sd, (1, 1))
        beta_scales_no_sub = beta_scales_no + np.random.normal(0, btwn_sub_noise_sd, (1, 1))
        unmod_beta_est_loop, rtmod_beta_est_loop, _, _, mean_rt, r2 = sim_fit_sub(n_trials, scan_length, 
            repetition_time, mu_expnorm, lam_expnorm,
            sigma_expnorm, max_rt, min_rt, event_duration, ISI_min, ISI_max, win_sub_noise_sd, center_rt,
            beta_scales_yes_sub, beta_scales_no_sub, hp_filter)
        unmod_beta_est_all_list_dict.append(unmod_beta_est_loop)
        rtmod_beta_est_all_list_dict.append(rtmod_beta_est_loop)
        mean_rt_all.append(mean_rt)
        r2_all.append(r2)

    # There must be a better way to do this
    unmod_beta_est_all_dict = {key: {key2: {} for key2 in \
        unmod_beta_est_all_list_dict[0]['dv_scales_yes']} for key in 
        unmod_beta_est_all_list_dict[0]}
    for i in unmod_beta_est_all_dict.keys():
        for j in unmod_beta_est_all_dict[i].keys():
            unmod_beta_est_all_dict[i][j] = \
                np.array(list(unmod_beta_est_all_dict[i][j] 
                for unmod_beta_est_all_dict in unmod_beta_est_all_list_dict))

    rtmod_beta_est_all_dict = {key: {key2: {} for key2 in \
        rtmod_beta_est_all_list_dict[0]['dv_scales_yes']} for key in 
        rtmod_beta_est_all_list_dict[0]}
    for i in rtmod_beta_est_all_dict.keys():
        for j in rtmod_beta_est_all_dict[i].keys():
            rtmod_beta_est_all_dict[i][j] = \
                np.array(list(rtmod_beta_est_all_dict[i][j] 
                for rtmod_beta_est_all_dict in rtmod_beta_est_all_list_dict))
    
    r2_dict = {key: {key2: {} for key2 in \
        r2_all[0]['dv_scales_yes']} for key in 
        r2_all[0]}
    for i in r2_dict.keys():
        for j in r2_dict[i].keys():
            r2_dict[i][j] = \
                np.array(list(r2_dict[i][j] 
                for r2_dict in r2_all))
    dat_out = {'unmod_beta_est': unmod_beta_est_all_dict, 
               'rtmod_beta_est': rtmod_beta_est_all_dict,
               'r2': r2_dict,
               'rt_mean': mean_rt_all}
    return dat_out


def sim_fit_group(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, win_sub_noise_sd=0.1, btwn_sub_noise_sd=1, nsub = 50,
              center_rt=True, beta_scales_yes = 1, beta_scales_no = 1, hp_filter=True):
    """FIX THIS Runs 3 models with two different dependent variables and estimates
    unmodulated parameter estimate.  For 
    first model, dependent variable is BOLD signal that scales with reaction
    time (duration is reaction time) while dependent variable for second model
    doesn't scale with reaction time (duration is max_rt).  

    The three models each include 2 regressors and unmodulated regressor and
    one modulated by mean centered reaction time.  The difference is the 
    durations of the two regressors which are set to 0s, event duration (max_rt)
    and mean of the reaction time across trials for each of the 3 models.
    """
    unmod_beta_est_all_list_dict = list()
    rtmod_beta_est_all_list_dict = list()
    mean_rt_all = list()
    r2_all = list()

    for sub in range(0, nsub):
        beta_scales_yes_sub = beta_scales_yes + np.random.normal(0, btwn_sub_noise_sd, (1, 1))
        beta_scales_no_sub = beta_scales_no + np.random.normal(0, btwn_sub_noise_sd, (1, 1))
        unmod_beta_est_loop, rtmod_beta_est_loop, _, _, mean_rt, r2 = sim_fit_sub(n_trials, scan_length, 
            repetition_time, mu_expnorm, lam_expnorm,
            sigma_expnorm, max_rt, min_rt, event_duration, ISI_min, ISI_max, win_sub_noise_sd, center_rt,
            beta_scales_yes_sub, beta_scales_no_sub, hp_filter)
        unmod_beta_est_all_list_dict.append(unmod_beta_est_loop)
        rtmod_beta_est_all_list_dict.append(rtmod_beta_est_loop)
        mean_rt_all.append(mean_rt)
        r2_all.append(r2)

    # There must be a better way to do this
    unmod_beta_est_all_dict = {key: {key2: {} for key2 in \
        unmod_beta_est_all_list_dict[0]['dv_scales_yes']} for key in 
        unmod_beta_est_all_list_dict[0]}
    for i in unmod_beta_est_all_dict.keys():
        for j in unmod_beta_est_all_dict[i].keys():
            unmod_beta_est_all_dict[i][j] = \
                np.array(list(unmod_beta_est_all_dict[i][j] 
                for unmod_beta_est_all_dict in unmod_beta_est_all_list_dict))

    rtmod_beta_est_all_dict = {key: {key2: {} for key2 in \
        rtmod_beta_est_all_list_dict[0]['dv_scales_yes']} for key in 
        rtmod_beta_est_all_list_dict[0]}
    for i in rtmod_beta_est_all_dict.keys():
        for j in rtmod_beta_est_all_dict[i].keys():
            rtmod_beta_est_all_dict[i][j] = \
                np.array(list(rtmod_beta_est_all_dict[i][j] 
                for rtmod_beta_est_all_dict in rtmod_beta_est_all_list_dict))

    unmod_cor_with_rt = {key: {key2: {} for key2 in unmod_beta_est_all_list_dict[0]['dv_scales_yes']} for key in 
        unmod_beta_est_all_list_dict[0]}
    rtmod_cor_with_rt = {key: {key2: {} for key2 in unmod_beta_est_all_list_dict[0]['dv_scales_yes']} for key in 
        rtmod_beta_est_all_list_dict[0]}
    unmod_1samp_t_pval = {key: {key2: {} for key2 in unmod_beta_est_all_list_dict[0]['dv_scales_yes']} for key in 
        unmod_beta_est_all_list_dict[0]}
    rtmod_1samp_t_pval = {key: {key2: {} for key2 in unmod_beta_est_all_list_dict[0]['dv_scales_yes']} for key in 
        unmod_beta_est_all_list_dict[0]}
    for i in unmod_cor_with_rt.keys():
        for j in unmod_cor_with_rt[i].keys():
            if np.isnan(unmod_beta_est_all_dict[i][j][0]):
                unmod_cor_with_rt[i][j] = float("nan")
            else: 
                unmod_cor_with_rt[i][j], _ = \
                    scipy.stats.pearsonr(unmod_beta_est_all_dict[i][j],
                    mean_rt_all)
    for i in rtmod_cor_with_rt.keys():
        for j in rtmod_cor_with_rt[i].keys():
            if np.isnan(rtmod_beta_est_all_dict[i][j][0]):
                rtmod_cor_with_rt[i][j] = float("nan")
            else:
                rtmod_cor_with_rt[i][j], _ = \
                    scipy.stats.pearsonr(rtmod_beta_est_all_dict[i][j],
                    mean_rt_all)
    for i in unmod_1samp_t_pval.keys():
        for j in unmod_1samp_t_pval[i].keys():
            X = [1] * len(mean_rt_all)
            ols_mod = sm.OLS(unmod_beta_est_all_dict[i][j], X)
            ols_p_values = ols_mod.fit().pvalues
            unmod_1samp_t_pval[i][j] = ols_p_values[0]
    for i in rtmod_1samp_t_pval.keys():
        for j in rtmod_1samp_t_pval[i].keys():
            X = [1] * len(mean_rt_all)
            ols_mod = sm.OLS(rtmod_beta_est_all_dict[i][j], X)
            ols_p_values = ols_mod.fit().pvalues
            rtmod_1samp_t_pval[i][j] = ols_p_values[0]
    return unmod_cor_with_rt, rtmod_cor_with_rt, unmod_1samp_t_pval, rtmod_1samp_t_pval
            

def many_sim_fit_group(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, win_sub_noise_sd=0.1, btwn_sub_noise_sd=1, 
              nsub=50, nsim=50, center_rt=True, beta_scales_yes=1, beta_scales_no=1, hp_filter=True):
    unmod_cor_with_rt_sims = list()
    rtmod_cor_with_rt_sims = list()
    unmod_1samp_t_pval_sims = list()
    rtmod_1samp_t_pval_sims = list()
    for sim in range(0, nsim):
        unmod_cor_with_rt_loop, rtmod_cor_with_rt_loop, unmod_1samp_t_pval_loop, rtmod_1samp_t_pval_loop = sim_fit_group(n_trials, scan_length,\
            repetition_time, mu_expnorm,
            lam_expnorm, sigma_expnorm, max_rt, 
            min_rt, event_duration, ISI_min, ISI_max, win_sub_noise_sd, btwn_sub_noise_sd, nsub, center_rt, 
            beta_scales_yes, beta_scales_no, hp_filter)
        unmod_cor_with_rt_sims.append(unmod_cor_with_rt_loop)
        rtmod_cor_with_rt_sims.append(rtmod_cor_with_rt_loop)
        unmod_1samp_t_pval_sims.append(unmod_1samp_t_pval_loop)
        rtmod_1samp_t_pval_sims.append(rtmod_1samp_t_pval_loop)

    unmod_cor_with_rt_sims_dict =  {key: {key2: {} for key2 in \
        unmod_cor_with_rt_sims[0]['dv_scales_yes']} for key in 
        unmod_cor_with_rt_sims[0]}   
    for i in unmod_cor_with_rt_sims_dict.keys():
        for j in unmod_cor_with_rt_sims_dict[i].keys():
            unmod_cor_with_rt_sims_dict[i][j] = tuple(unmod_cor_with_rt_sims_dict[i][j] 
                for unmod_cor_with_rt_sims_dict in unmod_cor_with_rt_sims)

    rtmod_cor_with_rt_sims_dict =  {key: {key2: {} for key2 in \
        rtmod_cor_with_rt_sims[0]['dv_scales_yes']} for key in 
        rtmod_cor_with_rt_sims[0]}   
    for i in rtmod_cor_with_rt_sims_dict.keys():
        for j in rtmod_cor_with_rt_sims_dict[i].keys():
            rtmod_cor_with_rt_sims_dict[i][j] = tuple(rtmod_cor_with_rt_sims_dict[i][j] 
                for rtmod_cor_with_rt_sims_dict in rtmod_cor_with_rt_sims) 
    unmod_1samp_t_pval_dict =  {key: {key2: {} for key2 in \
        unmod_1samp_t_pval_sims[0]['dv_scales_yes']} for key in 
        unmod_1samp_t_pval_sims[0]}   
    for i in unmod_1samp_t_pval_dict.keys():
        for j in unmod_1samp_t_pval_dict[i].keys():
            unmod_1samp_t_pval_dict[i][j] = tuple(unmod_1samp_t_pval_dict[i][j] 
                for unmod_1samp_t_pval_dict in unmod_1samp_t_pval_sims)
    rtmod_1samp_t_pval_dict =  {key: {key2: {} for key2 in \
        rtmod_1samp_t_pval_sims[0]['dv_scales_yes']} for key in 
        rtmod_1samp_t_pval_sims[0]}   
    for i in rtmod_1samp_t_pval_dict.keys():
        for j in rtmod_1samp_t_pval_dict[i].keys():
            rtmod_1samp_t_pval_dict[i][j] = tuple(rtmod_1samp_t_pval_dict[i][j] 
                for rtmod_1samp_t_pval_dict in rtmod_1samp_t_pval_sims)
    return unmod_cor_with_rt_sims_dict, rtmod_cor_with_rt_sims_dict, unmod_1samp_t_pval_dict, rtmod_1samp_t_pval_dict


def calc_pow_range(n_trials, scan_length, repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min_max_vec, win_sub_noise_sd,  
              center_rt, beta_scales_yes, beta_scales_no, hp_filter, nsim_pow):
    """Stuff
    """
    output_unmod_beta_power = {}
    output_rtmod_beta_power = {}
    for isi_loop in ISI_min_max_vec:
        print(isi_loop)
        output_unmod_beta_power[isi_loop] = make_empty_dict_model_keys()
        output_rtmod_beta_power[isi_loop] = make_empty_dict_model_keys()
        ISI_min = isi_loop[0]
        ISI_max = isi_loop[1]
        calc_cor_out = calc_cor_over_noise_range(100, n_trials, scan_length, repetition_time, mu_expnorm,
                    lam_expnorm, sigma_expnorm, max_rt,
                    min_rt, event_duration, ISI_min, ISI_max, win_sub_noise_sd, center_rt,
                    beta_scales_yes, beta_scales_no)
        output_unmod_beta_power[isi_loop]['cor_est_scales_no_filt'] = calc_cor_out['cor_est_scales_no_filt']
        output_rtmod_beta_power[isi_loop]['cor_est_scales_yes_filt'] = calc_cor_out['cor_est_scales_yes_filt']
        for win_sub_noise_sd_loop in win_sub_noise_sd:
            unmod_beta_p_vec = make_empty_dict_model_keys()
            rtmod_beta_p_vec = make_empty_dict_model_keys()  
            for sim in range(0, nsim_pow):
                unmod_beta_est, rtmod_beta_est, unmod_beta_p, rtmod_beta_p, mean_rt, _ = sim_fit_sub(n_trials, scan_length, repetition_time, mu_expnorm,
                        lam_expnorm, sigma_expnorm, max_rt, 
                        min_rt, event_duration, ISI_min, ISI_max, win_sub_noise_sd_loop, center_rt,
                        beta_scales_yes, beta_scales_no, hp_filter)
                for i in unmod_beta_p_vec.keys():
                    for j in unmod_beta_p_vec[i].keys():
                        unmod_beta_p_vec[i][j].append(unmod_beta_p[i][j])
                        rtmod_beta_p_vec[i][j].append(rtmod_beta_p[i][j])
            
            for i in unmod_beta_p_vec.keys():
                for j in unmod_beta_p_vec[i].keys():
                    output_unmod_beta_power[isi_loop][i][j].append(np.mean(np.array(unmod_beta_p_vec[i][j]) <= 0.05))
                    output_rtmod_beta_power[isi_loop][i][j].append(np.mean(np.array(rtmod_beta_p_vec[i][j]) <= 0.05))
            
    return output_unmod_beta_power, output_rtmod_beta_power


def pow_within_sub_range_isi(n_trials, scan_length, 
             repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min_max_vec, win_sub_noise_sd_range,  
              center_rt, beta_scales_yes, beta_scales_no, hp_filter, nsim_pow):
    output = {}
    for i in range(len(ISI_min_max_vec)):  
    #print(i)
        output[ISI_min_max_vec[i]] = {}
        calc_cor_out = calc_cor_over_noise_range(100, n_trials, scan_length, repetition_time, mu_expnorm,
                    lam_expnorm, sigma_expnorm, max_rt,
                    min_rt, event_duration, ISI_min_max_vec[i][0], ISI_min_max_vec[i][1], win_sub_noise_sd_range, center_rt,
                    beta_scales_yes, beta_scales_no)
        pow_var_epoch_mod, pow_unmod_var_impulse, pow_rtmod_var_impulse, _, _ = calc_pow_range(n_trials, scan_length, 
             repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min_max_vec[i][0], ISI_min_max_vec[i][1], win_sub_noise_sd_range,  
              center_rt, beta_scales_yes, beta_scales_no, hp_filter, nsim_pow)
        output[ISI_min_max_vec[i]]['correlation_scales_yes'] = calc_cor_out['cor_est_scales_yes_filt']
        output[ISI_min_max_vec[i]]['correlation_scales_no'] = calc_cor_out['cor_est_scales_no_filt']
        output[ISI_min_max_vec[i]]['power_var_epoch'] = pow_var_epoch_mod
        output[ISI_min_max_vec[i]]['pow_unmod_var_impulse'] = pow_unmod_var_impulse
        output[ISI_min_max_vec[i]]['pow_rtmod_var_impulse'] = pow_rtmod_var_impulse
    return output


def power_plot_1sub(output_unmod_beta_power, output_rtmod_beta_power, sim_type='dv_scales_yes',
                    zoom=False):
    isi_labels = list(output_unmod_beta_power.keys())
    nrows_plot = 2
    ncols_plot = len(isi_labels)//2
    fig, axs = plt.subplots(nrows_plot, ncols_plot, sharex=True, sharey=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Effect size (correlation)")
    plt.ylabel("Power")
    #fig.text(0.5, 0.04, 'common X', ha='center')
    #fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
    if zoom!=False:
        plt.setp(axs, xlim=zoom)
    for i in range(len(isi_labels)):
        if sim_type == 'dv_scales_yes':
            correlation = output_rtmod_beta_power[isi_labels[i]]['cor_est_scales_yes_filt']
        if sim_type == 'dv_scales_no':
            correlation = output_unmod_beta_power[isi_labels[i]]['cor_est_scales_no_filt']
        panel_row = i//ncols_plot
        panel_col = i%ncols_plot
        line1, = axs[panel_row, panel_col].plot(
                     correlation, 
                     output_rtmod_beta_power[isi_labels[i]][sim_type]['RT Duration only'], 
                     'tab:blue', label = 'RT duration')
        line2, = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_unmod_beta_power[isi_labels[i]][sim_type]['Impulse Duration'], 
                    'tab:green', label = 'Const (impulse)') 
        line3,  = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_rtmod_beta_power[isi_labels[i]][sim_type]['Impulse Duration'], 
                    'tab:green', linestyle = 'dashed', label = 'RT modulated (impulse)')
        line4, = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_unmod_beta_power[isi_labels[i]][sim_type]['Fixed/RT Duration (orth)'], 
                    color='tab:orange', label = 'Const (stimulus duration)') 
        line5,  = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_rtmod_beta_power[isi_labels[i]][sim_type]['Fixed/RT Duration (orth)'], 
                    color='tab:orange',linestyle='dashed', label = 'Orthogonalized RT duration')
        line6,  = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_unmod_beta_power[isi_labels[i]][sim_type]['Stimulus Duration'], 
                    color='tab:purple', label = 'Const (stimulus duration)')
        line7,  = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_rtmod_beta_power[isi_labels[i]][sim_type]['Stimulus Duration'], 
                    color='tab:purple',linestyle='dashed', label = 'RT modulated (stimulus duration)')
        line8,  = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_unmod_beta_power[isi_labels[i]][sim_type]['Mean RT Duration'], 
                    color='tab:red', label = 'Const (Mean RT duration)')
        line9,  = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_rtmod_beta_power[isi_labels[i]][sim_type]['Mean RT Duration'], 
                    color='tab:red',linestyle='dashed', label = 'RT modulated (mean RT duration)')
        line10,  = axs[panel_row, panel_col].plot(
                    correlation, 
                    output_unmod_beta_power[isi_labels[i]][sim_type]['No RT effect'], 
                    color='tab:olive', label = 'Const (stimulus duration)')
        #axs[panel_row, panel_col].set_xscale('log') 
        axs[panel_row, panel_col].set_title(f'ISI=U{isi_labels[i]}')
    fig.tight_layout()
    #plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10], 
    #           loc='center right', bbox_to_anchor=(panel_col+1.9, panel_row/2+1), ncol=1)
    plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10], 
              loc='center right', bbox_to_anchor=(panel_col+.7, panel_row/2), ncol=1)
    plt.show()


def plot_cor_violin(unmod_cor_with_rt_corplot, rtmod_cor_with_rt_corplot, 
                    mu_expnorm, lam_expnorm, sigma_expnorm, ISI_min, ISI_max, 
                    win_sub_noise_sd, btwn_sub_noise_sd, nsub, nsim, 
                    beta_scales_yes, beta_scales_no, hp_filter):
    unmodulated_scales_yes = pd.DataFrame(unmod_cor_with_rt_corplot['dv_scales_yes'])
    unmodulated_scales_yes['True Signal'] = 'Scales with RT'
    unmodulated_scales_no = pd.DataFrame(unmod_cor_with_rt_corplot['dv_scales_no'])
    unmodulated_scales_no['True Signal'] = 'Does not scale with RT'
    dat_unmodulated = pd.concat([unmodulated_scales_yes, unmodulated_scales_no])
    unmod_long = pd.melt(dat_unmodulated, id_vars = 'True Signal')
    unmod_long['Lower Level Estimate'] = 'Stimulus vs baseline'

    rtmodulated_scales_yes = pd.DataFrame(rtmod_cor_with_rt_corplot['dv_scales_yes'])
    rtmodulated_scales_yes['True Signal'] = 'Scales with RT'
    rtmodulated_scales_no = pd.DataFrame(rtmod_cor_with_rt_corplot['dv_scales_no'])
    rtmodulated_scales_no['True Signal'] = 'Does not scale with RT'
    dat_rtmodulated = pd.concat([rtmodulated_scales_yes, rtmodulated_scales_no])
    rtmod_long = pd.melt(dat_rtmodulated, id_vars = 'True Signal')
    rtmod_long['Lower Level Estimate'] = 'RT modulation'

    dat_long = pd.concat([rtmod_long, unmod_long])
    dat_long['Lower Level Estimate'] = \
                          dat_long['Lower Level Estimate'].astype('category')
    dat_long['Lower Level Estimate'] = \
             dat_long['Lower Level Estimate'].cat.reorder_categories(\
                 ['Stimulus vs baseline',
                 'RT modulation']) 
    cor_t_cutoff = abs(t.ppf(.025, nsub))
    cor_cutoff = cor_t_cutoff/(nsub - 2 +cor_t_cutoff**2)**.5

    sns.set_theme(style="whitegrid", font_scale=2)
    g = sns.catplot(data = dat_long, x='variable', y='value',
                hue='True Signal', row='Lower Level Estimate', kind='violin',
                palette='bright',  aspect = 5, height =4)
    g.set_ylabels('Correlation',size=30)
    g.set_xlabels('',size=30, clear_inner=False)
    titles = [r'Cor($\hat\beta_{trial}$, $RT_{WS}$)', r'Cor($\hat\beta_{RT_{BT}}$ , $RT_{WS}$)']
    for count, ax in enumerate(g.axes.flatten()):
        ax.tick_params(labelbottom=True)
        ax.axhline(0, color='black')
        ax.axhline(cor_cutoff, linestyle='dashed', color='gray')
        ax.axhline(-1*cor_cutoff, linestyle='dashed', color='gray')
        ax.set_title(titles[count])
    plt.subplots_adjust(hspace=.5)
    fig_root = Path('/Users/jeanettemumford/Dropbox/Research/Projects/RT_sims/Figures/')
    plt.savefig(f"{fig_root}/rt_cor_plot_mu_{round(mu_expnorm, 2)}_laminv_{round(1/lam_expnorm, 2)}_sig_{round(sigma_expnorm, 2)}_isi_{ISI_min}_{ISI_max}_sw_{win_sub_noise_sd}_sb_{btwn_sub_noise_sd}_nsub{nsub}_byes{beta_scales_yes}_bno{beta_scales_no}_hpfilt_{hp_filter}.pdf",
            format='pdf', transparent=True, pad_inches=.5, bbox_inches='tight')
    plt.show()
    

def group_power(n_trials, scan_length, repetition_time, 
              mu_expnorm, lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, win_sub_noise_sd, btwn_sub_noise_sd_vec, nsub, nsim, 
              center_rt, beta_scales_yes, beta_scales_no, hp_filter):
    """
    """
    power_unmod_1sampt_all = []
    power_rtmod_1sampt_all = []
    for btwn_sub_noise_sd in btwn_sub_noise_sd_vec:
        unmod_cor_rt, rtmod_cor_rt, unmod_1sampt_p, rtmod_1sampt_p = \
            many_sim_fit_group(n_trials, scan_length, repetition_time, 
                mu_expnorm, lam_expnorm, sigma_expnorm, max_rt, 
                min_rt, event_duration, ISI_min, ISI_max, win_sub_noise_sd, btwn_sub_noise_sd, nsub, nsim, 
                center_rt, beta_scales_yes, beta_scales_no, hp_filter)
        power_unmod_1sampt, power_rtmod_1sampt = calc_pow_single_group(unmod_1sampt_p, rtmod_1sampt_p)
        power_unmod_1sampt_all.append(power_unmod_1sampt)
        power_rtmod_1sampt_all.append(power_rtmod_1sampt)
        print(f"between-subject SD = {btwn_sub_noise_sd}\n")
        plot_cor_violin(unmod_cor_rt, rtmod_cor_rt, 
                    mu_expnorm, lam_expnorm, sigma_expnorm, ISI_min, ISI_max, 
                    win_sub_noise_sd, btwn_sub_noise_sd, nsub, nsim, 
                    beta_scales_yes, beta_scales_no, hp_filter)
    power_unmod_1sampt_output = defaultdict(dict)
    power_rtmod_1sampt_output = defaultdict(dict)
    for k in power_unmod_1sampt_all[0].keys():
        for j in power_unmod_1sampt_all[0]['dv_scales_yes'].keys():
            power_unmod_1sampt_output[k][j] = \
                tuple(val[k][j] for 
                      val in 
                      power_unmod_1sampt_all)
            power_rtmod_1sampt_output[k][j] = \
                tuple(val[k][j] for 
                      val in 
                      power_rtmod_1sampt_all)
    return dict(power_unmod_1sampt_output), dict(power_rtmod_1sampt_output)  

def calc_pow_single_group(unmod_1sampt_p, rtmod_1sampt_p):
    power_unmod_1sampt = {key: {key2: {} for key2 in unmod_1sampt_p['dv_scales_yes']} \
       for key in unmod_1sampt_p}
    power_rtmod_1sampt = {key: {key2: {} for key2 in rtmod_1sampt_p['dv_scales_yes']} \
       for key in rtmod_1sampt_p}
    for i in unmod_1sampt_p.keys():
        for j in unmod_1sampt_p[i].keys():
           pvals_loop = np.array(unmod_1sampt_p[i][j])
           power_unmod_1sampt[i][j] = np.mean(pvals_loop<0.05)
           pvals_loop = np.array(rtmod_1sampt_p[i][j])
           power_rtmod_1sampt[i][j] = np.mean(pvals_loop<0.05)
    return power_unmod_1sampt, power_rtmod_1sampt


def power_plot_group(unmod_output_power, rtmod_output_power, btwn_sub_noise_sd_vec, sim_type='dv_scales_yes',
                    zoom=False):
    #fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    fig = plt.figure()
    axs = plt.subplot(111, frameon=False)
    #fig.add_subplot(111, frameon=False)
    #plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Between-subject SD")
    plt.ylabel("Power")
    if zoom!=False:
        plt.setp(axs, xlim=zoom)
    line1, = axs.plot(
                     btwn_sub_noise_sd_vec, 
                     rtmod_output_power[sim_type]['RT Duration only'], 
                     'tab:blue', label = 'RT duration')
    line2, = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    unmod_output_power[sim_type]['Impulse Duration'], 
                    'tab:green', label = 'Const (impulse)') 
    line3,  = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    rtmod_output_power[sim_type]['Impulse Duration'], 
                    'tab:green', linestyle = 'dashed', label = 'RT modulated (impulse)')
    line4, = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    unmod_output_power[sim_type]['Fixed/RT Duration (orth)'], 
                    color='tab:orange', label = 'Const (stimulus duration)') 
    line5,  = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    rtmod_output_power[sim_type]['Fixed/RT Duration (orth)'], 
                    color='tab:orange',linestyle='dashed', label = 'Orthogonalized RT duration')
    line6,  = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    unmod_output_power[sim_type]['Stimulus Duration'], 
                    color='tab:purple', label = 'Const (stimulus duration)')
    line7,  = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    rtmod_output_power[sim_type]['Stimulus Duration'], 
                    color='tab:purple',linestyle='dashed', label = 'RT modulated (stimulus duration)')
    line8,  = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    unmod_output_power[sim_type]['Mean RT Duration'], 
                    color='tab:red', label = 'Const (Mean RT duration)')
    line9,  = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    rtmod_output_power[sim_type]['Mean RT Duration'], 
                    color='tab:red',linestyle='dashed', label = 'RT modulated (mean RT duration)')
    line10,  = axs.plot(
                    btwn_sub_noise_sd_vec, 
                    unmod_output_power[sim_type]['No RT effect'], 
                    color='tab:olive', label = 'Const (stimulus duration)')
        #axs[panel_row, panel_col].set_xscale('log') 
    #axs[panel_row, panel_col].set_title(f'ISI=U{isi_labels[i]}')
    fig.tight_layout()
    #plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10], 
    #           loc='center right', bbox_to_anchor=(panel_col+1.9, panel_row/2+1), ncol=1)
    plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10], 
              loc='center right', bbox_to_anchor=(2.5, 1/2), ncol=1)
    plt.show()
