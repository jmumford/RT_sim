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
import time


def make_3column_onsets(onsets, durations, amplitudes):
    """Utility function to generate 3 column onset structure
    """
    return(np.transpose(np.c_[onsets, durations, amplitudes]))


def make_regressors_one_trial_type(n_trials, scan_length,
                                   repetition_time=1, mu_expnorm=600,
                                   lam_expnorm=1 / 100, sigma_expnorm=75,
                                   max_rt=2000, min_rt=0, event_duration=2, 
                                   ISI_min=2, ISI_max=5, center_rt=True):
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
    ISI = np.random.uniform(low=ISI_min, high=ISI_max, size=n_trials - 1)
    onsets = np.cumsum(np.append([5], rt_trials[0:(n_trials-1)]+ISI))
    frame_times = np.arange(0, scan_length*repetition_time, repetition_time)
    if center_rt:
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
                dur_stim_unmod = \
                    regressors['fixed_event_duration']['unmodulated']
                _, _, _, predy, _, _ = runreg(non_orth_dur_rt, 
                                    dur_stim_unmod, 
                                    contrasts, hp_filter=False, 
                                    compute_stats=False)
                regressors[duration_type][modulation_type] = \
                                    non_orth_dur_rt - predy 
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
        contrasts = np.concatenate((contrasts, np.zeros((n_contrasts, 
                                   n_basis))),axis = 1)

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


def calc_cor_over_noise_range(nsim, n_trials, scan_length, repetition_time=1, 
              mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max = 5, 
              win_sub_noise_sd_range_scales_yes = .1, 
              win_sub_noise_sd_range_scales_no = .1, center_rt=True,
              beta_scales_yes = 1, beta_scales_no = 1):
    if len(win_sub_noise_sd_range_scales_yes) != \
                         len(win_sub_noise_sd_range_scales_no):
        print("win_sub_noise_sd_range_scales_yes/no must have same lengths")
        return
    cor_est_scales_no_filt_yes = []
    cor_est_scales_yes_filt_yes = []
    cor_est_scales_no_filt_yes_sd = []
    cor_est_scales_yes_filt_yes_sd = []
    num_sd = len(win_sub_noise_sd_range_scales_yes)
    for sd_ind in range(0, num_sd):
        eff_size_out = sim_avg_eff_size(nsim, n_trials, scan_length, 
              repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, 
              win_sub_noise_sd_range_scales_yes[sd_ind], 
              win_sub_noise_sd_range_scales_no[sd_ind], center_rt,
              beta_scales_yes, beta_scales_no)
        cor_est_scales_yes_filt_yes.append(
                        eff_size_out['sim_effect_scales_yes_hp_filt_yes'])
        cor_est_scales_no_filt_yes.append(
                        eff_size_out['sim_effect_scales_no_hp_filt_yes'])
        cor_est_scales_yes_filt_yes_sd.append(
                        eff_size_out['sim_effect_scales_yes_hp_filt_yes_sd'])
        cor_est_scales_no_filt_yes_sd.append(
                        eff_size_out['sim_effect_scales_no_hp_filt_yes_sd'])
    output = {'cor_est_scales_yes_filt_yes': cor_est_scales_yes_filt_yes, 
               'cor_est_scales_no_filt_yes': cor_est_scales_no_filt_yes,
               'cor_est_scales_yes_filt_yes_sd': cor_est_scales_yes_filt_yes_sd, 
               'cor_est_scales_no_filt_yes_sd': cor_est_scales_no_filt_yes_sd}
    return output



def sim_avg_eff_size(nsim, n_trials, scan_length, repetition_time=1, 
               mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max = 5, 
              win_sub_noise_sd_scales_yes=0.1, 
              win_sub_noise_sd_scales_no=0.1, center_rt=True,
              beta_scales_yes = 1, beta_scales_no = 1):
    """STUFF
    """
    eff_size_scales_yes_filt_yes_sim = []
    eff_size_scales_no_filt_yes_sim = []  
    for sim in range(0, nsim):
        regressors, _ = make_regressors_one_trial_type(n_trials, scan_length, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, 
                                   ISI_max, center_rt)
        dv_scales_yes = beta_scales_yes*regressors['rt']['unmodulated'] + \
                     np.random.normal(0, win_sub_noise_sd_scales_yes, 
                                      (scan_length, 1))
        dv_scales_no = beta_scales_no*regressors['fixed_zero']['unmodulated']+ \
                     np.random.normal(0, win_sub_noise_sd_scales_no, 
                                      (scan_length, 1))                              
        x_duration_rt_only = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['rt']['unmodulated']), axis=1)
        contrasts = np.array([[0, 1]])
        _, t_dv_scales_yes, _, _, n_basis, _ = runreg(dv_scales_yes, 
                                         x_duration_rt_only, contrasts, 
                                         hp_filter=True, compute_stats=True)
        eff_size_dv_scales_yes_filtered = \
            t_dv_scales_yes/(scan_length - 
                                   (2 + n_basis) + t_dv_scales_yes ** 2) ** .5
        eff_size_scales_yes_filt_yes_sim.append(eff_size_dv_scales_yes_filtered)

        x_cons_dur_only = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['fixed_zero']['unmodulated']), axis=1)
        _, t_dv_scales_no, _, _, n_basis, _ = runreg(dv_scales_no, 
                                        x_cons_dur_only, contrasts,
                                        hp_filter=True, compute_stats=True)
        eff_size_dv_scales_no_filtered = \
            t_dv_scales_no/(scan_length - (2 + n_basis) + t_dv_scales_no ** 2) ** .5
        eff_size_scales_no_filt_yes_sim.append(eff_size_dv_scales_no_filtered)
        output = {'sim_effect_scales_yes_hp_filt_yes' : 
                        np.mean(eff_size_scales_yes_filt_yes_sim),
                  'sim_effect_scales_no_hp_filt_yes' : 
                        np.mean(eff_size_scales_no_filt_yes_sim),
                  'sim_effect_scales_yes_hp_filt_yes_sd' : 
                        np.std(eff_size_scales_yes_filt_yes_sim),
                  'sim_effect_scales_no_hp_filt_yes_sd' : 
                        np.std(eff_size_scales_no_filt_yes_sim)}
    return  output

def make_empty_dict_model_keys(numrep):
    models = {'Impulse Duration',
              'Fixed/RT Duration (orth)',
              'Stimulus Duration',
              'Mean RT Duration',
              'RT Duration only',
              'No RT effect'}
    dependent_variables = {'dv_scales_yes',
            'dv_scales_no'}
    empty_model_dict = {key: {key2: [None]*numrep for key2 in models} for key in 
        dependent_variables}
    return empty_model_dict

def make_design_matrices(regressors):
    """
    Input: regressor output from make_regressors_one_trial_type
    Output: Design matrices for models of interest in simulations
    """
    regressor_shape = regressors['fixed_zero']['unmodulated'].shape
    x_duration_0 = np.concatenate((np.ones(regressor_shape),
                    regressors['fixed_zero']['unmodulated'],
                    regressors['fixed_zero']['modulated']), axis=1)
    x_duration_event_duration = np.concatenate((np.ones(regressor_shape),
                    regressors['fixed_event_duration']['unmodulated'],
                    regressors['fixed_event_duration']['modulated']), axis=1)
    x_duration_mean_rt = np.concatenate((np.ones(regressor_shape),
                    regressors['fixed_mean_rt']['unmodulated'],
                    regressors['fixed_mean_rt']['modulated']), axis=1)
    x_duration_rt_only = np.concatenate((np.ones(regressor_shape),
                    regressors['rt']['unmodulated']), axis=1)
    x_duration_event_only = np.concatenate((np.ones(regressor_shape),
                    regressors['fixed_event_duration']['unmodulated']), axis=1)
    x_duration_event_duration_rt_orth = np.concatenate((np.ones(regressor_shape),
                    regressors['fixed_event_duration']['unmodulated'],
                    regressors['rt_orth']['unmodulated']), axis=1)
    
    models = {'Impulse Duration': x_duration_0,
              'Fixed/RT Duration (orth)': x_duration_event_duration_rt_orth,
              'Stimulus Duration': x_duration_event_duration,
              'Mean RT Duration': x_duration_mean_rt,
              'RT Duration only': x_duration_rt_only,
              'No RT effect': x_duration_event_only}
    return models

def est_win_sub_mod_sd(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, center_rt=True,
              hp_filter=True, nsim=100):
    """
    Function used to estimate the first level SD contribution from the design
      matrix, *only*.  Used along with within-subject sd to assess overall 
      level 1 variance estimate and ratio to mixed effects variance to 
      total within-subject variance when choosing simulation settings
    Input: RT and other settings for design matrix contstruciton
    Output: The square root of the diagonal element of inv(X'X) that corresponds
            to the RT-duration only model and Impulse duration only model,
            potentially accounting for highpass filtering.
    """
    rt_dur_des_sd = []
    zero_dur_des_sd = []
    for i in range(0, nsim):
        regressors, _ = make_regressors_one_trial_type(n_trials, scan_length, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, 
                                   ISI_max, center_rt)
        models = make_design_matrices(regressors)
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
        output = {'dv_scales_yes': np.mean(rt_dur_des_sd), 
                  'dv_scales_no': np.mean(zero_dur_des_sd)}
    return(output)


def sim_fit_sub(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, 
              win_sub_noise_sd_scales_yes=0.1, win_sub_noise_sd_scales_no=0.1, 
              center_rt=True, beta_scales_yes = 1, beta_scales_no = 1, 
              hp_filter=True):
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
                                   max_rt, min_rt, event_duration, ISI_min, 
                                   ISI_max, center_rt)
    dv_scales_yes = beta_scales_yes*regressors['rt']['unmodulated'] + \
                     np.random.normal(0, win_sub_noise_sd_scales_yes, 
                     (scan_length, 1))
    dv_scales_no = beta_scales_no*regressors['fixed_zero']['unmodulated']+ \
                     np.random.normal(0, win_sub_noise_sd_scales_no, 
                     (scan_length, 1))
    dependent_variables = {'dv_scales_yes': dv_scales_yes,
            'dv_scales_no': dv_scales_no}
    models = make_design_matrices(regressors)

    unmod_beta_est = make_empty_dict_model_keys(1)
    rtmod_beta_est = make_empty_dict_model_keys(1)
    unmod_beta_p = make_empty_dict_model_keys(1)
    rtmod_beta_p = make_empty_dict_model_keys(1)
    model_r2 = make_empty_dict_model_keys(1)
    for model_name, model_mtx in models.items():
        for dependent_variable_name, dv in dependent_variables.items():
            if model_mtx.shape[1] == 2:
                contrasts = np.array([[0, 1]])
            if model_mtx.shape[1] == 3:
                contrasts = np.array([[0, 1, 0], [0, 0, 1]])
            con_estimates, _, p_values, _, _, r2  = runreg(dv, model_mtx, 
                    contrasts, hp_filter, compute_stats=True)
            model_r2[dependent_variable_name][model_name] = r2
            if (model_mtx.shape[1] == 2) & (model_name == 'RT Duration only'):
                unmod_beta_est[dependent_variable_name][model_name] = np.nan
                unmod_beta_p[dependent_variable_name][model_name] = np.nan
                rtmod_beta_est[dependent_variable_name][model_name] = \
                    con_estimates[0]
                rtmod_beta_p[dependent_variable_name][model_name] = p_values[0]
            elif (model_mtx.shape[1] == 2) & (model_name == 'No RT effect'):
                unmod_beta_est[dependent_variable_name][model_name] = \
                    con_estimates[0]
                unmod_beta_p[dependent_variable_name][model_name] = p_values[0]
                rtmod_beta_est[dependent_variable_name][model_name] = np.nan
                rtmod_beta_p[dependent_variable_name][model_name] = np.nan
            else:
                unmod_beta_est[dependent_variable_name][model_name] = \
                    con_estimates[0]
                unmod_beta_p[dependent_variable_name][model_name] = p_values[0]
                rtmod_beta_est[dependent_variable_name][model_name] = \
                    con_estimates[1]
                rtmod_beta_p[dependent_variable_name][model_name] = p_values[1]
                
    return unmod_beta_est, rtmod_beta_est, unmod_beta_p, rtmod_beta_p, mean_rt,\
           model_r2

def list_dicts_to_dict_tuples(list_of_dicts):
    """
    Converts a list of dictionaries to a dicitonary of tuples
    list_of_dicts: A list of dictionaries where the key structure is identical 
    for all dictionaries.

    Assumes dictionary has two levels (dictionary within-dictionary) where the
    keys of the first level dictionaries match
    """
    out_dict = defaultdict(dict)
    top_level_keys = list(list_of_dicts[0].keys())
    second_level_keys = list(list_of_dicts[0][top_level_keys[0]].keys())

    for i in top_level_keys:
        for j in second_level_keys:
            out_dict[i][j] = tuple(val[i][j] for val in list_of_dicts)
    return dict(out_dict)
    
     

def calc_win_sub_pow_range(n_trials, scan_length, repetition_time, mu_expnorm,
              lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min_max_vec,
              win_sub_noise_sd_range_scales_yes, 
              win_sub_noise_sd_range_scales_no,  
              center_rt, beta_scales_yes, beta_scales_no, hp_filter, nsim_pow):
    """Stuff
    """
    if len(win_sub_noise_sd_range_scales_yes) != \
                         len(win_sub_noise_sd_range_scales_no):
        print("win_sub_noise_sd_range_scales_yes/no must have same lengths")
        return
    num_sd = len(win_sub_noise_sd_range_scales_yes)
    output_unmod_beta_power = {}
    output_rtmod_beta_power = {}
    for isi_loop in ISI_min_max_vec:
        print(isi_loop)
        output_unmod_beta_power[isi_loop] = make_empty_dict_model_keys(num_sd)
        output_rtmod_beta_power[isi_loop] = make_empty_dict_model_keys(num_sd)
        ISI_min = isi_loop[0]
        ISI_max = isi_loop[1]
        calc_cor_out = calc_cor_over_noise_range(100, n_trials, scan_length, 
                    repetition_time, mu_expnorm,lam_expnorm, sigma_expnorm, 
                    max_rt,min_rt, event_duration, ISI_min, ISI_max, 
                    win_sub_noise_sd_range_scales_yes, 
                    win_sub_noise_sd_range_scales_no, 
                    center_rt,beta_scales_yes, beta_scales_no)
        output_unmod_beta_power[isi_loop]['cor_est_scales_no_filt_yes'] = \
                    calc_cor_out['cor_est_scales_no_filt_yes']
        output_rtmod_beta_power[isi_loop]['cor_est_scales_yes_filt_yes'] = \
                    calc_cor_out['cor_est_scales_yes_filt_yes']
        for sd_ind in range(0, num_sd):
            unmod_beta_p_vec = make_empty_dict_model_keys(nsim_pow)
            rtmod_beta_p_vec = make_empty_dict_model_keys(nsim_pow) 
            start_time = time.time() 
            for sim in range(0, nsim_pow):
                if sim%1000 == 0:
                    print(sim)
                    print(start_time - time.time())
                    start_time = time.time()
                _, _, unmod_beta_p, rtmod_beta_p, _, _ = \
                    sim_fit_sub(n_trials, scan_length, repetition_time, 
                                mu_expnorm,lam_expnorm, sigma_expnorm, max_rt, 
                                min_rt, event_duration, ISI_min, ISI_max, 
                                win_sub_noise_sd_range_scales_yes[sd_ind], 
                                win_sub_noise_sd_range_scales_no[sd_ind], 
                                center_rt, beta_scales_yes, beta_scales_no,
                                hp_filter)
                for i in unmod_beta_p_vec.keys():
                    for j in unmod_beta_p_vec[i].keys():
                        unmod_beta_p_vec[i][j][sim] = unmod_beta_p[i][j]
                        rtmod_beta_p_vec[i][j][sim] = rtmod_beta_p[i][j]
            print('calculating power')              
            for i in unmod_beta_p_vec.keys():
                for j in unmod_beta_p_vec[i].keys():
                    output_unmod_beta_power[isi_loop][i][j][sd_ind] = \
                        np.mean(np.array(unmod_beta_p_vec[i][j]) <= 0.05)
                    output_rtmod_beta_power[isi_loop][i][j][sd_ind]=\
                        np.mean(np.array(rtmod_beta_p_vec[i][j]) <= 0.05)       
    return output_unmod_beta_power, output_rtmod_beta_power


def power_plot_1sub(output_unmod_beta_power, output_rtmod_beta_power, 
                    sim_type='dv_scales_yes',zoom=False, display_plots=True):
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
            correlation = \
                output_rtmod_beta_power[isi_labels[i]]['cor_est_scales_yes_filt_yes']
        if sim_type == 'dv_scales_no':
            correlation = \
                output_unmod_beta_power[isi_labels[i]]['cor_est_scales_no_filt_yes']
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
        axs[panel_row, panel_col].set_title(f'ISI=U{isi_labels[i]}')
    fig.tight_layout()
    plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, 
               line9, line10], 
              loc='center right', bbox_to_anchor=(panel_col+.7, panel_row/2), 
              ncol=1)
    if display_plots == True:
        plt.show()


def plot_cor_violin(unmod_cor_with_rt_corplot, rtmod_cor_with_rt_corplot, 
                    mu_expnorm, lam_expnorm, sigma_expnorm, ISI_min, ISI_max, 
                    win_sub_noise_sd_scales_yes, win_sub_noise_sd_scales_no, 
                    btwn_sub_noise_sd_scales_yes, btwn_sub_noise_sd_scales_no, 
                    nsub, nsim, beta_scales_yes, beta_scales_no, hp_filter,
                    fig_dir, display_plots=True):
    fig_dir = Path(fig_dir)
    if fig_dir.exists() == False | fig_dir.is_dir() == False:
        print("Figure directory does not exist")
        return
    unmodulated_scales_yes = \
        pd.DataFrame(unmod_cor_with_rt_corplot['dv_scales_yes'])
    unmodulated_scales_yes['True Signal'] = 'Scales with RT'
    unmodulated_scales_no = \
        pd.DataFrame(unmod_cor_with_rt_corplot['dv_scales_no'])
    unmodulated_scales_no['True Signal'] = 'Does not scale with RT'
    dat_unmodulated = pd.concat([unmodulated_scales_yes, unmodulated_scales_no])
    unmod_long = pd.melt(dat_unmodulated, id_vars = 'True Signal')
    unmod_long['Lower Level Estimate'] = 'Stimulus vs baseline'

    rtmodulated_scales_yes = \
        pd.DataFrame(rtmod_cor_with_rt_corplot['dv_scales_yes'])
    rtmodulated_scales_yes['True Signal'] = 'Scales with RT'
    rtmodulated_scales_no = \
        pd.DataFrame(rtmod_cor_with_rt_corplot['dv_scales_no'])
    rtmodulated_scales_no['True Signal'] = 'Does not scale with RT'
    dat_rtmodulated = pd.concat([rtmodulated_scales_yes, rtmodulated_scales_no])
    rtmod_long = pd.melt(dat_rtmodulated, id_vars = 'True Signal')
    rtmod_long['Lower Level Estimate'] = 'RT modulation'

    dat_long = pd.concat([rtmod_long, unmod_long])
    dat_long = dat_long.loc[dat_long['variable'] != 'Fixed/RT Duration (orth)']
    dat_long['Lower Level Estimate'] = \
                          dat_long['Lower Level Estimate'].astype('category')
    dat_long['Lower Level Estimate'] = \
             dat_long['Lower Level Estimate'].cat.reorder_categories(\
                 ['Stimulus vs baseline',
                 'RT modulation']) 
    dat_long['variable'] = \
        dat_long['variable'].astype('category')          
    dat_long['variable'] = \
        dat_long['variable'].cat.reorder_categories(\
                 ['Impulse Duration',
                 'Stimulus Duration', 
                  'Mean RT Duration',
                  'RT Duration only',
                  'No RT effect'])
    cor_t_cutoff = abs(t.ppf(.025, nsub))
    cor_cutoff = cor_t_cutoff/(nsub - 2 +cor_t_cutoff**2)**.5
    sns.set_theme(style="whitegrid", font_scale=2)
    g = sns.catplot(data = dat_long,
                x='variable', y='value',
                hue='True Signal', row='Lower Level Estimate', kind='violin',
                palette='gray',  aspect = 5, height =4)
    g.set_ylabels('Correlation',size=30)
    g.set_xlabels('',size=30, clear_inner=False)
    titles = [r'Cor($\hat\beta_{trial}$, $RT_{WS}$)', 
             r'Cor($\hat\beta_{RT_{BT}}$ , $RT_{WS}$)']
    for count, ax in enumerate(g.axes.flatten()):
        ax.tick_params(labelbottom=True)
        ax.axhline(0, color='black')
        ax.axhline(cor_cutoff, linestyle='dashed', color='gray')
        ax.axhline(-1*cor_cutoff, linestyle='dashed', color='gray')
        ax.set_title(titles[count])
    plt.subplots_adjust(hspace=.5)
    plt.savefig(f"{fig_dir}/rt_cor_plot_mu_{round(mu_expnorm, 2)}_laminv_{round(1/lam_expnorm, 2)}_sig_{round(sigma_expnorm, 2)}_isi_{ISI_min}_{ISI_max}_sw_scalesyes_no{win_sub_noise_sd_scales_yes}_{win_sub_noise_sd_scales_no}_sb_yes_no{btwn_sub_noise_sd_scales_yes}_{btwn_sub_noise_sd_scales_no}_nsub{nsub}_byes{beta_scales_yes}_bno{beta_scales_no}_hpfilt_{hp_filter}.pdf",
            format='pdf', transparent=True, pad_inches=.5, bbox_inches='tight')
    if display_plots == True:
        plt.show()
    

def sim_one_group(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, 
              win_sub_noise_sd_scales_yes=0.1, win_sub_noise_sd_scales_no=0.1, 
              btwn_sub_noise_sd_scales_yes=1, btwn_sub_noise_sd_scales_no=1, 
              nsub = 50, center_rt=True, beta_scales_yes = 1, 
              beta_scales_no = 1, hp_filter=True):
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
    r2_all_list_dict = list()
    mean_rt_all = list()
    
    for sub in range(0, nsub):
        beta_scales_yes_sub = beta_scales_yes + \
             np.random.normal(0, btwn_sub_noise_sd_scales_yes, (1, 1))
        beta_scales_no_sub = beta_scales_no + \
             np.random.normal(0, btwn_sub_noise_sd_scales_no, (1, 1))
        unmod_beta_est_loop, rtmod_beta_est_loop, _, _, mean_rt, r2 = \
            sim_fit_sub(n_trials, scan_length, 
            repetition_time, mu_expnorm, lam_expnorm,
            sigma_expnorm, max_rt, min_rt, event_duration, ISI_min, ISI_max, 
            win_sub_noise_sd_scales_yes, win_sub_noise_sd_scales_no, center_rt,
            beta_scales_yes_sub, beta_scales_no_sub, hp_filter)
        unmod_beta_est_all_list_dict.append(unmod_beta_est_loop)
        rtmod_beta_est_all_list_dict.append(rtmod_beta_est_loop)
        mean_rt_all.append(mean_rt)
        r2_all_list_dict.append(r2)

    unmod_beta_est_all_dict = \
        list_dicts_to_dict_tuples(unmod_beta_est_all_list_dict)
    rtmod_beta_est_all_dict = \
        list_dicts_to_dict_tuples(rtmod_beta_est_all_list_dict)
    r2_dict = list_dicts_to_dict_tuples(r2_all_list_dict)
    
    unmod_cor_with_rt = make_empty_dict_model_keys(1)
    rtmod_cor_with_rt = make_empty_dict_model_keys(1)
    unmod_1samp_t_pval = make_empty_dict_model_keys(1)
    rtmod_1samp_t_pval = make_empty_dict_model_keys(1)
    for i in unmod_cor_with_rt.keys():
        for j in unmod_cor_with_rt[i].keys():
            if np.isnan(unmod_beta_est_all_dict[i][j][0]):
                unmod_cor_with_rt[i][j] = float("nan")
            else: 
                unmod_cor_with_rt[i][j], _ = \
                    scipy.stats.pearsonr(unmod_beta_est_all_dict[i][j],
                    mean_rt_all)
            if np.isnan(rtmod_beta_est_all_dict[i][j][0]):
                rtmod_cor_with_rt[i][j] = float("nan")
            else:
                rtmod_cor_with_rt[i][j], _ = \
                    scipy.stats.pearsonr(rtmod_beta_est_all_dict[i][j],
                    mean_rt_all)
    for i in unmod_1samp_t_pval.keys():
        for j in unmod_1samp_t_pval[i].keys():
            X = [1] * len(mean_rt_all)
            ols_unmod = sm.OLS(unmod_beta_est_all_dict[i][j], X)
            ols_unmod_p_values = ols_unmod.fit().pvalues
            unmod_1samp_t_pval[i][j] = ols_unmod_p_values[0]
            ols_rtmod = sm.OLS(rtmod_beta_est_all_dict[i][j], X)
            ols_rtmod_p_values = ols_rtmod.fit().pvalues
            rtmod_1samp_t_pval[i][j] = ols_rtmod_p_values[0]
    return unmod_cor_with_rt, rtmod_cor_with_rt,\
           unmod_1samp_t_pval, rtmod_1samp_t_pval
       

def sim_many_group(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, 
              win_sub_noise_sd_scales_yes=0.1, win_sub_noise_sd_scales_no=0.1, 
              btwn_sub_noise_sd_scales_yes=1, btwn_sub_noise_sd_scales_no=1, 
              nsub=50, nsim=50, center_rt=True, beta_scales_yes=1, 
              beta_scales_no=1, hp_filter=True):
    unmod_cor_with_rt_list_dict = list()
    rtmod_cor_with_rt_list_dict = list()
    unmod_1samp_t_pval_list_dict = list()
    rtmod_1samp_t_pval_list_dict = list()
    for sim in range(0, nsim):
        unmod_cor_with_rt_loop, rtmod_cor_with_rt_loop, unmod_1samp_t_pval_loop, rtmod_1samp_t_pval_loop = \
            sim_one_group(n_trials, scan_length,repetition_time, mu_expnorm,
            lam_expnorm, sigma_expnorm, max_rt,  min_rt, event_duration, 
            ISI_min, ISI_max, win_sub_noise_sd_scales_yes, 
            win_sub_noise_sd_scales_no, btwn_sub_noise_sd_scales_yes, 
            btwn_sub_noise_sd_scales_no, nsub, center_rt, 
            beta_scales_yes, beta_scales_no, hp_filter)
        unmod_cor_with_rt_list_dict.append(unmod_cor_with_rt_loop)
        rtmod_cor_with_rt_list_dict.append(rtmod_cor_with_rt_loop)
        unmod_1samp_t_pval_list_dict.append(unmod_1samp_t_pval_loop)
        rtmod_1samp_t_pval_list_dict.append(rtmod_1samp_t_pval_loop)

    unmod_cor_with_rt_sims_dict = \
        list_dicts_to_dict_tuples(unmod_cor_with_rt_list_dict)
    rtmod_cor_with_rt_sims_dict =  \
        list_dicts_to_dict_tuples(rtmod_cor_with_rt_list_dict)
    unmod_1samp_t_pval_dict = \
        list_dicts_to_dict_tuples(unmod_1samp_t_pval_list_dict)
    rtmod_1samp_t_pval_dict = \
        list_dicts_to_dict_tuples(rtmod_1samp_t_pval_list_dict)
    unmod_1samp_t_pow = p_dict_to_power_dict(unmod_1samp_t_pval_dict)
    rtmod_1samp_t_pow = p_dict_to_power_dict(rtmod_1samp_t_pval_dict)
    return unmod_cor_with_rt_sims_dict, rtmod_cor_with_rt_sims_dict,\
           unmod_1samp_t_pow, rtmod_1samp_t_pow

def p_dict_to_power_dict(p_dict):
    pow_dict = defaultdict(dict)
    for i in p_dict.keys():
        for j in p_dict[i].keys():
           pvals_loop = np.array(p_dict[i][j])
           pow_dict[i][j] = np.mean(pvals_loop<0.05)
    return pow_dict



def calc_group_eff_size(all_sd_params, beta):
    """
    Calculates group effect size based on within subject model variability, 
      within subject variability and between subject variabiity
    """
    sim_type = ['dv_scales_yes', 'dv_scales_no']
    output  = dict.fromkeys(sim_type)
    for scale_type in sim_type:
        des_sd = all_sd_params['win_sub_mod_sd'][scale_type]
        win_sub_noise_sd = all_sd_params['win_sub_noise_sd'][scale_type]
        btwn_sub_noise_sd_vec = all_sd_params['btwn_noise_sd_vec'][scale_type]
        beta_loop = beta[scale_type]
        mfx_sd = np.sqrt((win_sub_noise_sd*des_sd)**2 + 
                        btwn_sub_noise_sd_vec**2)
        total_within_sd_ratio = mfx_sd/(des_sd*win_sub_noise_sd)
        cohens_d = beta_loop/mfx_sd
        output[scale_type] = {'total_within_sd_ratio': total_within_sd_ratio,
                            'cohens_d': cohens_d}
    return(output)

def group_power_range_btwn_sd(n_trials, scan_length, repetition_time, 
              mu_expnorm, lam_expnorm, sigma_expnorm, max_rt, 
              min_rt, event_duration, ISI_min, ISI_max, 
              all_sd_params, nsub, nsim, center_rt, beta, 
              hp_filter, fig_dir, display_plots=True):
    """
    Calculates power over a range of between-subject SD's
    Plots correlataions with RT in violin plots
    Returns power values
    """
    fig_dir_path = Path(fig_dir)
    if fig_dir_path.exists() == False | fig_dir_path.is_dir() == False:
        print("Figure directory does not exist")
        return
    btwn_sub_noise_sd_vec_scales_yes = \
        all_sd_params['btwn_noise_sd_vec']['dv_scales_yes']
    btwn_sub_noise_sd_vec_scales_no = \
        all_sd_params['btwn_noise_sd_vec']['dv_scales_no']
    if len(btwn_sub_noise_sd_vec_scales_yes) != \
                         len(btwn_sub_noise_sd_vec_scales_no):
        print("btwn_sub_noise_sd_vec_scales_yes/no must have same lengths")
        return
    power_unmod_1sampt_all = []
    power_rtmod_1sampt_all = []
    for btwn_sub_sd_ind in range(0, len(btwn_sub_noise_sd_vec_scales_yes)):
        unmod_cor_rt, rtmod_cor_rt, unmod_1sampt_pow, rtmod_1sampt_pow = \
            sim_many_group(n_trials, scan_length, repetition_time, 
                mu_expnorm, lam_expnorm, sigma_expnorm, max_rt, 
                min_rt, event_duration, ISI_min, ISI_max, 
                all_sd_params['win_sub_noise_sd']['dv_scales_yes'],
                all_sd_params['win_sub_noise_sd']['dv_scales_no'],
                btwn_sub_noise_sd_vec_scales_yes[btwn_sub_sd_ind],
                btwn_sub_noise_sd_vec_scales_no[btwn_sub_sd_ind], nsub, nsim, 
                center_rt, beta['dv_scales_yes'], beta['dv_scales_no'], hp_filter)
        power_unmod_1sampt_all.append(unmod_1sampt_pow)
        power_rtmod_1sampt_all.append(rtmod_1sampt_pow)
        print(f"Scales yes between-subject SD = {btwn_sub_noise_sd_vec_scales_yes[btwn_sub_sd_ind]}\n")
        print(f"Scales no between-subject SD = {btwn_sub_noise_sd_vec_scales_no[btwn_sub_sd_ind]}\n")
        plot_cor_violin(unmod_cor_rt, rtmod_cor_rt, 
                    mu_expnorm, lam_expnorm, sigma_expnorm, ISI_min, ISI_max, 
                    all_sd_params['win_sub_noise_sd']['dv_scales_yes'],
                    all_sd_params['win_sub_noise_sd']['dv_scales_no'],
                    btwn_sub_noise_sd_vec_scales_yes[btwn_sub_sd_ind],
                    btwn_sub_noise_sd_vec_scales_no[btwn_sub_sd_ind], nsub, nsim, 
                    beta['dv_scales_yes'], beta['dv_scales_no'], hp_filter,
                    fig_dir, display_plots)
    power_unmod_1sampt_output = \
        list_dicts_to_dict_tuples(power_unmod_1sampt_all)
    power_rtmod_1sampt_output = \
        list_dicts_to_dict_tuples(power_rtmod_1sampt_all)
    output = calc_group_eff_size(all_sd_params, beta)
    output['dv_scales_yes']['power_rtmod_1sampt'] =  power_rtmod_1sampt_output['dv_scales_yes']
    output['dv_scales_yes']['power_unmod_1sampt'] =  power_unmod_1sampt_output['dv_scales_yes']
    output['dv_scales_no']['power_rtmod_1sampt'] =  power_rtmod_1sampt_output['dv_scales_no']
    output['dv_scales_no']['power_unmod_1sampt'] =  power_unmod_1sampt_output['dv_scales_no']
    return output  


#I do not like the following two functions as they require the use of 
# global variables, but it won't work otherwise.  Used to generate 
# secondary x-axis in plot.

cohens_d = 1
total_to_within_sd_ratio = 1
def cohen_to_sdratio(x):
    return np.interp(-1*x, -1*cohens_d, total_to_within_sd_ratio)


def sdratio_to_cohen(x):
    return np.interp(x, total_to_within_sd_ratio, cohens_d)


def power_plot_group(power_output, fig_dir, 
                     zoom=False, show_rt_mod=False, display_plots=True):
    """
    """
    fig_dir_path = Path(fig_dir)
    if fig_dir_path.exists() == False | fig_dir_path.is_dir() == False:
        print("error: Figure directory does not exist or is not a directory, check path")
        return
    fig, axs = plt.subplots(2, sharex=True, sharey=True,
                            figsize=(9.5, 9))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.xlabel("Total SD/within-subject SD")
    plt.xlabel("Cohen's D")
    plt.ylabel("Power (group)", labelpad = 20)
    if zoom!=False:
        plt.setp(axs, xlim=zoom)
    for idx, scale_type in enumerate(["dv_scales_yes", "dv_scales_no"]):
        total_to_within_sd_ratio = power_output[scale_type]['total_within_sd_ratio']
        cohens_d = power_output[scale_type]['cohens_d']
        if scale_type == 'dv_scales_yes':
            plot_label = "Scales with RT"
        else:
            plot_label = "Doesn't scale with RT"
        axs[idx].set_title(f"Forced Choice\n{plot_label}", pad = 20)
        #secax = axs[idx].secondary_xaxis('top', functions=(cohen_to_sdratio, sdratio_to_cohen))
        #secax.set_xlabel('Total SD/within-subject SD', fontsize=10)
        #secax.tick_params(labelsize=10)
        axs[idx].grid(True)
        axs[idx].set_yticks(np.linspace(0, 1, 6))
        line1, = axs[idx].plot(
                    cohens_d, 
                    power_output[scale_type]['power_unmod_1sampt']['Impulse Duration'],
                    'tab:green', label = 'Const (Impulse*)') 
        line2,  = axs[idx].plot(
                    cohens_d, 
                    power_output[scale_type]['power_unmod_1sampt']['Stimulus Duration'], 
                    color='tab:purple', label = 'Const (Stimulus Duration*)')
        line3,  = axs[idx].plot(
                    cohens_d, 
                    power_output[scale_type]['power_unmod_1sampt']['Mean RT Duration'], 
                    color='tab:red', label = 'Const (Mean RT Duration*)')
        line4, = axs[idx].plot(
                     cohens_d, 
                     power_output[scale_type]['power_rtmod_1sampt']['RT Duration only'], 
                     'tab:blue', label = 'RT duration')
        line5,  = axs[idx].plot(
                    cohens_d, 
                    power_output[scale_type]['power_unmod_1sampt']['No RT effect'], 
                    color='tab:olive', label = 'Const (Stimulus Duration)')
        if show_rt_mod == True:
            line1b, = axs[idx].plot(
                    cohens_d, 
                    power_output[scale_type]['power_rtmod_1sampt']['Impulse Duration'],
                    'tab:green', linestyle = 'dashed',
                    label = 'RT Modulated (Impulse*)') 
            line2b,  = axs[idx].plot(
                    cohens_d, 
                    power_output[scale_type]['power_rtmod_1sampt']['Stimulus Duration'], 
                    color='tab:purple', linestyle = 'dashed',
                    label = 'RT Modulated (Stimulus Duration*)')
            line3b,  = axs[idx].plot(
                    cohens_d, 
                    power_output[scale_type]['power_rtmod_1sampt']['Mean RT Duration'], 
                    color='tab:red', linestyle = 'dashed',
                    label = 'RT Modulated (Mean RT Duration*)')
    fig.subplots_adjust(hspace=.5, bottom=0.1)
    fig.tight_layout()
    if show_rt_mod == False:
        plt.legend(handles=[line1, line2, line3, line4, line5], 
              loc='center right', bbox_to_anchor=(1.75, .5), ncol=1)
    else:
        plt.legend(handles=[line1, line1b, line2, line2b, line3, line3b,
              line4, line5], 
              loc='center right', bbox_to_anchor=(1.95, .5), ncol=1)
    if show_rt_mod == True:
        fig_file = f"{fig_dir}/group_power_with_modulated_regressors.pdf"
    else:
        fig_file = f"{fig_dir}/group_power_only_unmodulated_regressors.pdf"
    plt.savefig(fig_file,
            format='pdf', transparent=True, pad_inches=.5, bbox_inches='tight')
    if display_plots == True:
        plt.show()


def diff_power_plot_group(unmod_output_power, rtmod_output_power, total_to_within_sd_ratio, figpath, sim_type='dv_scales_yes',
                    zoom=False, display_plots=True):
    fig = plt.figure()
    axs = plt.subplot(111, frameon=False)
    loc = plticker.MultipleLocator(base=.1) # this locator puts ticks at regular intervals
    axs.yaxis.set_major_locator(loc)
    plt.xlabel("Total SD/within-subject SD")
    plt.ylabel("Power Difference")
    if zoom!=False:
        plt.setp(axs, xlim=zoom)
    if sim_type == 'dv_scales_yes':
        true_power = np.array(rtmod_output_power[sim_type]['RT Duration only'])
    else:
        true_power = np.array(unmod_output_power[sim_type]['No RT effect'])
    line1, = axs.plot(
                     total_to_within_sd_ratio, 
                     true_power - rtmod_output_power[sim_type]['RT Duration only'], 
                     'tab:blue', label = 'RT duration')
    line2, = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(unmod_output_power[sim_type]['Impulse Duration']), 
                    'tab:green', label = 'Const (impulse)') 
    line3,  = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(rtmod_output_power[sim_type]['Impulse Duration']), 
                    'tab:green', linestyle = 'dashed', label = 'RT modulated (impulse)')
    line4, = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(unmod_output_power[sim_type]['Fixed/RT Duration (orth)']), 
                    color='tab:orange', label = 'Const (stimulus duration)') 
    line5,  = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(rtmod_output_power[sim_type]['Fixed/RT Duration (orth)']), 
                    color='tab:orange',linestyle='dashed', label = 'Orthogonalized RT duration')
    line6,  = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(unmod_output_power[sim_type]['Stimulus Duration']), 
                    color='tab:purple', label = 'Const (stimulus duration)')
    line7,  = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(rtmod_output_power[sim_type]['Stimulus Duration']), 
                    color='tab:purple',linestyle='dashed', label = 'RT modulated (stimulus duration)')
    line8,  = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(unmod_output_power[sim_type]['Mean RT Duration']), 
                    color='tab:red', label = 'Const (Mean RT duration)')
    line9,  = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(rtmod_output_power[sim_type]['Mean RT Duration']), 
                    color='tab:red',linestyle='dashed', label = 'RT modulated (mean RT duration)')
    line10,  = axs.plot(
                    total_to_within_sd_ratio, 
                    true_power - np.array(unmod_output_power[sim_type]['No RT effect']), 
                    color='tab:olive', label = 'Const (stimulus duration)')
    fig.tight_layout()
    plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9, line10], 
              loc='center right', bbox_to_anchor=(2.5, 1/2), ncol=1)
    plt.savefig(figpath,
            format='pdf', transparent=True, pad_inches=.5, bbox_inches='tight')
    if display_plots == True:
        plt.show()

