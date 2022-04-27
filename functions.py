from scipy.stats import exponnorm
import numpy as np
from nilearn.glm.first_level import hemodynamic_models
from nilearn.glm.first_level.design_matrix import _cosine_drift
import statsmodels.api as sm
from copy import deepcopy
import random


def make_multi_level_empty_dict(list_keys, name_newvar, n_empty):
    if name_newvar is None:
        out_dict = np.array([np.nan]*n_empty)
    else:
        out_dict = {name_newvar: np.array([np.nan]*n_empty)}
    for current_key_set in reversed(list_keys):
        out_dict_tmp = {key: deepcopy(out_dict) for key in current_key_set}
        out_dict = deepcopy(out_dict_tmp)
    return out_dict


def make_group_names():
    stimulus_types = ['blocked', 'random']
    #model_types = ['Two stimulus types, no RT', 'Two stimulus types, RT mod',
    #               'Two stimulus types, RT dur',
    #               'Two stimulus types, 2 RT dur only']
    model_types = ['Two stimulus types, no RT', 'Two stimulus types, RT mod',
                   'Two stimulus types, 2 RT dur only']
    dv_types = ['dv_scales_yes', 'dv_scales_no']
    return stimulus_types, model_types, dv_types


def t_to_cor(t_val, nobs, nbeta):
    corest = t_val/(nobs - nbeta + t_val ** 2) ** .5
    return corest


def make_dct_basis(n_time_points):
    dct_basis = _cosine_drift(.01, np.arange(n_time_points))
    dct_basis = np.delete(dct_basis, -1, axis=1)
    return dct_basis


def make_3column_onsets(onsets, durations, amplitudes):
    """Utility function to generate 3 column onset structure
    """
    return(np.transpose(np.c_[onsets, durations, amplitudes]))


def sample_rts_2stim(n_trials, mu_expnorm=600, lam_expnorm=1 / 100,
                     sigma_expnorm=75, max_rt=2000, min_rt=0, rt_diff_s=.1):
    """
    """
    shape_expnorm = 1 / (sigma_expnorm * lam_expnorm)
    sim_num_trials = 0
    while sim_num_trials < n_trials:
        subject_specific_mu_expnorm_short = exponnorm.rvs(shape_expnorm,
                                                          mu_expnorm,
                                                          sigma_expnorm, 1) -\
                                                          1 / lam_expnorm -\
                                                          (rt_diff_s/2 * 1000)
        subject_specific_mu_expnorm_long = \
            subject_specific_mu_expnorm_short + rt_diff_s * 1000 
        rt_trials_twice_what_needed_shorter = \
            exponnorm.rvs(shape_expnorm, subject_specific_mu_expnorm_short,
                          sigma_expnorm, n_trials)
        rt_trials_filtered_shorter = rt_trials_twice_what_needed_shorter[
            np.where((rt_trials_twice_what_needed_shorter < max_rt) &
                     (rt_trials_twice_what_needed_shorter > min_rt))]
        rt_trials_twice_what_needed_longer = \
            exponnorm.rvs(shape_expnorm, subject_specific_mu_expnorm_long,
                          sigma_expnorm, n_trials)
        rt_trials_filtered_longer = \
            rt_trials_twice_what_needed_longer[
                np.where((rt_trials_twice_what_needed_longer < max_rt) &
                         (rt_trials_twice_what_needed_longer > min_rt))]
        sim_num_trials = (subject_specific_mu_expnorm_short > 0) *\
                         (rt_trials_filtered_shorter.shape[0] > 
                          int(n_trials/2)) * \
                         (rt_trials_filtered_longer.shape[0] > 
                          int(n_trials/2))*n_trials
    rt_trials_shorter = rt_trials_filtered_shorter[:int(n_trials/2)] / 1000
    rt_trials_longer = rt_trials_filtered_longer[:int(n_trials/2)] / 1000
    return rt_trials_shorter, rt_trials_longer


def make_regressors_two_trial_types(n_trials, 
                                    repetition_time=1, mu_expnorm=600,
                                    lam_expnorm=1 / 100, sigma_expnorm=75,
                                    max_rt=2000, min_rt=0, event_duration=2,
                                    ISI_min=2, ISI_max=5, center_rt=True,
                                    rt_diff_s=.1):
    """
    """
    if n_trials % 8 != 0:
        print("Error: Please number of trials divisible by 8")
        return 
    rt_trials_shorter, rt_trials_longer = \
        sample_rts_2stim(n_trials, mu_expnorm, lam_expnorm,
                         sigma_expnorm, max_rt, min_rt, rt_diff_s)
    ISI = np.random.uniform(low=ISI_min, high=ISI_max, size=n_trials - 1)
    scan_length = rt_trials_shorter.sum() + \
        rt_trials_longer.sum() + ISI.sum() + 50
    frame_times = np.arange(0, scan_length*repetition_time, repetition_time)
    half_n = int(n_trials/2)

    block_ind = np.tile([1, 1, 1, 1, 0, 0, 0, 0], int(n_trials/8))
    rt_vec_blocked = block_ind.astype(float)*0
    rt_vec_blocked[block_ind == 1] = rt_trials_shorter
    rt_vec_blocked[block_ind == 0] = rt_trials_longer
    onsets_blocked = np.cumsum(np.append([5], 
                               rt_vec_blocked[0:(n_trials-1)]+ISI))
    onsets_block1 = onsets_blocked[block_ind == 1].copy()
    onsets_block2 = onsets_blocked[block_ind == 0].copy()
    rt_block1 = rt_vec_blocked[block_ind == 1].copy()
    rt_block2 = rt_vec_blocked[block_ind == 0].copy()

    rand_vals = random.sample(range(n_trials), half_n)
    ind_set1 = np.full(n_trials, True, dtype=bool)
    ind_set1[rand_vals] = False
    rt_vec_random = ind_set1.astype(float)*0
    rt_vec_random[~ind_set1] = rt_trials_shorter
    rt_vec_random[ind_set1] = rt_trials_longer
    onsets_random = np.cumsum(np.append([5], 
                              rt_vec_blocked[0:(n_trials-1)]+ISI))
    onsets_random1 = onsets_random[~ind_set1]
    onsets_random2 = onsets_random[ind_set1]
    rt_random1 = rt_vec_random[~ind_set1].copy()
    rt_random2 = rt_vec_random[ind_set1].copy()

    if center_rt is True:
        center_val_blocked = np.mean(rt_vec_blocked)
        center_val_random = np.mean(rt_vec_random)
    else:
        center_val_blocked = 0
        center_val_random = 0

    rt_random_mn1 = np.mean(rt_random1)
    rt_random_mn2 = np.mean(rt_random2)
    rt_random_mc = rt_vec_random - center_val_random
    rt_block_mn1 = np.mean(rt_block1)
    rt_block_mn2 = np.mean(rt_block2)
    rt_block_mc = rt_vec_blocked - center_val_blocked
    fixed_event_duration = np.zeros(onsets_block2.shape) + event_duration
    modulation_half = np.ones(onsets_block1.shape)
    reg_types = ['stim1_blocked', 'stim2_blocked',
                 'stim1_random', 'stim2_random',
                 'rt_mod_2stim_blocked', 'rt_mod_2stim_random',
                 'rt_dur_2stim_blocked', 'rt_dur_2stim_random',
                 'rt_dur_2stim_blocked1', 'rt_dur_2stim_blocked2',
                 'rt_dur_2stim_random1', 'rt_dur_2stim_random2']
    col_ons = {}
    col_ons['stim1_blocked'] =\
        make_3column_onsets(onsets_block1, 
                            fixed_event_duration,
                            modulation_half)
    col_ons['stim2_blocked'] =\
        make_3column_onsets(onsets_block2,
                            fixed_event_duration,
                            modulation_half)
    col_ons['stim1_random'] =\
        make_3column_onsets(onsets_random1,
                            fixed_event_duration,
                            modulation_half)
    col_ons['stim2_random'] =\
        make_3column_onsets(onsets_random2,
                            fixed_event_duration,
                            modulation_half)
    col_ons['rt_mod_2stim_blocked'] =\
        make_3column_onsets(onsets_blocked,
                            np.zeros(onsets_blocked.shape) + event_duration,
                            rt_block_mc)
    col_ons['rt_mod_2stim_random'] =\
        make_3column_onsets(onsets_random,
                            np.zeros(onsets_random.shape) + event_duration,
                            rt_random_mc)
    col_ons['rt_dur_2stim_blocked'] =\
        make_3column_onsets(onsets_blocked,
                            rt_vec_blocked,
                            np.ones(onsets_blocked.shape))
    col_ons['rt_dur_2stim_random'] = \
        make_3column_onsets(onsets_random,
                            rt_vec_random,
                            np.ones(onsets_random.shape))
    col_ons['rt_dur_2stim_blocked1'] = \
        make_3column_onsets(onsets_block1,
                            rt_block1,
                            modulation_half)
    col_ons['rt_dur_2stim_blocked2'] = \
        make_3column_onsets(onsets_block2,
                            rt_block2,
                            modulation_half)
    col_ons['rt_dur_2stim_random1'] = \
        make_3column_onsets(onsets_random1,
                            rt_random1,
                            modulation_half)
    col_ons['rt_dur_2stim_random2'] = \
        make_3column_onsets(onsets_random2,
                            rt_random2,
                            modulation_half)
    regressors = {}
    for reg_type in reg_types:
        regressors[reg_type], _ =\
            hemodynamic_models.compute_regressor(
            col_ons[reg_type], 'spm', frame_times, oversampling=16)
    rt_means = {'block1_mean': rt_block_mn1,
                'block2_mean': rt_block_mn2,
                'random1_mean': rt_random_mn1,
                'random2_mean': rt_random_mn2}
    return regressors, rt_means

    
def make_design_matrices_2stim(regressors):
    """
    Input: regressor output from make_regressors_one_trial_type
    Output: Design matrices for models of interest in simulations
    """
    regressor_shape = regressors['stim1_blocked'].shape

    x_duration_event_duration_2stim_blocked_nort = \
        np.concatenate((np.ones(regressor_shape),
                       regressors['stim1_blocked'],
                       regressors['stim2_blocked']), axis=1)
    x_duration_event_duration_2stim_blocked_rtmod = \
        np.concatenate((np.ones(regressor_shape),
                       regressors['stim1_blocked'],
                       regressors['stim2_blocked'],
                       regressors['rt_mod_2stim_blocked']), axis=1)
    x_duration_event_duration_2stim_random_nort = \
        np.concatenate((np.ones(regressor_shape),
                       regressors['stim1_random'],
                       regressors['stim2_random']), axis=1)
    x_duration_event_duration_2stim_random_rtmod = \
        np.concatenate((np.ones(regressor_shape),
                       regressors['stim1_random'],
                       regressors['stim2_random'],
                       regressors['rt_mod_2stim_random']), axis=1)
    x_duration_event_duration_2stim_blocked_rtdur = \
        np.concatenate((np.ones(regressor_shape),
                       regressors['stim1_blocked'],
                       regressors['stim2_blocked'],
                       regressors['rt_dur_2stim_blocked']), axis=1)
    x_duration_event_duration_2stim_random_rtdur = \
        np.concatenate((np.ones(regressor_shape),
                       regressors['stim1_random'],
                       regressors['stim2_random'],
                       regressors['rt_dur_2stim_random']), axis=1)
    x_duration_rt_only_duration_2stim_blocked = \
        np.concatenate((np.ones(regressor_shape),
                       regressors['rt_dur_2stim_blocked1'],
                       regressors['rt_dur_2stim_blocked2']), axis=1)
    x_duration_rt_only_duration_2stim_random = \
        np.concatenate((np.ones(regressor_shape),
                       regressors['rt_dur_2stim_random1'],
                       regressors['rt_dur_2stim_random2']), axis=1)
    models = {}
    models['blocked'] = {'Two stimulus types, no RT':
                         x_duration_event_duration_2stim_blocked_nort,
                         'Two stimulus types, RT mod':
                         x_duration_event_duration_2stim_blocked_rtmod,
                         #'Two stimulus types, RT dur':
                         #x_duration_event_duration_2stim_blocked_rtdur,
                         'Two stimulus types, 2 RT dur only':
                         x_duration_rt_only_duration_2stim_blocked}
    models['random'] = {'Two stimulus types, no RT':
                        x_duration_event_duration_2stim_random_nort,
                        'Two stimulus types, RT mod':
                        x_duration_event_duration_2stim_random_rtmod,
                        #'Two stimulus types, RT dur':
                        #x_duration_event_duration_2stim_random_rtdur,
                        'Two stimulus types, 2 RT dur only':
                        x_duration_rt_only_duration_2stim_random}
    return models


def make_lev1_contrasts(dct_basis):
    contrasts = {'Two stimulus types, no RT': 
                 np.array([[0, -1, 1] + [0]*dct_basis.shape[1]]),
                 'Two stimulus types, RT mod': 
                 np.array([[0, -1, 1, 0] + [0]*dct_basis.shape[1]]),
                 #'Two stimulus types, RT dur': 
                 #np.array([[0, -1, 1, 0] + [0]*dct_basis.shape[1]]),
                 'Two stimulus types, 2 RT dur only':
                 np.array([[0, -1, 1] + [0]*dct_basis.shape[1]])}
    return contrasts


def make_data_scales_yes_no(beta_scales_yes, beta_scales_no, model,
                            win_sub_noise_sd):
    """
    """
    scan_length = model['Two stimulus types, 2 RT dur only'].shape[0]
    dv_scales_yes = 100 + beta_scales_yes[0] * \
        model['Two stimulus types, 2 RT dur only'][:, 1] + \
        beta_scales_yes[1] * \
        model['Two stimulus types, 2 RT dur only'][:, 2] +\
        np.random.normal(0, win_sub_noise_sd['dv_scales_yes'], (scan_length))
    dv_scales_yes = dv_scales_yes/np.mean(dv_scales_yes)
    dv_scales_no = 100 +  \
        beta_scales_no[0] * model['Two stimulus types, RT mod'][:, 1] + \
        beta_scales_no[1] * model['Two stimulus types, RT mod'][:, 2] +\
        np.random.normal(0, win_sub_noise_sd['dv_scales_no'], (scan_length))
    dv_scales_no = dv_scales_no/np.mean(dv_scales_no)
    dependent_variables = {'dv_scales_yes': dv_scales_yes,
                           'dv_scales_no': dv_scales_no}
    return dependent_variables


def sim_fit_sub_2stim(n_trials, repetition_time=1, mu_expnorm=600,
                      lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000,
                      min_rt=0, event_duration=2, ISI_min=2, ISI_max=5,
                      win_sub_noise_sd={'dv_scales_yes': .1, 
                                        'dv_scales_no': .1},
                      center_rt=True, beta_scales_yes=np.array([.1, .1]),
                      beta_scales_no=np.array([.1, .1]),
                      rt_diff_s=1):
    """ 
    """
    regressors, mean_rt = \
        make_regressors_two_trial_types(n_trials,
                                        repetition_time, mu_expnorm,
                                        lam_expnorm, sigma_expnorm,
                                        max_rt, min_rt, event_duration,
                                        ISI_min, ISI_max, center_rt, rt_diff_s)
    models = make_design_matrices_2stim(regressors)
    output = {keys: {} for keys in models}
    for stim_pres in models:
        output[stim_pres] = {key: {} for key in models[stim_pres]}
        dependent_variables =\
            make_data_scales_yes_no(beta_scales_yes, beta_scales_no, 
                                    models[stim_pres],
                                    win_sub_noise_sd)
        scan_length = \
            models[stim_pres]['Two stimulus types, 2 RT dur only'].shape[0]
        dct_basis = make_dct_basis(scan_length)
        contrasts = make_lev1_contrasts(dct_basis)
        for model_name, model_mtx in models[stim_pres].items():
            output[stim_pres][model_name] =\
                {key: {} for key in dependent_variables}
            for dependent_variable_name, dv in dependent_variables.items():
                model = np.concatenate((model_mtx, dct_basis), axis=1)
                model_setup = sm.OLS(dv, model)
                fit = model_setup.fit()
                con_test = fit.t_test(contrasts[model_name])
                output[stim_pres][model_name][dependent_variable_name] = {
                    'beta_diff_est': con_test.effect,
                    'p_beta_diff': con_test.pvalue,
                    'beta1_est': fit.params[1],
                    'p_beta1': fit.pvalues[1],
                    'beta2_est': fit.params[2],
                    'p_beta2': fit.pvalues[2]}     
    return output, mean_rt


def est_within_sub_eff_size_2stim(n_trials, repetition_time=1,
                                  mu_expnorm=600, lam_expnorm=1 / 100, 
                                  sigma_expnorm=75, max_rt=2000, min_rt=0, 
                                  event_duration=2, ISI_min=2, ISI_max=5,
                                  win_sub_noise_sd={'dv_scales_yes': .1, 
                                                    'dv_scales_no': .1},
                                  beta_scales_yes=np.array([.1, .1]),
                                  beta_scales_no=np.array([.1, .1]),
                                  center_rt=True, rt_diff_s=1, nsim=500):
    """
    """
    beta_types = ['beta1_scales_yes', 'beta2_scales_yes',
                  'beta1_scales_no', 'beta2_scales_no']
    effect_size =\
        make_multi_level_empty_dict([['blocked', 'random'], 
                                    beta_types], 'eff_size_cor', nsim)
    for sim_num in range(nsim):
        regressors, _ = \
            make_regressors_two_trial_types(n_trials,
                                            repetition_time, mu_expnorm,
                                            lam_expnorm, sigma_expnorm,
                                            max_rt, min_rt, event_duration,
                                            ISI_min, ISI_max, center_rt,
                                            rt_diff_s)
        scan_length = len(regressors['stim1_random'])
        dct_basis = make_dct_basis(scan_length)
        models = make_design_matrices_2stim(regressors)
        for stim_pres in models:
            dv_scales_yes = 100 + beta_scales_yes[0]*models[stim_pres]\
                    ['Two stimulus types, 2 RT dur only'][:,1] + \
                    beta_scales_yes[1]*models[stim_pres]\
                    ['Two stimulus types, 2 RT dur only'][:,2] +\
                    np.random.normal(0, win_sub_noise_sd['dv_scales_yes'], 
                    (scan_length))
            dv_scales_no = 100 + beta_scales_no[0]*models[stim_pres]\
                       ['Two stimulus types, RT mod'][:,1] + \
                       beta_scales_no[1]*models[stim_pres]\
                       ['Two stimulus types, RT mod'][:,2] +\
                       np.random.normal(0, win_sub_noise_sd['dv_scales_no'], 
                       (scan_length))
            desmat_scales_yes = np.concatenate((models[stim_pres]\
                    ['Two stimulus types, 2 RT dur only'], dct_basis), axis=1)
            desmat_scales_no = np.concatenate((models[stim_pres]\
                    ['Two stimulus types, no RT'], dct_basis), axis=1)
            mod_scales_yes = sm.OLS(dv_scales_yes, desmat_scales_yes)
            fit_yes = mod_scales_yes.fit()
            mod_scales_no = sm.OLS(dv_scales_no, desmat_scales_no)
            fit_no = mod_scales_no.fit()
            tvals = dict()
            tvals['beta1_scales_yes'] = fit_yes.tvalues[1]
            tvals['beta2_scales_yes'] = fit_yes.tvalues[2]
            tvals['beta1_scales_no'] = fit_no.tvalues[1]
            tvals['beta2_scales_no'] = fit_no.tvalues[2]
            num_mod_params = 3 + dct_basis.shape[1]
            for beta_type in beta_types:
                effect_size[stim_pres][beta_type]['eff_size_cor'][sim_num] = \
                    t_to_cor(tvals[beta_type], scan_length, num_mod_params)
    eff_size_out = make_multi_level_empty_dict([['blocked', 'random'], 
                     beta_types],'eff_size_cor', 1)
    for stim_pres in models:
        for beta_type in beta_types:
            eff_size_out[stim_pres][beta_type]['eff_size_cor']=\
            np.mean(effect_size[stim_pres][beta_type]['eff_size_cor'])
    return  eff_size_out


def est_var_ratio_2stim(n_trials, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, 
              win_sub_noise_sd={'dv_scales_yes': .1, 'dv_scales_no': .1},
              btwn_sub_noise_sd={'dv_scales_yes': 1, 'dv_scales_no': 1},
              center_rt=True, 
              rt_diff_s = 1, nsim= 500):
    """
    """
    stimulus_types, model_types, dv_types = make_group_names()
    des_sd_con_sim = make_multi_level_empty_dict([stimulus_types,
                model_types], 'des_sd', nsim)
    for sim_num in range(nsim):
        regressors, _ = make_regressors_two_trial_types(n_trials, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, 
                                   ISI_max, center_rt, rt_diff_s)
        scan_length = len(regressors['stim1_random'])
        dct_basis = make_dct_basis(scan_length)
        contrasts = make_lev1_contrasts(dct_basis)
        models = make_design_matrices_2stim(regressors)
        for stim_pres in models:
            for model_name, model_mtx in models[stim_pres].items():
                model = np.concatenate((model_mtx, dct_basis), axis=1)
                contrast_loop = contrasts[model_name]
                inv_xtx = np.linalg.inv(model.T.dot(model))
                des_sd_con_sim[stim_pres][model_name]['des_sd'][sim_num] = \
                                 np.sqrt(np.linalg.multi_dot([contrast_loop, \
                                 inv_xtx, contrast_loop.T]))
    sd_ratio_out_beta_diff = make_multi_level_empty_dict([dv_types, 
                 stimulus_types, model_types], 
                 'sd_total_div_sd_win_beta_diff', 1)
    for stim_pres in models:
        for model_name in models[stim_pres]:
            des_sd_con_avg_loop = np.sqrt(\
                np.mean(des_sd_con_sim[stim_pres][model_name]['des_sd']**2))  
            for dv_type_loop in dv_types:
                # I multiply the between sub var by 2 because
                # that's the variance for the difference in betas.
                mfx_sd_diff = np.sqrt((win_sub_noise_sd[dv_type_loop]*\
                                  des_sd_con_avg_loop)**2 + 
                                  2*btwn_sub_noise_sd[dv_type_loop]**2)
                total_within_sd_ratio = mfx_sd_diff/(des_sd_con_avg_loop*\
                                                 win_sub_noise_sd[dv_type_loop])
                sd_ratio_out_beta_diff[dv_type_loop][stim_pres][model_name]\
                                ['sd_total_div_sd_win_beta_diff'][0] = \
                                total_within_sd_ratio
    return sd_ratio_out_beta_diff


def est_group_cohen_d(n_trials,  repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, 
              win_sub_noise_sd={'dv_scales_yes': .1, 'dv_scales_no': .1},
              btwn_sub_noise_sd={'dv_scales_yes': 1, 'dv_scales_no': 1},
              beta_scales_yes=np.array([.1, .1]), 
              beta_scales_no=np.array([.1, .1]), 
              center_rt=True, 
              rt_diff_s = 1, nsim= 500):

    """
    """
    stim_types, _, dv_types = make_group_names()
    desmat_names = ['Two stimulus types, 2 RT dur only', 
                   'Two stimulus types, no RT']
    con_est = make_multi_level_empty_dict([stim_types, dv_types, 
                  desmat_names, ['diff', 'beta1']], 'con_est', nsim)
    for sim_num in range(nsim):
        regressors, _ = make_regressors_two_trial_types(n_trials, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, 
                                   ISI_max, center_rt, rt_diff_s)
        scan_length = len(regressors['stim1_random'])
        dct_basis = make_dct_basis(scan_length)
        models = make_design_matrices_2stim(regressors)
        for stim_pres in stim_types:
            beta_scales_yes_sub = beta_scales_yes + \
                 np.random.normal(0, 
                 btwn_sub_noise_sd['dv_scales_yes'], (1, 2))[0]
            beta_scales_no_sub = beta_scales_no + \
                 np.random.normal(0, 
                 btwn_sub_noise_sd['dv_scales_no'], (1, 2))[0]
            dependent_vars = dict()
            dependent_vars['dv_scales_yes'] = 100 +  \
                    beta_scales_yes_sub[0]*models[stim_pres]\
                    ['Two stimulus types, 2 RT dur only'][:,1] + \
                    beta_scales_yes_sub[1]*models[stim_pres]\
                    ['Two stimulus types, 2 RT dur only'][:,2] +\
                    np.random.normal(0, win_sub_noise_sd['dv_scales_yes'], 
                    (scan_length))
            dependent_vars['dv_scales_no'] = 100 + \
                       beta_scales_no_sub[0]*models[stim_pres]\
                       ['Two stimulus types, no RT'][:,1] + \
                       beta_scales_no_sub[1]*models[stim_pres]\
                       ['Two stimulus types, no RT'][:,2] +\
                       np.random.normal(0, win_sub_noise_sd['dv_scales_no'], 
                       (scan_length))
            desmats = dict()
            for desmat_name in desmat_names:
                desmats[desmat_name] = np.concatenate((models[stim_pres]\
                    [desmat_name], dct_basis), axis=1)
            for dv_key, dv in dependent_vars.items():
                for desmat_key, des in desmats.items():
                    mod_loop = sm.OLS(dv, des)
                    fit = mod_loop.fit()
                    diff_est_loop = fit.params[2]-fit.params[1]
                    con_est[stim_pres][dv_key][desmat_key]['diff']['con_est']\
                        [sim_num] = diff_est_loop
                    con_est[stim_pres][dv_key][desmat_key]['beta1']['con_est']\
                        [sim_num] = fit.params[1]
    cohens_d_vals = \
                  make_multi_level_empty_dict([stim_types, dv_types, desmats,
                  ['diff', 'beta1']],
                   'cohens_d', 1)
    for stim_type in stim_types:
        for dv_type in dv_types:
            for desmat in desmats:
                con_loop = \
                    con_est[stim_type][dv_type][desmat]['diff']['con_est']
                beta1_loop = \
                    con_est[stim_type][dv_type][desmat]['beta1']['con_est']
                cohens_d_vals[stim_type][dv_type][desmat]['diff']\
                    ['cohens_d'] = np.mean(con_loop)/np.std(con_loop)
                cohens_d_vals[stim_type][dv_type][desmat]['beta1']\
                    ['cohens_d'] = np.mean(beta1_loop)/np.std(beta1_loop)
    return cohens_d_vals


def group_2stim_rt_diff_vec(n_trials, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, 
              win_sub_noise_sd={'dv_scales_yes': .1, 'dv_scales_no': .1},
              btwn_sub_noise_sd={'dv_scales_yes': 1, 'dv_scales_no': 1},
              center_rt=True, beta_scales_yes=np.array([.1, .1]),
              beta_scales_no=np.array([.1, .1]), 
              rt_diff_s_vec=[0, .1], nsub=50, nsim=1000):
    """
    """
    stimulus_types, model_types, dv_types = make_group_names()
    num_rt_diff = len(rt_diff_s_vec)
    con_types = ['beta_diff_est']
    group_rej_rate = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                                    con_types], None , num_rt_diff)
    group_rej_rate_ci_low = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                                    con_types], None , num_rt_diff)
    group_rej_rate_ci_up = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                                    con_types], None , num_rt_diff)
    group_rtdiff_cor_avg = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                                    con_types], None , num_rt_diff)
    group_rtmn_cor_avg = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                                    con_types], None , num_rt_diff)
    for idx_rt_diff_s, rt_diff_s in enumerate(rt_diff_s_vec):
        stimulus_types, model_types, dv_types = make_group_names()
        group_p = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                        con_types], None, nsim)
        group_rtdiff_cor = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                        con_types], None, nsim)
        group_rtmn_cor = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                        con_types], None, nsim)
        for simnum in range(nsim):
            con_est_subs = make_multi_level_empty_dict([stimulus_types, model_types, 
                                dv_types, con_types], None, nsub)
            mnrt_diff_subs = {key: np.array([np.nan]*nsub) \
                          for key in stimulus_types}
            rt_avg_subs = {key: np.array([np.nan]*nsub) for key in stimulus_types}
            for subnum in range(nsub):
                beta_scales_yes_sub = beta_scales_yes + \
                    np.random.normal(0, 
                    btwn_sub_noise_sd['dv_scales_yes'], (1, 2))[0]
                beta_scales_no_sub = beta_scales_no + \
                    np.random.normal(0, 
                    btwn_sub_noise_sd['dv_scales_no'], (1, 2))[0]
                output_model, mns = sim_fit_sub_2stim(n_trials,  
                repetition_time, mu_expnorm,
                lam_expnorm, sigma_expnorm, max_rt, 
                min_rt, event_duration, ISI_min, ISI_max, 
                win_sub_noise_sd, 
                center_rt, beta_scales_yes_sub, beta_scales_no_sub, 
                rt_diff_s)
                mnrt_diff_subs['blocked'][subnum] = mns['block2_mean'] - \
                                                mns['block1_mean']
                mnrt_diff_subs['random'][subnum] = mns['random2_mean'] -\
                                               mns['random1_mean']
                rt_avg_subs['blocked'][subnum] = (mns['block2_mean'] +\
                                              mns['block1_mean'])/2
                rt_avg_subs['random'][subnum] = (mns['random2_mean'] +\
                                             mns['random1_mean'])/2
                for cur_stimulus_type in stimulus_types:
                    for cur_model_type in model_types:
                        for cur_dv_type in dv_types:
                            for con_type in con_types:
                                con_est_subs[cur_stimulus_type][cur_model_type]\
                                    [cur_dv_type][con_type][subnum] = \
                                output_model[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type]
            for cur_stimulus_type in stimulus_types:
                desmat_rtdiff = sm.add_constant(mnrt_diff_subs\
                             [cur_stimulus_type])
                desmat_mnrt = sm.add_constant(rt_avg_subs\
                             [cur_stimulus_type])
                for cur_model_type in model_types:
                    for cur_dv_type in dv_types:
                        for con_type in con_types:
                            group_dv = con_est_subs[cur_stimulus_type]\
                                [cur_model_type][cur_dv_type][con_type]
                            mod_1samp = np.ones(group_dv.shape)
                            run_1samp = sm.OLS(group_dv, mod_1samp).fit()
                            group_p[cur_stimulus_type][cur_model_type][cur_dv_type]\
                                [con_type][simnum] = run_1samp.pvalues[0]
                            run_rtdiff_cor = sm.OLS(group_dv, desmat_rtdiff).fit()
                            run_mnrt_cor = sm.OLS(group_dv, desmat_mnrt).fit()
                            group_rtdiff_cor[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type][simnum] = \
                                t_to_cor(run_rtdiff_cor.tvalues[1], nsub, 2)
                            group_rtmn_cor[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type][simnum] = \
                                t_to_cor(run_mnrt_cor.tvalues[1],nsub,2)
        for cur_stimulus_type in stimulus_types:
            for cur_model_type in model_types:
                for cur_dv_type in dv_types:
                    for con_type in con_types:
                        group_rej_rate_ci_low[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type][idx_rt_diff_s] = \
                                    np.quantile(group_p\
                                    [cur_stimulus_type][cur_model_type]\
                                        [cur_dv_type][con_type], 0.025)
                        group_rej_rate_ci_up[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type][idx_rt_diff_s] = \
                                    np.quantile(group_p\
                                    [cur_stimulus_type][cur_model_type]\
                                        [cur_dv_type][con_type], 0.975)
                        group_rej_rate[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type][idx_rt_diff_s] = np.mean(group_p\
                                    [cur_stimulus_type][cur_model_type]\
                                        [cur_dv_type][con_type] <=0.05)
                        group_rtdiff_cor_avg[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type][idx_rt_diff_s] = np.mean(group_rtdiff_cor\
                                    [cur_stimulus_type][cur_model_type]\
                                        [cur_dv_type][con_type])
                        group_rtmn_cor_avg[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type][idx_rt_diff_s] = np.mean(group_rtmn_cor\
                                    [cur_stimulus_type][cur_model_type]\
                                        [cur_dv_type][con_type])
    output = {'rt_diff' : rt_diff_s_vec,
            'group_rej_rate': group_rej_rate,
            'group_rej_rate_ci_low': group_rej_rate_ci_low,
            'group_rej_rate_ci_up': group_rej_rate_ci_up,
            'group_rtdiff_cor_avg': group_rtdiff_cor_avg,
            'group_rtmn_cor_avg': group_rtmn_cor_avg}
    return output



def group_2stim_beta2_vec(n_trials, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, 
              win_sub_noise_sd={'dv_scales_yes': .1, 'dv_scales_no': .1},
              btwn_sub_noise_sd={'dv_scales_yes': 1, 'dv_scales_no': 1},
              center_rt=True, beta_scales_yes={'beta1': 1, 'beta2': [1, 1.5]},
              beta_scales_no={'beta1': 1, 'beta2': [1, 1.5]}, 
              rt_diff_s=0.1, nsub=50, nsim=1000):
    """
    """
    if len(beta_scales_yes['beta2']) != len(beta_scales_no['beta2']):
        print("beta_2 vectors must have same lengths")
        return
    stimulus_types, model_types, dv_types = make_group_names()
    num_beta2 = len(beta_scales_yes['beta2'])
    con_types = ['beta_diff_est', 'beta2_est']
    group_rej_rate = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                                    con_types], None , num_beta2)
    for idx_beta2 in range(num_beta2):
        stimulus_types, model_types, dv_types = make_group_names()
        group_p = make_multi_level_empty_dict([stimulus_types, model_types, dv_types,
                        con_types], None, nsim)
        for simnum in range(nsim):
            con_est_subs = make_multi_level_empty_dict([stimulus_types, model_types, 
                                dv_types, con_types], None, nsub)
            for subnum in range(nsub):
                beta_scales_yes_loop = np.array([beta_scales_yes['beta1'],
                        beta_scales_yes['beta2'][idx_beta2]])
                beta_scales_yes_sub = beta_scales_yes_loop + \
                    np.random.normal(0, 
                    btwn_sub_noise_sd['dv_scales_yes'], (1, 2))[0]
                beta_scales_no_loop = np.array([beta_scales_no['beta1'],
                        beta_scales_no['beta2'][idx_beta2]])
                beta_scales_no_sub = beta_scales_no_loop + \
                    np.random.normal(0, 
                    btwn_sub_noise_sd['dv_scales_no'], (1, 2))[0]
                output_model, mns = sim_fit_sub_2stim(n_trials, 
                repetition_time, mu_expnorm,
                lam_expnorm, sigma_expnorm, max_rt, 
                min_rt, event_duration, ISI_min, ISI_max, 
                win_sub_noise_sd, 
                center_rt, beta_scales_yes_sub, beta_scales_no_sub, 
                rt_diff_s)
                for cur_stimulus_type in stimulus_types:
                    for cur_model_type in model_types:
                        for cur_dv_type in dv_types:
                            for con_type in con_types:
                                con_est_subs[cur_stimulus_type][cur_model_type]\
                                    [cur_dv_type][con_type][subnum] = \
                                output_model[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type]
            for cur_stimulus_type in stimulus_types:
                for cur_model_type in model_types:
                    for cur_dv_type in dv_types:
                        for con_type in con_types:
                            group_dv = con_est_subs[cur_stimulus_type]\
                                [cur_model_type][cur_dv_type][con_type]
                            mod_1samp = np.ones(group_dv.shape)
                            run_1samp = sm.OLS(group_dv, mod_1samp).fit()
                            group_p[cur_stimulus_type][cur_model_type][cur_dv_type]\
                                [con_type][simnum] = run_1samp.pvalues[0]
        for cur_stimulus_type in stimulus_types:
            for cur_model_type in model_types:
                for cur_dv_type in dv_types:
                    for con_type in con_types:
                        group_rej_rate[cur_stimulus_type][cur_model_type]\
                                [cur_dv_type][con_type][idx_beta2] = np.mean(group_p\
                                    [cur_stimulus_type][cur_model_type]\
                                        [cur_dv_type][con_type] <=0.05)
    output = {'group_rej_rate': group_rej_rate}
    return output


