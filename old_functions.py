
def est_true_eff_size(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max = 5, 
              win_sub_noise_sd_scales_yes=0.1, win_sub_noise_sd_scales_no=0.1,
              center_rt=True,
              beta_scales_yes = 1, beta_scales_no = 1):
    """STUFF
    """
    regressors, mean_rt = make_regressors_one_trial_type(n_trials, scan_length, 
                                   repetition_time, mu_expnorm, 
                                   lam_expnorm, sigma_expnorm,
                                   max_rt, min_rt, event_duration, ISI_min, ISI_max, center_rt)
    dv_scales_yes = beta_scales_yes*regressors['rt']['unmodulated'] + \
                     np.random.normal(0, win_sub_noise_sd_scales_yes, (scan_length, 1))
    dv_scales_no = beta_scales_no*regressors['fixed_zero']['unmodulated']+ \
                     np.random.normal(0, win_sub_noise_sd_scales_no, (scan_length, 1))
    eff_size_theory_scales_yes = beta_scales_yes*np.var(regressors['rt']['unmodulated'].T) / \
                                 np.sqrt(np.var(regressors['rt']['unmodulated'].T)*
                                 (beta_scales_yes**2 * np.var(regressors['rt']['unmodulated'].T) + win_sub_noise_sd_scales_yes**2))
    eff_size_theory_scales_no = beta_scales_no*np.var(regressors['fixed_event_duration']['unmodulated'].T) / \
                                 np.sqrt(np.var(regressors['fixed_event_duration']['unmodulated'].T)*
                                 (beta_scales_no**2 * np.var(regressors['fixed_event_duration']['unmodulated'].T) + win_sub_noise_sd_scales_no**2))                                
    eff_size_dv_scales_yes = np.corrcoef(regressors['rt']['unmodulated'].T, dv_scales_yes.T)[1,0]
    eff_size_dv_scales_no = np.corrcoef(regressors['fixed_zero']['unmodulated'].T, dv_scales_no.T)[1,0]
    x_duration_rt_only = np.concatenate((np.ones(dv_scales_yes.shape),
                    regressors['rt']['unmodulated']), axis=1)
    contrasts = np.array([[0, 1]])
    _, t_dv_scales_yes, _, _, n_basis, _ = runreg(dv_scales_yes, x_duration_rt_only, contrasts, hp_filter=True, compute_stats=True)
    eff_size_dv_scales_yes_filtered = t_dv_scales_yes/(scan_length - (2 + n_basis) + t_dv_scales_yes ** 2) ** .5
    return eff_size_theory_scales_yes, eff_size_theory_scales_no, eff_size_dv_scales_yes, eff_size_dv_scales_no, eff_size_dv_scales_yes_filtered




def lev1_many_subs(n_trials, scan_length, repetition_time=1, mu_expnorm=600,
              lam_expnorm=1 / 100, sigma_expnorm=75, max_rt=2000, 
              min_rt=0, event_duration=2, ISI_min=2, ISI_max=5, 
              win_sub_noise_sd_scales_yes=0.1, win_sub_noise_sd_scales_no=0.1, 
              btwn_sub_noise_sd_scales_yes=1, btwn_sub_noise_sd_scales_no=1,
              nsub = 50, center_rt=True, beta_scales_yes = 1,
              beta_scales_no = 1,  hp_filter=True):    
    """
    Stuff
    """
    unmod_beta_est_all_list_dict = list()
    rtmod_beta_est_all_list_dict = list()
    r2_list_dict = list()
    mean_rt_all = list()


    for sub in range(0, nsub):
        beta_scales_yes_sub = beta_scales_yes + \
                    np.random.normal(0, btwn_sub_noise_sd_scales_yes, (1, 1))
        beta_scales_no_sub = beta_scales_no + \
                    np.random.normal(0, btwn_sub_noise_sd_scales_no, (1, 1))
        unmod_beta_est_loop, rtmod_beta_est_loop, _, _, mean_rt, r2 = \
            sim_fit_sub(n_trials, scan_length, repetition_time, mu_expnorm, 
            lam_expnorm, sigma_expnorm, max_rt, min_rt, event_duration, 
            ISI_min, ISI_max, win_sub_noise_sd_scales_yes, 
            win_sub_noise_sd_scales_no, center_rt, beta_scales_yes_sub, 
            beta_scales_no_sub, hp_filter)
        unmod_beta_est_all_list_dict.append(unmod_beta_est_loop)
        rtmod_beta_est_all_list_dict.append(rtmod_beta_est_loop)
        mean_rt_all.append(mean_rt)
        r2_list_dict.append(r2)

    unmod_beta_est_all_dict = \
        list_dicts_to_dict_tuples(unmod_beta_est_all_list_dict)
    rtmod_beta_est_all_dict = \
        list_dicts_to_dict_tuples(rtmod_beta_est_all_list_dict)
    r2_dict = list_dicts_to_dict_tuples(r2_list_dict) 
    dat_out = {'unmod_beta_est': unmod_beta_est_all_dict, 
               'rtmod_beta_est': rtmod_beta_est_all_dict,
               'r2': r2_dict,
               'rt_mean': mean_rt_all}
    return dat_out



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



def group_power(unmod_1sampt_p, rtmod_1sampt_p):
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
