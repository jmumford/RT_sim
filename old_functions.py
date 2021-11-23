
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
