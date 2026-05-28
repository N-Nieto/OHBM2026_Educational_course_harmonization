# --------------------- Placeholder helper functions ---------------------
"""
Translated from MATLAB code used in preprint: 

Tackling Bias in Cortical Thickness Estimation in UK Biobank Using Harmonisation Approaches
Jacob Turnbull, Gaurav Bhalerao, Rach Dawson, Frederik Lange, Fidel Alfaro-Almagro, Stephen Smith, Ludovica Griffanti
bioRxiv 2026.05.22.726536; doi: https://doi.org/10.64898/2026.05.22.726536


"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import math
import re
import sys
import warnings

import numpy as np
import numpy.linalg as la
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy.stats import chi2, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# --------------------- Placeholder helper functions ---------------------
# Translated from MATLAB, need to have concistency checked with NeuroComBat
def aprior(delta_hat):
    """Calculate the aprior parameter for the inverse gamma distribution based on the method of moments."""
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    """Calculate the bprior parameter for the inverse gamma distribution based on the method of moments."""
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    """Calculate the posterior mean for the batch effect parameters."""
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    """Calculate the posterior variance for the batch effect parameters."""
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def itSol(sdat_batch, gamma_hat, delta_hat, gamma_bar, t2, a, b,
          conv=0.001, return_hist=False):
    """
    Iteratively solve for posterior mean and variance of batch-effect parameters.

    If return_hist=True, also returns:
        count : number of iterations
        hist  : dictionary storing EB values at each iteration
    """
    import numpy as np

    g_old = gamma_hat.copy()
    d_old = delta_hat.copy()
    change = 1
    count = 0
    n = sdat_batch.shape[1]

    hist = {
        "iter": [],
        "g": [],
        "d": [],
        "sum2": [],
        "delta_hat": [],
        "change": []
    }

    while change > conv:
        g_new = postmean(gamma_hat, gamma_bar, n, d_old, t2)

        sum2 = np.sum((sdat_batch - g_new[:, None]) ** 2, axis=1)

        d_new = postvar(sum2, n, a, b)

        change = max(
            np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), np.finfo(float).eps)),
            np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), np.finfo(float).eps))
        )

        count += 1

        hist["iter"].append(count)
        hist["g"].append(g_new.copy())
        hist["d"].append(d_new.copy())
        hist["sum2"].append(sum2.copy())

        if n > 1:
            hist["delta_hat"].append(sum2 / (n - 1))
        else:
            hist["delta_hat"].append(np.full_like(sum2, np.nan))

        hist["change"].append(change)

        g_old = g_new
        d_old = d_new

        if count > 100:
            print("Warning: itSol did not converge after 100 iterations")
            break

    adjust = np.vstack([g_new, d_new])

    if return_hist:
        return adjust, count, hist

    return adjust

def adjust_nums(numerical_covariates, drop_idxs):
    # if we dropped some values, have to adjust those with a larger index.
    if numerical_covariates is None: return drop_idxs
    return [nc - sum(nc < di for di in drop_idxs) for nc in numerical_covariates]

# ----------------------------- Main functions -----------------------------
# Define ComBat harmonisation function
def combat(data, batch, mod=[], parametric=True,
           DeltaCorrection=True, UseEB=True, ReferenceBatch=None,
           RegressCovariates=False, GammaCorrection=True, covbat_mode=False, return_priors=False):
    """
    Run ComBat harmonisation on the data and return the harmonized data.

    This version accepts numpy arrays or pandas DataFrame/Series for data, batch, and mod.
    If a DataFrame is supplied, columns are treated as samples (so data.shape == (n_features, n_samples)).
    The function will auto-transpose data or mod if it detects that samples were provided as rows.
    The returned bayesdata is the same type as input data (DataFrame -> DataFrame, ndarray -> ndarray).

    Note: helper functions aprior, bprior, itSol must be defined in scope.

    Args:
        data (np.array or pd.DataFrame): The data matrix to be harmonized, with shape (n_features, n_samples).
        batch (np.array or pd.Series): A vector of batch labels for each sample, with length n_samples.
        mod (np.array or pd.DataFrame, optional): An optional design matrix of covariates to adjust for, with shape (n_samples, n_covariates).
        parametric (bool, optional): Whether to use parametric adjustments. Default is True.
        DeltaCorrection (bool, optional): Whether to apply delta (scale) correction. Default is True.
        UseEB (bool, optional): Whether to use empirical Bayes adjustments. Default is True.
        ReferenceBatch (str or int, optional): If provided, the name or index of the reference batch to use for fitting priors. Default is None (no reference).
        RegressCovariates (bool, optional): Whether to regress out covariate effects before harmonisation. Default is False.
        GammaCorrection (bool, optional): Whether to apply gamma (mean) correction. Default is True.
        covbat_mode (bool, optional): Whether to run in CovBat mode which includes additional covariance correction steps. Default is False.
        return_priors (bool, optional): Whether to return the estimated parameters from the ComBat model along with the harmonized data. Default is False.

    Returns:
        bayesdata (np.array or pd.DataFrame): The harmonized data, in the same format as the input data.
        priors (dict, optional): A dictionary containing the estimated parameters from the ComBat model, including:
            - gamma_hat: raw batch effect mean estimates (n_batch, n_features)
            - delta_hat: raw batch effect variance estimates (n_batch, n_features)
            - gamma_star: empirical Bayes adjusted batch effect means (n_batch, n_features)
            - delta_star: empirical Bayes adjusted batch effect variances (n_batch, n_features)
            - gamma_bar: mean of gamma_hat across batches (n_batch,)
            - t2: variance of gamma_hat across batches (n_batch,)
            - a_prior: aprior parameters for each batch (n_batch,)
            - b_prior: bprior parameters for each batch (n_batch,)  
 
    Note:
    If using this version of ComBat, please cite:

    Jean-Philippe Fortin, Drew Parker, Birkan Tunc, Takanori Watanabe, Mark A Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur, Raquel E Gur, Robert T Schultz, Ragini Verma, Russell T Shinohara. Harmonisation Of Multi-Site Diffusion Tensor Imaging Data. NeuroImage, 161, 149-170, 2017
    Jean-Philippe Fortin, Nicholas Cullen, Yvette I. Sheline, Warren D. Taylor, Irem Aselcioglu, Philip A. Cook, Phil Adams, Crystal Cooper, Maurizio Fava, Patrick J. McGrath, Melvin McInnis, Mary L. Phillips, Madhukar H. Trivedi, Myrna M. Weissman, Russell T. Shinohara. Harmonisation of cortical thickness measurements across scanners and sites. NeuroImage, 167, 104-120, 2018
    W. Evan Johnson and Cheng Li, Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8(1):118-127, 2007.

    """
    import pandas as pd
    import numpy as np
    
    # Remember whether inputs were pandas objects so we can restore types/labels on output
    dat_was_df = isinstance(data, pd.DataFrame)
    batch_was_series = isinstance(batch, (pd.Series, pd.Index))
    mod_was_df = isinstance(mod, pd.DataFrame)

    # Keep original labels (if any) to restore later
    dat_orig_index = data.index if dat_was_df else None
    dat_orig_columns = data.columns if dat_was_df else None
    batch_index = batch.index if batch_was_series else None
    mod_orig_index = mod.index if mod_was_df else None
    mod_orig_columns = mod.columns if mod_was_df else None

    # Convert pandas -> numpy, but allow transposing if the user supplied samples as rows
    # For data: desired internal shape = (n_features, n_samples) (rows=features, cols=samples)
    if dat_was_df:
        dat_np = data.values.astype(float)
        # if batch length matches number of rows, assume user gave samples as rows and transpose
        len_batch = len(batch)
        if len_batch == dat_np.shape[0] and len_batch != dat_np.shape[1]:
            dat_np = dat_np.T
            dat_transposed = True
        else:
            dat_transposed = False
    else:
        dat_np = np.asarray(data, dtype=float)
        dat_transposed = False

    # Normalize batch into 1D numpy array
    if batch_was_series:
        batch_np = batch.values
    else:
        batch_np = np.asarray(batch)
    batch_np = batch_np.ravel()

    # If dat_np and batch lengths mismatch, try to detect transposed data (samples as rows)
    if dat_np.ndim != 2:
        raise ValueError('Data matrix "data" must be 2-dimensional (features x samples).')

    # If batch length matches rows instead of columns, transpose dat_np
    if batch_np.shape[0] == dat_np.shape[0] and batch_np.shape[0] != dat_np.shape[1]:
        dat_np = dat_np.T
        dat_transposed = not dat_transposed  # flip if we already flipped earlier

    # Now dat_np shape[1] should equal batch length
    if dat_np.shape[1] != batch_np.shape[0]:
        raise ValueError('Number of samples in "data" must match length of "batch" vector.')

    # Handle mod (design covariates). Desired internal shape: (n_samples, n_covariates)
    if mod is None:
        mod_np = None
    else:
        if mod_was_df:
            mod_np = mod.values.astype(float)
            # If mod rows equal n_samples -> OK; else if mod.columns equal n_samples -> transpose
            n_samples = dat_np.shape[1]
            if mod_np.shape[0] == n_samples:
                pass
            elif mod_np.shape[1] == n_samples:
                mod_np = mod_np.T
            else:
                raise ValueError('Design matrix "mod" shape not compatible with data samples.')
        else:
            mod_np = np.asarray(mod, dtype=float)
            if mod_np.ndim == 1:
                # single covariate vector
                if mod_np.shape[0] == dat_np.shape[1]:
                    mod_np = mod_np.reshape(-1, 1)
                elif mod_np.shape[0] == dat_np.shape[0]:
                    # maybe passed per-feature by mistake
                    mod_np = mod_np.reshape(1, -1).T
                else:
                    raise ValueError('Design matrix "mod" length is incompatible with number of samples.')
            else:
                # 2D array: check orientation
                if mod_np.shape[0] != dat_np.shape[1] and mod_np.shape[1] == dat_np.shape[1]:
                    # mod provided as (n_covariates x n_samples) -> transpose
                    mod_np = mod_np.T
                elif mod_np.shape[0] != dat_np.shape[1] and mod_np.shape[1] != dat_np.shape[1]:
                    raise ValueError('Design matrix "mod" rows must match number of samples in "data".')

    # Use these working arrays from now on
    data = dat_np
    batch = batch_np
    mod = mod_np

    # Check the given parameters and print status messages
    if ReferenceBatch is None:
        print('Reference batch not given, defaulting to no reference')
    else:
        print(f'ReferenceBatch = {ReferenceBatch} -- fitting prior estimates using this batch and leaving batch unchanged')

    if not UseEB:
        print('Empirical Bayes set to false, using first estimates from raw mean and variances')
    else:
        print('Empirical Bayes set to true')

    if RegressCovariates:
        print('Regress Covariates set to true, skipping re-addition of OLS covariate estimates ')

    if not DeltaCorrection:
        print('Delta correction set to False, applying no delta (scale) correction on data')

    if not GammaCorrection:
        print('Gamma correction set to False, applying no gamma (mean) correction on data')

    # Basic input validation (after conversions)
    if data.ndim != 2:
        raise ValueError('Data matrix "data" must be 2-dimensional (features x samples).')
    if batch.ndim != 1:
        raise ValueError('Batch vector "batch" must be 1-dimensional (samples,).')
    if data.shape[1] != batch.shape[0]:
        raise ValueError('Number of samples in "data" must match length of "batch" vector.')
    if mod is not None:
        if mod.ndim != 2:
            raise ValueError('Design matrix "mod" must be 2-dimensional (samples x covariates).')
        if mod.shape[0] != data.shape[1]:
            raise ValueError('Number of samples in "data" must match number of rows in "mod" design matrix.')

    # --------------------- Begin ComBat core logic ---------------------

    # Compute SDs across samples for each feature (row)
    sds = np.std(data, axis=1, ddof=1)
    wh = np.where(sds == 0)[0]
    if wh.size > 0:
        raise ValueError('Error. There are rows with constant values across samples. Remove these rows and rerun ComBat.')

    # Convert batch vector to categorical and create dummy variables
    batch_cat = pd.Categorical(batch)
    batchmod = pd.get_dummies(batch_cat, drop_first=False).values  # shape (n_samples, n_batch)

    # Number of batches
    n_batch = batchmod.shape[1]
    levels = np.array(batch_cat.categories)
    print(f'[combat] Found {n_batch} batches')

    # Create list of arrays each containing sample indices for a batch
    batches = [np.where(batch == lev)[0] for lev in levels]

    # Size of each batch and total number of samples
    n_batches = np.array([len(b) for b in batches])
    n_array = np.sum(n_batches)

    # Construct design matrix including batch and additional covariates (mod)
    if mod is None:
        mod_arr = np.zeros((data.shape[1], 0))
    else:
        mod_arr = np.asarray(mod, dtype=float)
        if mod_arr.ndim == 1:
            mod_arr = mod_arr.reshape(-1, 1)

    design = np.hstack([batchmod, mod_arr])  # shape (n_samples, n_batch + n_cov)

    # Remove intercept column if present
    intercept = np.ones((n_array, 1))
    cols_to_keep = []
    for j in range(design.shape[1]):
        if not np.allclose(design[:, j], intercept.ravel()):
            cols_to_keep.append(j)
    design = design[:, cols_to_keep]

    print(f'[combat] Adjusting for {design.shape[1] - n_batch} covariate(s) of covariate level(s)')

    # Check for confounding between batch and covariates
    if np.linalg.matrix_rank(design) < design.shape[1]:
        nn = design.shape[1]
        if nn == (n_batch + 1):
            raise ValueError('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
        if nn > (n_batch + 1):
            temp = design[:, (n_batch):nn]
            if np.linalg.matrix_rank(temp) < temp.shape[1]:
                raise ValueError('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
            else:
                raise ValueError('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')

    print('[combat] Standardizing Data across features')

    # Estimate coefficients B_hat using least squares: B_hat = inv(design' * design) * design' * data'
    XtX = design.T @ design
    inv_XtX = np.linalg.pinv(XtX) # Find the pseudo-inverse in case XtX is singular
    B_hat = inv_XtX @ design.T @ data.T  # shape (k, n_features) 

    # Reference batch handling
    # Storage for EB iteration history
    if ReferenceBatch is not None:
        try:
            ref_idx = int(np.where(levels == ReferenceBatch)[0][0])
        except Exception:
            raise ValueError('ReferenceBatch not found in batch levels.')

        ref_samples = batches[ref_idx]
        ref_batch_effect = B_hat[ref_idx, :]

        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            Cov_effects = (tmp @ B_hat).T
        else:
            Cov_effects = np.zeros((data.shape[0], data.shape[1]))

        design_ref = design[ref_samples, :]
        predicted_ref = (design_ref @ B_hat).T
        residuals_ref = data[:, ref_samples] - predicted_ref
        var_ref = np.mean(residuals_ref ** 2, axis=1)

        stand_mean = np.tile(ref_batch_effect[:, None], (1, n_array))
        stand_mean = stand_mean + Cov_effects
        var_pooled = var_ref.copy()
        print(f'The size of the var_pooled array is {var_pooled.shape}')
    else:
        n_features = data.shape[0]
        n_samples = data.shape[1]
        XtX = design.T @ design
        inv_XtX = np.linalg.pinv(XtX)
        B_hat = inv_XtX @ design.T @ data.T
        grand_mean = (n_batches / n_array) @ B_hat[0:n_batch, :]
        predicted = (design @ B_hat).T
        resid = data - predicted
        var_pooled = np.mean(resid ** 2, axis=1)
        if np.any(var_pooled == 0):
            nonzeros = var_pooled[var_pooled != 0]
            if nonzeros.size > 0:
                var_pooled[var_pooled == 0] = np.median(nonzeros)
            else:
                var_pooled[var_pooled == 0] = 1e-6

        stand_mean = np.tile(grand_mean[:, None], (1, n_array))
        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            stand_mean = stand_mean + (tmp @ B_hat).T

    # Optional: regress covariates
    if design.shape[1] > n_batch:
        X_cov = design[:, n_batch:]
        X_cov = X_cov - np.mean(X_cov, axis=0, keepdims=True)
        B_cov = B_hat[n_batch:, :]
        Cov_effects = (X_cov @ B_cov).T
    else:
        Cov_effects = np.zeros_like(data)

    # Standardize the data, adding in small constant to avoid division by zero
    s_data = (data - stand_mean) / (np.sqrt(var_pooled)[:, None] + 1e-8)

    # Estimate batch effect parameters using least squares
    print('[combat] Fitting L/S model and finding priors')
    batch_design = design[:, :n_batch]  # samples x n_batch
    XtX_b = batch_design.T @ batch_design
    inv_XtX_b = np.linalg.pinv(XtX_b)
    gamma_hat = inv_XtX_b @ batch_design.T @ s_data.T  # shape (n_batch, n_features)
    print(f'Size of gamma hat: {gamma_hat.shape}')

    # Estimate batch-specific variances
    delta_hat = np.zeros((n_batch, data.shape[0]))
    for i in range(n_batch):
        indices = batches[i]
        if len(indices) > 1:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=1)
        else:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=0) + 1e-6

    print(f'Size of delta hat: {delta_hat.shape}')

    # Compute hyperparameters
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    t2[t2 == 0] = 1e-6

    a_prior = np.zeros(n_batch)
    b_prior = np.zeros(n_batch)
    for i in range(n_batch):
        a_prior[i] = aprior(delta_hat[i, :])
        b_prior[i] = bprior(delta_hat[i, :])

    # Apply empirical Bayes estimates (parametric)
    # Storage for EB iteration history
    eb_hist = {
        "by_batch": {},
        "counts": {},
        "levels": levels.copy()
    }

    # Apply empirical Bayes estimates (parametric)
    if parametric:
        print('[combat] Finding parametric adjustments')
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        for i in range(n_batch):
            indices = batches[i]
            if len(indices) == 0:
                continue
            temp, count, hist = itSol(
                s_data[:, indices],
                gamma_hat[i, :],
                delta_hat[i, :],
                gamma_bar[i],
                t2[i],
                a_prior[i],
                b_prior[i],
                conv=0.001,
                return_hist=True
            )

            gamma_star[i, :] = temp[0, :]
            delta_star[i, :] = temp[1, :]

            batch_label = levels[i]
            eb_hist["by_batch"][batch_label] = hist
            eb_hist["counts"][batch_label] = count

        if ReferenceBatch is not None:
            gamma_star[ref_idx, :] = np.zeros(data.shape[0])
            delta_star[ref_idx, :] = np.ones(data.shape[0])
    else:
        gamma_star = gamma_hat.copy()
        delta_star = delta_hat.copy()

    print('Size of gamma_star:', gamma_star.shape)
    bayesdata = s_data.copy()


    if not UseEB:
        print('Discounting the EB adjustments and using Raw estimates, this is not advised')
        delta_star = delta_hat.copy()
        gamma_star = gamma_hat.copy()
    if DeltaCorrection:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None]) / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
        else:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = bayesdata[:, indices] / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
    else:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None])
        else:
            print('Warning: Both Gamma and delta have been set to false, no ComBat adjustments have been applied')
    if covbat_mode:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # --- assume these are provided:
        # bayesdata : numpy array shape (n_features, n_samples)
        # batch : whatever your function needs
        # stand_mean : either shape (n_features,) or (n_features, 1)
        # data (optional) : original pandas DataFrame if you want to keep sample names
        # combat_for_covbat : your function (may accept numpy or pandas)

        # Optional: keep sample names in an array if you still want them
        # If you don't have `data`, just omit this.
        try:
            sample_names = np.asarray(data.columns)
        except Exception:
            sample_names = None

        print('[covbat] Adjusting the Data')

        # CovBat adjustment via PCA
        comdata = bayesdata.T                      # shape: (n_samples, n_features)
        bmu = np.mean(comdata, axis=0)             # mean across samples -> shape (n_features,)

        # standardize data before PCA
        scaler = StandardScaler()
        comdata_std = scaler.fit_transform(comdata)  # (n_samples, n_features)

        pca = PCA()
        pca.fit(comdata_std)
        pc_comp = pca.components_                    # (n_components, n_features)

        # full_scores as numpy array: shape (n_components, n_samples)
        full_scores = pca.transform(comdata_std).T   # pca.transform -> (n_samples, n_components) -> .T -> (n_components, n_samples)

        # Hard code pct_var for now:
        pct_var = 0.95
        var_exp = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
        # npc: number of PCs needed to exceed pct_var
        npc = int(np.min(np.where(var_exp > pct_var))) + 1

        # slice the scores to the first npc components
        scores = full_scores[:npc, :]               # shape (npc, n_samples)

        # If combat_for_covbat accepts numpy arrays, call directly:
        
        scores_com = combat(scores, batch, mod=None, parametric=True,UseEB=False)
        
            # If it expects a pandas DataFrame, convert temporarily:
        import pandas as pd
        # Output is a numpy array:
        
        full_scores[:npc, :] = scores_com

        # prepare output array (same shape as bayesdata)
        x_covbat = np.zeros_like(bayesdata)         # shape (n_features, n_samples)

        # project back to original space
        # full_scores.T shape (n_samples, n_components)
        # pc_comp shape (n_components, n_features)
        # np.dot -> (n_samples, n_features) -> .T -> (n_features, n_samples)
        proj = np.dot(full_scores.T, pc_comp).T     # shape (n_features, n_samples)

        # inverse transform the standardization
        # scaler was fit on comdata (n_samples, n_features) so we need to pass proj.T (n_samples, n_features)
        x_recon = scaler.inverse_transform(proj.T).T  # back to shape (n_features, n_samples)

        # add reconstructed signal and the stored mean (stand_mean)
        x_covbat += x_recon

        # ensure stand_mean broadcasts across columns: make it (n_features, 1) if it's (n_features,)
        """stand_mean_arr = np.asarray(stand_mean)
        if stand_mean_arr.ndim == 1:
            stand_mean_arr = stand_mean_arr.reshape(-1, 1)   # (n_features, 1)

        x_covbat += stand_mean_arr    # broadcasting across columns
        """

        # final output
        bayesdata = x_covbat.copy()
        print('[covbat] Finished CovBat adjustment')

    # Transform data back to original scale
    if RegressCovariates:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + (stand_mean - Cov_effects)
    else:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + stand_mean
    
    # Flip bayes data back if we transposed at the start
    if dat_transposed:
        bayesdata = bayesdata.T
    if return_priors:
        priors = {
            "levels": levels,
            "gamma_bar": gamma_bar,
            "t2": t2,
            "a_prior": a_prior,
            "b_prior": b_prior,
            "gamma_hat": gamma_hat,
            "delta_hat": delta_hat,
            "gamma_star": gamma_star,
            "delta_star": delta_star,
            "num_iter": eb_hist["counts"],
            "hist": eb_hist
        }

        output = {
            "bayesdata": bayesdata,
            "B_hat": B_hat,
            "priors": priors,

            # Optional flat copies for backwards compatibility
            "gamma_bar": gamma_bar,
            "t2": t2,
            "a_prior": a_prior,
            "b_prior": b_prior,
            "delta_hat": delta_hat,
            "gamma_hat": gamma_hat,
            "delta_star": delta_star,
            "gamma_star": gamma_star,
            "hist": eb_hist
        }

        return output
    else:
        return bayesdata
