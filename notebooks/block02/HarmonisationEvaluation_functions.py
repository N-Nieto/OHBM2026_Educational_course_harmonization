import numpy as np
import pandas as pd
from typing import Sequence,Optional
from scipy.stats import spearmanr,chi2,fligner
from numpy.random import default_rng
from typing import Tuple, Dict, Any, List, Iterable
import warnings
import statsmodels.formula.api as smf


def simulate_harmonisation_data(
    n_subjects=15,
    n_timepoints=2,
    n_features=2,
    seed=1,
    subject_mean=1500,
    subject_sd=80,
    noise_sd=20,
):
    np.random.seed(seed)

    # labels
    subjects = np.arange(1, n_subjects + 1)
    timepoints = [f"TP{i}" for i in range(n_timepoints)]
    sites = [f"Site_{chr(65+i)}" for i in range(n_timepoints)]  # Site_A, Site_B, ...

    feature_names = [f"Region_{i+1}" for i in range(n_features)]

    # subject-specific true values for each feature
    feature_means = np.linspace(subject_mean, subject_mean + 1000, n_features)
    true_values = {
        feat: np.random.normal(loc=feature_means[i], scale=subject_sd, size=n_subjects)
        for i, feat in enumerate(feature_names)
    }

    # site-specific bias for each feature
    # each site gets a different shift per feature
    site_bias = {
        site: np.random.normal(loc=0, scale=50, size=n_features)
        for site in sites
    }

    rows = []
    for subj_idx, subj in enumerate(subjects):
        for tp_idx, tp in enumerate(timepoints):
            site = sites[tp_idx]   # one site per timepoint
            row = {
                "Subject": subj,
                "Timepoint": tp,
                "Site": site,
            }

            for feat_idx, feat in enumerate(feature_names):
                value = (
                    true_values[feat][subj_idx] +
                    site_bias[site][feat_idx] +
                    np.random.normal(0, noise_sd)
                )
                row[feat] = round(value, 1)

            rows.append(row)

    df = pd.DataFrame(rows)
    return df

##########################################################################################################
# FOR LONGITUDINAL DATA: 2) Within subject variability: % difference (2 timepoints) OR CoV (>2 timepoints)
##########################################################################################################

def WithinSubjVar_long(
    idp_matrix: np.ndarray,
    subjects: Sequence,
    timepoints: Sequence,
    idp_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:

    if not isinstance(idp_matrix, np.ndarray):
        idp_matrix = np.asarray(idp_matrix, dtype=float)
    if idp_matrix.ndim != 2:
        raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")
    n_samples, n_idps = idp_matrix.shape

    if len(subjects) != n_samples:
        raise ValueError("Length of subjects must match number of rows in idp_matrix.")
    if len(timepoints) != n_samples:
        raise ValueError("Length of timepoints must match number of rows in idp_matrix.")

    if idp_names is None:
        idp_names = [f"idp_{i+1}" for i in range(n_idps)]
    else:
        idp_names = list(idp_names)
        if len(idp_names) != n_idps:
            raise ValueError("idp_names length must match idp_matrix.shape[1].")

    df = pd.DataFrame(idp_matrix, columns=idp_names)
    df["subject"] = list(subjects)
    # df["timepoint"] = list(timepoints)  # keep if you need it later

    out_rows = []
    for subj, g in df.groupby("subject", sort=False):
        row = {"subject": subj}
        for col in idp_names:
            arr = g[col].dropna().to_numpy(dtype=float)
            n = arr.size
            if n == 0:
                row[col] = np.nan
                continue
            mean_val = arr.mean()
            if mean_val == 0:
                row[col] = np.nan
                continue
            if n == 2:
                row[col] = float(abs(arr[0] - arr[1]) / mean_val * 100.0)
            elif n > 2:
                row[col] = float(arr.std(ddof=1) / mean_val * 100.0)
            else:
                row[col] = np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows)

##########################################################################################################
# FOR LONGITUDINAL DATA: Subject Order Consistency using Spearman Correlations and Permutation Testing
##########################################################################################################

def _force_numeric_vector(series_like) -> np.ndarray:
    """Convert input to 1D float numpy array; non-convertible -> np.nan."""
    if isinstance(series_like, (pd.Series, pd.DataFrame)):
        arr = series_like.to_numpy().ravel()
    else:
        arr = np.asarray(series_like).ravel()
    try:
        return arr.astype(float)
    except Exception:
        out = np.empty(arr.shape, dtype=float)
        for i, v in enumerate(arr):
            try:
                out[i] = float(v)
            except Exception:
                out[i] = np.nan
        return out

def SubjectOrder_long(
    idp_matrix: np.ndarray,
    subjects: Sequence,
    timepoints: Sequence,
    idp_names: Optional[Sequence[str]] = None,
    nPerm: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Pairwise Spearman with permutation testing using scipy.stats.spearmanr.

    Returns DataFrame with columns:
      ["TimeA","TimeB","IDP","nPairs","SpearmanRho","NullMeanRho","pValue"]
    pValue is computed with the recommended +1 correction:
      p = (1 + count_ge) / (1 + valid_null_count)
    """
    # Validate and normalize inputs
    if not isinstance(idp_matrix, np.ndarray):
        idp_matrix = np.asarray(idp_matrix, dtype=float)
    if idp_matrix.ndim != 2:
        raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")
    n_samples, n_idps = idp_matrix.shape

    if len(subjects) != n_samples:
        raise ValueError("Length of subjects must match number of rows in idp_matrix.")
    if len(timepoints) != n_samples:
        raise ValueError("Length of timepoints must match number of rows in idp_matrix.")

    if idp_names is None:
        idp_names = [f"idp_{i+1}" for i in range(n_idps)]
    else:
        idp_names = list(idp_names)
        if len(idp_names) != n_idps:
            raise ValueError("idp_names length must match idp_matrix.shape[1].")

    if not isinstance(nPerm, int) or nPerm < 1:
        raise ValueError("nPerm must be an integer >= 1.")

    subjects_arr = np.asarray(subjects).astype(str)
    timepoints_arr = np.asarray(timepoints).astype(str)

    # Preserve order of first appearance for timepoints
    tp_index = pd.Index(timepoints_arr)
    tp_levels = tp_index.unique().tolist()

    rng = default_rng(seed)

    rows = []
    for ia in range(len(tp_levels) - 1):
        for ib in range(ia + 1, len(tp_levels)):
            tpA = tp_levels[ia]; tpB = tp_levels[ib]

            idxA_all = np.nonzero(timepoints_arr == tpA)[0]
            idxB_all = np.nonzero(timepoints_arr == tpB)[0]

            if idxA_all.size == 0 or idxB_all.size == 0:
                for idp in idp_names:
                    rows.append({"TimeA": tpA, "TimeB": tpB, "IDP": idp,
                                 "nPairs": 0, "SpearmanRho": np.nan,
                                 "NullMeanRho": np.nan, "pValue": np.nan})
                continue

            Ta_subj = subjects_arr[idxA_all]
            Tb_subj = subjects_arr[idxB_all]

            # subjects in A that also appear in B
            maskA = np.isin(Ta_subj, Tb_subj)
            if not np.any(maskA):
                for idp in idp_names:
                    rows.append({"TimeA": tpA, "TimeB": tpB, "IDP": idp,
                                 "nPairs": 0, "SpearmanRho": np.nan,
                                 "NullMeanRho": np.nan, "pValue": np.nan})
                continue

            common_subj = Ta_subj[maskA]
            idxA = idxA_all[np.nonzero(maskA)[0]]

            # map first occurrence in Tb to global row indices
            tb_index_map = {}
            for i, val in enumerate(Tb_subj):
                if val not in tb_index_map:
                    tb_index_map[val] = idxB_all[i]
            idxB = np.array([tb_index_map[s] for s in common_subj], dtype=int)

            # iterate idps (columns)
            for j, idp_name in enumerate(idp_names):
                xa_raw = idp_matrix[idxA, j] if j < n_idps else np.array([], dtype=float)
                yb_raw = idp_matrix[idxB, j] if j < n_idps else np.array([], dtype=float)

                xa = _force_numeric_vector(xa_raw)
                yb = _force_numeric_vector(yb_raw)

                valid_mask = ~(np.isnan(xa) | np.isnan(yb))
                xa = xa[valid_mask]; yb = yb[valid_mask]
                nPairs = xa.size

                if nPairs < 3:
                    rows.append({"TimeA": tpA, "TimeB": tpB, "IDP": idp_name,
                                 "nPairs": int(nPairs), "SpearmanRho": np.nan,
                                 "NullMeanRho": np.nan, "pValue": np.nan})
                    continue

                # Compute observed Spearman directly (handles ties)
                obs_rho, _ = spearmanr(xa, yb, nan_policy="omit")
                abs_obs = None if np.isnan(obs_rho) else abs(obs_rho)

                # Permutation null distribution (permute yb within matched pairs)
                sum_null = 0.0
                valid_null_count = 0
                count_ge = 0

                for _ in range(nPerm):
                    perm_idx = rng.permutation(nPairs)
                    null_rho, _ = spearmanr(xa, yb[perm_idx], nan_policy="omit")
                    if not np.isnan(null_rho):
                        sum_null += null_rho
                        valid_null_count += 1
                        if abs_obs is not None and abs(null_rho) >= abs_obs:
                            count_ge += 1

                null_mean = float(sum_null / valid_null_count) if valid_null_count > 0 else np.nan
                pval = float((1 + count_ge) / (1 + valid_null_count)) if valid_null_count > 0 else float("nan")

                rows.append({
                    "TimeA": tpA,
                    "TimeB": tpB,
                    "IDP": idp_name,
                    "nPairs": int(nPairs),
                    "SpearmanRho": float(obs_rho) if not np.isnan(obs_rho) else np.nan,
                    "NullMeanRho": null_mean,
                    "pValue": pval,
                })

    results = pd.DataFrame(rows, columns=["TimeA", "TimeB", "IDP", "nPairs", "SpearmanRho", "NullMeanRho", "pValue"])
    return results

##########################################################################################################
# FOR LONGITUDINAL DATA: 
# Additive batch effects - can 'batch' add additional variance in comparison to no 'batch'
# Multiplicative batch effects - comparing variance across batches
##########################################################################################################

# -------------------------
# Helper: build fixed terms
# -------------------------
def _build_fixed_formula_terms(fix_eff: Sequence[str], data: pd.DataFrame, do_zscore_predictors: bool = True) -> List[str]:
    """
    For each predictor v in fix_eff that is numeric in `data`, create zscore_v in `data` (per-feature/local df).
    Then return list of predictor column names to use in the formula, preferring zscore_v if present.
    """
    terms: List[str] = []
    if do_zscore_predictors and fix_eff is not None:
        for v in fix_eff:
            if v in data.columns and pd.api.types.is_numeric_dtype(data[v]):
                zname = f"zscore_{v}"
                if zname not in data.columns:
                    mu = data[v].mean(skipna=True)
                    sd = data[v].std(skipna=True)
                    if pd.isna(sd) or sd == 0:
                        data[zname] = 0.0
                    else:
                        data[zname] = (data[v] - mu) / sd
    for v in fix_eff or []:
        z = f"zscore_{v}"
        if do_zscore_predictors and z in data.columns:
            use = z
        else:
            use = v
        if use in data.columns:
            terms.append(use)
    return terms

# -------------------------
# Helper: safe fit wrapper
# -------------------------
def _safe_fit_mixedlm(formula_fixed: str, data: pd.DataFrame, group: str, reml: bool = False):
    if group not in data.columns:
        raise KeyError(f"group '{group}' not in data")
    data = data.copy()
    try:
        mdl = smf.mixedlm(formula_fixed, data, groups=data[group])
        res = mdl.fit(reml=reml, method="lbfgs")
        return res
    except Exception as e:
        warnings.warn(f"MixedLM fit failed for formula '{formula_fixed}': {e}")
        raise

# ----------------------------------------
# Helper: build input df
# ----------------------------------------
def _build_input_df_if_needed(
    data: Optional[pd.DataFrame],
    idp_matrix: Optional[np.ndarray],
    subjects: Optional[Sequence],
    timepoints: Optional[Sequence],
    batch_name: Optional[Sequence],
    idp_names: Optional[Iterable[str]],
    idvar: Optional[str] = None,
    batchvar: Optional[str] = None,
    timevar: Optional[str] = None,
    covariates: Optional[Dict[str, Sequence]] = None,
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Build or return a DataFrame for modeling and return the actual column names used:
      (df, idvar, batchvar, timevar).
    Defaults to idvar='subjects', batchvar='batches', timevar='timepoints' if not provided.
    """
    idvar = idvar if idvar is not None else "subject_ids"
    batchvar = batchvar if batchvar is not None else "batch"
    timevar = timevar if timevar is not None else "timepoints"

    if data is not None:
        df = data.copy()
    else:
        if idp_matrix is None:
            raise ValueError("Either `data` or `idp_matrix` must be provided.")
        if subjects is None:
            raise ValueError("`subjects` must be provided when `data` is None.")
        if batch_name is None:
            raise ValueError("`batch_name` must be provided when `data` is None.")
        if not isinstance(idp_matrix, np.ndarray):
            idp_matrix = np.asarray(idp_matrix, dtype=float)
        if idp_matrix.ndim != 2:
            raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")
        n_samples, n_features = idp_matrix.shape
        if len(subjects) != n_samples:
            raise ValueError("Length of subjects must match idp_matrix rows.")
        if len(batch_name) != n_samples:
            raise ValueError("Length of batch_name must match idp_matrix rows.")
        if timepoints is not None and len(timepoints) != n_samples:
            raise ValueError("Length of timepoints must match idp_matrix rows.")
        if idp_names is None:
            idp_names = [f"idp_{i+1}" for i in range(n_features)]
        else:
            idp_names = list(idp_names)
            if len(idp_names) != n_features:
                raise ValueError("idp_names length must match idp_matrix.shape[1].")
        df = pd.DataFrame(idp_matrix, columns=idp_names)
        df[idvar] = pd.Series(subjects, dtype="object").values
        if timepoints is not None:
            df[timevar] = pd.Series(timepoints, dtype="object").values
        df[batchvar] = pd.Series(batch_name, dtype="object").values

    covariates = dict(covariates or {})
    if covariates:
        for cname, seq in covariates.items():
            if cname in df.columns:
                warnings.warn(f"Covariate '{cname}' already exists in DataFrame; skipping insertion.")
                continue
            seq = list(seq)
            if len(seq) != len(df):
                raise ValueError(f"Length of covariate '{cname}' ({len(seq)}) does not match number of rows ({len(df)}).")
            df[cname] = pd.Series(seq).values

    if batchvar in df.columns:
        try:
            df[batchvar] = df[batchvar].astype("category")
        except Exception:
            pass

    return df, idvar, batchvar, timevar

# ----------------------------------------
# Main: AdditiveEffect_long (per-feature zscore behavior)
# ----------------------------------------
model_defs_add=[]
def AdditiveEffect_long(
    data: Optional[pd.DataFrame] = None,
    idp_matrix: Optional[np.ndarray] = None,
    subjects: Optional[Sequence] = None,
    timepoints: Optional[Sequence] = None,
    batch_name: Optional[Sequence] = None,
    idp_names: Optional[Iterable[str]] = None,
    covariates: Optional[Dict[str, Sequence]] = None,
    *,
    idvar: Optional[str] = None,
    batchvar: Optional[str] = None,
    timevar: Optional[str] = None,
    fix_eff: Optional[Iterable[str]] = None,
    ran_eff: Optional[Iterable[str]] = None,
    do_zscore: bool = True,
    reml: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test additive batch effect per feature (IDP), per-feature zscoring rules:

    - Numeric fixed predictors are z-scored per-feature (always).
    - If do_zscore == True: response (feature) is z-scored per-feature as well.
    - If do_zscore == False: response is left in original units.
    - Defaults for names: idvar='subjects', timevar='timepoints', batchvar='batches',
      unless the caller overrides idvar/batchvar/timevar.
    """
    # Build dataframe and canonical column names
    df, idcol, batchcol, tpcol = _build_input_df_if_needed(
        data=data,
        idp_matrix=idp_matrix,
        subjects=subjects,
        timepoints=timepoints,
        batch_name=batch_name,
        idp_names=idp_names,
        idvar=idvar,
        batchvar=batchvar,
        timevar=timevar,
        covariates=covariates,
    )

    covariates = dict(covariates or {})
    if fix_eff is None:
        fix_eff = list(covariates.keys())
    else:
        fix_eff = list(fix_eff)

    # RANDOM EFFECTS: infer only when caller omitted ran_eff (is None).
    if ran_eff is None:
        if idcol in df.columns:
            ran_eff = [idcol]
        else:
            raise KeyError("ran_eff not provided and idvar column not found in data.")
    else:
        ran_eff = list(ran_eff)

    # Validate referenced names exist
    to_check = {"fix_eff": fix_eff, "ran_eff": ran_eff}
    missing = {k: [x for x in v if x not in df.columns] for k, v in to_check.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise KeyError(f"Variables not found in data columns: {missing}. Available columns: {list(df.columns)}")

    # Determine feature columns (IDPs)
    exclude = {idcol, batchcol, tpcol}
    exclude |= set(covariates.keys())
    if idp_names is not None:
        feature_cols = list(idp_names)
    else:
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    V = len(feature_cols)
    if verbose:
        print(f"[AdditiveEffect_long] found {V} features")

    rows: List[Dict[str, Any]] = []
    for idx, feat in enumerate(feature_cols, 1):
        if verbose:
            print(f"[AdditiveEffect_long] ({idx}/{V}) testing additive batch effect for feature: {feat}")

        # Build local df for this feature (so per-feature zscoring is clean)
        # include feature, predictors, batch column, and grouping columns
        local_cols = [feat] + list(fix_eff) + [batchcol] + ran_eff
        local_df = df.loc[:, [c for c in local_cols if c in df.columns]].copy()

        # Drop rows where response is NaN (so zscores and fits use valid rows only)
        local_df = local_df[~local_df[feat].isna()].copy()
        if local_df.shape[0] < 3:
            if verbose:
                print(f"  skipping {feat}: too few non-NaN rows ({local_df.shape[0]})")
            rows.append({"Feature": feat, "TestStat": np.nan, "df": np.nan, "p-value": np.nan, "method": None})
            continue

        # Always z-score numeric predictors per-feature (local_df)
        # _build_fixed_formula_terms will create zscore_<v> for numeric fix_eff
        fixed_terms = _build_fixed_formula_terms(list(fix_eff or []), local_df, do_zscore_predictors=True)
        missing_fix = [vv for vv in (fix_eff or []) if (f"zscore_{vv}" not in local_df.columns) and (vv not in local_df.columns)]
        if missing_fix:
            warnings.warn(f"Fixed-effect columns not found in data (or zscore missing): {missing_fix}")

        # If do_zscore True -> z-score response per-feature and use z_<feat> as LHS
        if do_zscore:
            zresp = f"z_{feat}"
            if zresp not in local_df.columns:
                mu_r = local_df[feat].mean(skipna=True)
                sd_r = local_df[feat].std(skipna=True)
                if pd.isna(sd_r) or sd_r == 0:
                    local_df[zresp] = 0.0
                else:
                    local_df[zresp] = (local_df[feat] - mu_r) / sd_r
            lhs = zresp
        else:
            lhs = feat  # keep original units

        fixed_str = " + ".join(fixed_terms) if len(fixed_terms) > 0 else "1"
        full_fixed = f"{lhs} ~ {fixed_str} + C({batchcol})"
        reduced_fixed = f"{lhs} ~ {fixed_str}"

        # right before fitting
        if verbose:
            print("\n[MODEL]")
            print("Feature:", feat)
            print("Full model :", full_fixed)
            print("Reduced    :", reduced_fixed)
            print("Random eff :", ran_eff)

        model_defs_add.append ({
            "Feature": feat,
            "full_formula": full_fixed,
            "reduced_formula": reduced_fixed,
            "fix_eff": list(fix_eff),
            "ran_eff": list(ran_eff),})

        res_full = res_red = None
        group_name = ran_eff[0]
        try:
            res_full = _safe_fit_mixedlm(full_fixed, local_df, group=group_name, reml=reml)
        except Exception as e:
            if verbose:
                print(f"  full fit failed for {feat}: {e}")
        try:
            res_red = _safe_fit_mixedlm(reduced_fixed, local_df, group=group_name, reml=reml)
        except Exception as e:
            if verbose:
                print(f"  reduced fit failed for {feat}: {e}")

        LR = np.nan; df_stat = np.nan; pval = np.nan; used = None

        # Likelihood ratio test if both fits available
        if (res_full is not None) and (res_red is not None):
            try:
                llf_full = float(getattr(res_full, "llf", np.nan))
                llf_red = float(getattr(res_red, "llf", np.nan))
                if np.isfinite(llf_full) and np.isfinite(llf_red):
                    LR = 2.0 * (llf_full - llf_red)
                    try:
                        n_levels = int(pd.Categorical(local_df[batchcol]).nunique())
                        df_stat = max(n_levels - 1, 1)
                    except Exception:
                        df_stat = np.nan
                    if not np.isnan(df_stat):
                        pval = float(1.0 - chi2.cdf(LR, df_stat))
                    used = "LRT"
            except Exception as e:
                if verbose:
                    print(f"  LRT computation failed for {feat}: {e}")

        # If LRT didn't yield finite pval, try Wald-like test using full model params
        if not np.isfinite(pval) and (res_full is not None):
            try:
                pnames = list(res_full.params.index)
                batch_param_indices = []
                for i, pn in enumerate(pnames):
                    if (f"C({batchcol})" in pn) or (f"{batchcol}[T." in pn) or pn.startswith(f"{batchcol}_"):
                        batch_param_indices.append(i)
                if len(batch_param_indices) == 0:
                    cats = pd.Categorical(local_df[batchcol]).categories
                    for i, pn in enumerate(pnames):
                        for lvl in cats:
                            if f"{lvl}" in str(pn) and (batchcol in pn or f"C({batchcol})" in pn):
                                batch_param_indices.append(i)
                                break
                if len(batch_param_indices) > 0:
                    beta = res_full.params.to_numpy(dtype=float)
                    cov = res_full.cov_params()
                    cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
                    beta_b = beta[batch_param_indices]
                    Sigma_bb = cov_mat[np.ix_(batch_param_indices, batch_param_indices)]
                    try:
                        inv_Sigma_bb = np.linalg.inv(Sigma_bb)
                        W = float(beta_b.T @ inv_Sigma_bb @ beta_b)
                        df_w = len(beta_b)
                        p_w = float(1.0 - chi2.cdf(W, df_w))
                        LR, df_stat, pval = W, df_w, p_w
                        used = "Wald"
                    except np.linalg.LinAlgError:
                        Sigma_bb_pinv = np.linalg.pinv(Sigma_bb)
                        W = float(beta_b.T @ Sigma_bb_pinv @ beta_b)
                        df_w = len(beta_b)
                        p_w = float(1.0 - chi2.cdf(W, df_w))
                        LR, df_stat, pval = W, df_w, p_w
                        used = "Wald_pinv"
            except Exception as e:
                if verbose:
                    print(f"  Wald fallback failed for {feat}: {e}")

        rows.append({"Feature": feat, "TestStat": LR, "df": df_stat, "p-value": pval, "method": used})

    out = pd.DataFrame(rows)
    out = out.sort_values(by="TestStat", ascending=False).reset_index(drop=True)
    return out,model_defs_add

model_defs_mul = []
def MultiplicativeEffect_long(
    data: Optional[pd.DataFrame] = None,
    idp_matrix: Optional[np.ndarray] = None,
    subjects: Optional[Sequence] = None,
    timepoints: Optional[Sequence] = None,
    batch_name: Optional[Sequence] = None,
    idp_names: Optional[Iterable[str]] = None,
    covariates: Optional[Dict[str, Sequence]] = None,
    *,
    idvar: Optional[str] = None,
    batchvar: Optional[str] = None,
    timevar: Optional[str] = None,
    fix_eff: Optional[Iterable[str]] = None,
    ran_eff: Optional[Iterable[str]] = None,
    do_zscore: bool = True,
    reml: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Multiplicative (variability) batch test per feature.

    Robust to _build_input_df_if_needed returning either a DataFrame or (df, idcol, batchcol, tpcol).
    """
    # Call the helper and be defensive about the return type
    helper_out = _build_input_df_if_needed(
        data=data,
        idp_matrix=idp_matrix,
        subjects=subjects,
        timepoints=timepoints,
        batch_name=batch_name,
        idp_names=idp_names,
        idvar=idvar,
        batchvar=batchvar,
        timevar=timevar,
        covariates=covariates,
    )

    # Unpack in a robust way
    if isinstance(helper_out, tuple) and len(helper_out) >= 1:
        # New helper: expects (df, idcol, batchcol, tpcol)
        try:
            df = helper_out[0]
            # default fallback names if not provided
            idcol = helper_out[1] if len(helper_out) > 1 else (idvar if idvar is not None else "subjects")
            batchcol = helper_out[2] if len(helper_out) > 2 else (batchvar if batchvar is not None else "batches")
            tpcol = helper_out[3] if len(helper_out) > 3 else (timevar if timevar is not None else "timepoints")
        except Exception as e:
            raise RuntimeError(f"_build_input_df_if_needed returned an unexpected tuple shape: {e}")
    elif isinstance(helper_out, pd.DataFrame):
        # Old helper style: returned only a DataFrame
        df = helper_out
        idcol = idvar if idvar is not None else "subjects"
        batchcol = batchvar if batchvar is not None else "batches"
        tpcol = timevar if timevar is not None else "timepoints"
        if verbose:
            warnings.warn("Detected old _build_input_df_if_needed signature (returned DataFrame). Falling back to conventional column names.")
    else:
        raise RuntimeError(f"_build_input_df_if_needed returned unsupported type: {type(helper_out)}")

    # now df is a DataFrame and idcol/batchcol/tpcol are set
    covariates = dict(covariates or {})
    if fix_eff is None:
        fix_eff = list(covariates.keys())
    else:
        fix_eff = list(fix_eff)

    # RANDOM EFFECTS: infer only when caller omitted ran_eff (is None).
    if ran_eff is None:
        if idcol in df.columns:
            ran_eff = [idcol]
        else:
            raise KeyError("ran_eff not provided and idvar column not found in data.")
    else:
        ran_eff = list(ran_eff)

    # Ensure batch column is categorical (use returned batchcol)
    if batchcol in df.columns:
        try:
            df[batchcol] = df[batchcol].astype("category")
        except Exception:
            pass

    # Defaults: ran_eff -> [idcol], fix_eff -> covariate keys + tpcol + batchcol
    if len(ran_eff) == 0:
        if idcol in df.columns:
            ran_eff = [idcol]
        else:
            raise KeyError("ran_eff not provided and idvar column not found in data.")
    if len(fix_eff) == 0:
        inferred_fix = list(covariates.keys())
        if tpcol in df.columns and tpcol not in inferred_fix:
            inferred_fix.append(tpcol)
        if batchcol in df.columns and batchcol not in inferred_fix:
            inferred_fix.append(batchcol)
        fix_eff = inferred_fix

    # Validate referenced names exist
    to_check = {"fix_eff": fix_eff, "ran_eff": ran_eff}
    missing = {k: [x for x in v if x not in df.columns] for k, v in to_check.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise KeyError(f"Variables not found in data columns: {missing}. Available columns: {list(df.columns)}")

    # Determine feature columns (IDPs)
    exclude = {idcol, batchcol, tpcol}
    exclude |= set(covariates.keys())
    if idp_names is not None:
        feature_cols = list(idp_names)
    else:
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    V = len(feature_cols)
    if verbose:
        print(f"[MultEffect_long] found {V} features")

    rows: List[Dict[str, Any]] = []

    for idx, feat in enumerate(feature_cols, 1):
        if verbose:
            print(f"[MultEffect_long] ({idx}/{V}) testing multiplicative batch effect for feature: {feat}")

        # Build local df for this feature
        local_cols = [feat] + list(fix_eff) + [batchcol] + ran_eff
        local_df = df.loc[:, [c for c in local_cols if c in df.columns]].copy()

        # Drop rows where response is NaN
        local_df = local_df[~local_df[feat].isna()].copy()
        if local_df.shape[0] < 3:
            if verbose:
                print(f"  skipping {feat}: too few non-NaN rows ({local_df.shape[0]})")
            rows.append({"Feature": feat, "ChiSq": np.nan, "DF": np.nan, "p-value": np.nan, "method": None})
            continue

        # Always z-score numeric predictors per-feature (in local_df)
        fixed_terms = _build_fixed_formula_terms(list(fix_eff or []), local_df, do_zscore_predictors=True)
        missing_fix = [vv for vv in (fix_eff or []) if (f"zscore_{vv}" not in local_df.columns) and (vv not in local_df.columns)]
        if missing_fix:
            warnings.warn(f"Fixed-effect columns not found in data (or zscore missing): {missing_fix}")

        # Optionally z-score response per-feature
        if do_zscore:
            zresp = f"z_{feat}"
            if zresp not in local_df.columns:
                mu_r = local_df[feat].mean(skipna=True)
                sd_r = local_df[feat].std(skipna=True)
                if pd.isna(sd_r) or sd_r == 0:
                    local_df[zresp] = 0.0
                else:
                    local_df[zresp] = (local_df[feat] - mu_r) / sd_r
            lhs = zresp
        else:
            lhs = feat

        fixed_str = " + ".join(fixed_terms) if len(fixed_terms) > 0 else "1"
        full_fixed = f"{lhs} ~ {fixed_str} + C({batchcol})"

        if verbose:
            print("\n[MODEL]")
            print("Feature     :", feat)
            print("Full model  :", full_fixed)
            print("Random eff  :", ran_eff)

        model_defs_mul.append({
            "Feature": feat,
            "formula": full_fixed,
            "ran_eff": list(ran_eff),
            "fix_eff": list(fix_eff),})


        # Fit full mixed model to obtain residuals
        res_full = None
        group_name = ran_eff[0]
        try:
            res_full = _safe_fit_mixedlm(full_fixed, local_df, group=group_name, reml=reml)
        except Exception as e:
            rows.append({"Feature": feat, "ChiSq": np.nan, "DF": np.nan, "p-value": np.nan, "method": None})
            if verbose:
                print(f"  fit failed for {feat}: {e}")
            continue

        # Obtain residuals aligned with local_df indices
        try:
            resid = res_full.resid
            resid = pd.Series(np.asarray(resid), index=local_df.index)
        except Exception:
            resid = pd.Series(np.asarray(getattr(res_full, "resid", np.asarray([]))), index=local_df.index if len(local_df) == len(getattr(res_full, "resid", [])) else None)
        

        # Group residuals by batch level and run Fligner test
        try:
            cats = pd.Categorical(local_df[batchcol]).categories
            groups = [resid[local_df[batchcol] == lvl].dropna().values for lvl in cats]
            if len(groups) < 2 or any(len(g) == 0 for g in groups):
                raise ValueError("Not enough observations per batch group for Fligner test.")
            stat, pval = fligner(*groups, center="median")
            df_stat = len(groups) - 1
            method = "Fligner"
        except Exception as e:
            if verbose:
                print(f"  fligner test failed for {feat}: {e}")
            stat = np.nan; pval = np.nan; df_stat = np.nan; method = None

        rows.append({
            "Feature": feat,
            "ChiSq": float(stat) if not np.isnan(stat) else np.nan,
            "DF": df_stat,
            "p-value": float(pval) if not np.isnan(pval) else np.nan,
            "method": method
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(by="ChiSq", ascending=False).reset_index(drop=True)
    return out, model_defs_mul

def simulate_longitudinal_batch_data_mixed(
    n_subjects=20,
    n_timepoints=3,
    n_sites=2,
    n_features=4,
    seed=1,
    feature_names=None,
    timepoints=None,
    sites=None,
    base_mean=1500,
    base_sd=80,
    slope_sd=8,
    noise_sd=20,
    additive_shift=None,
    multiplicative_scale=None,
    include_age=True,
):
    np.random.seed(seed)

    if timepoints is None:
        timepoints = [f"TP{i}" for i in range(n_timepoints)]
    if sites is None:
        sites = [f"Site_{chr(65+i)}" for i in range(n_sites)]
    if feature_names is None:
        feature_names = [f"Region_{i+1}" for i in range(n_features)]

    additive_shift = additive_shift or {}
    multiplicative_scale = multiplicative_scale or {}

    subject_ids = np.arange(1, n_subjects + 1)
    ages = np.random.normal(65, 8, n_subjects) if include_age else np.zeros(n_subjects)

    feature_means = np.linspace(base_mean, base_mean + 800, n_features)

    baselines = {
        feat: np.random.normal(feature_means[i], base_sd, n_subjects)
        for i, feat in enumerate(feature_names)
    }

    slopes = {
        feat: np.random.normal(0, slope_sd, n_subjects)
        for feat in feature_names
    }

    age_effects = {
        feat: np.random.uniform(-4, 4)
        for feat in feature_names
    }

    rows = []

    for subj_idx, subj in enumerate(subject_ids):
        age = ages[subj_idx]

        # random site order for this subject, so site is not identical to timepoint
        subject_sites = np.random.choice(sites, size=n_timepoints, replace=True)

        for tp_idx, tp in enumerate(timepoints):
            site = subject_sites[tp_idx]

            row = {
                "Subject": subj,
                "Timepoint": tp,
                "Site": site,
                "Age": round(age, 1),
            }

            for feat in feature_names:
                true_signal = (
                    baselines[feat][subj_idx]
                    + slopes[feat][subj_idx] * tp_idx
                    + age_effects[feat] * age
                )

                add = additive_shift.get(site, {}).get(feat, 0.0)
                scale = multiplicative_scale.get(site, {}).get(feat, 1.0)

                observed = true_signal + add + np.random.normal(0, noise_sd * scale)
                row[feat] = round(observed, 1)

            rows.append(row)

    return pd.DataFrame(rows)

def simulate_cross_sectional_harmonization_data(
    n_subjects=60,
    n_sites=3,
    n_features=4,
    seed=1,
    feature_names=None,
    site_names=None,
    additive_shift=None,
    multiplicative_scale=None,
):
    """
    Simulate cross-sectional site/batch data with age, subject ID, and feature values.

    Data structure:
        Subject, Age, Site, Region_1 ... Region_n

    Additive effects:
        fixed mean shifts by site and feature

    Multiplicative effects:
        site-specific variance scaling by feature
    """
    rng = np.random.default_rng(seed)

    if site_names is None:
        site_names = [f"Site_{chr(65+i)}" for i in range(n_sites)]
    if feature_names is None:
        feature_names = [f"Region_{i+1}" for i in range(n_features)]

    if len(site_names) != n_sites:
        raise ValueError("len(site_names) must match n_sites")
    if len(feature_names) != n_features:
        raise ValueError("len(feature_names) must match n_features")

    additive_shift = additive_shift or {}
    multiplicative_scale = multiplicative_scale or {}

    # Balanced site allocation, then shuffled
    sites = np.tile(site_names, int(np.ceil(n_subjects / n_sites)))[:n_subjects]
    rng.shuffle(sites)

    # Covariate
    age = rng.normal(loc=65, scale=8, size=n_subjects)

    # Feature-level biology
    feature_means = np.linspace(1500, 2200, n_features)
    age_slopes = rng.uniform(-6, 6, size=n_features)  # linear age effects

    # Subject-specific baseline variation
    subject_baselines = rng.normal(0, 70, size=(n_subjects, n_features))

    rows = []
    for i in range(n_subjects):
        row = {
            "Subject": i + 1,
            "Age": round(float(age[i]), 1),
            "Site": sites[i],
        }

        for j, feat in enumerate(feature_names):
            site = sites[i]

            true_value = (
                feature_means[j]
                + subject_baselines[i, j]
                + age_slopes[j] * age[i]
            )

            add = additive_shift.get(site, {}).get(feat, 0.0)
            scale = multiplicative_scale.get(site, {}).get(feat, 1.0)

            observed = true_value + add + rng.normal(0, 20 * scale)
            row[feat] = round(float(observed), 1)

        rows.append(row)

    return pd.DataFrame(rows)

def regression_harmonize_site(
    df,
    feature_cols,
    site_col="Site",
    covariates=("Age", "Timepoint"),
):
    """
    Remove additive site effects using feature-wise linear regression.

    Harmonized value:
        y_adj = y - site_component + mean(site_component)

    This preserves the covariate effects while removing the estimated site shift.
    """
    out = df.copy()
    model_rows = []

    for feat in feature_cols:
        cov_terms = []
        for c in covariates:
            if c in out.columns:
                if pd.api.types.is_numeric_dtype(out[c]):
                    cov_terms.append(c)
                else:
                    cov_terms.append(f"C({c})")

        formula = f"{feat} ~ " + " + ".join(cov_terms + [f"C({site_col})"])

        fit = smf.ols(formula, data=out).fit()

        # prediction with and without site
        formula_no_site = f"{feat} ~ " + " + ".join(cov_terms)
        fit_no_site = smf.ols(formula_no_site, data=out).fit()

        # harmonized values = fitted values without site + residuals
        out[f"{feat}_harm"] = fit_no_site.fittedvalues + fit.resid

        model_rows.append({
            "Feature": feat,
            "Formula": formula,
            "Site_pvalue": fit.pvalues[[c for c in fit.pvalues.index if "C(Site)" in c]].min()
                           if any("C(Site)" in c for c in fit.pvalues.index) else np.nan,
            "R2": fit.rsquared
        })

    model_summary = pd.DataFrame(model_rows)
    return out, model_summary