from __future__ import annotations

import math
import re
import warnings
from typing import Sequence, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf


def iqm_harmonise(
    data: pd.DataFrame,
    idp_list: Sequence[str],
    qc_list: Sequence[str],
    preserve_covars: Sequence[str] = ("age",),
    adjust_covars: Sequence[str] = ("timepoint",),
    reverse_guard_covars: Optional[Sequence[str]] = None,
    categorical_covars: Sequence[str] = ("timepoint", "batch"),
    batch_col: str = "batch",
    subject_col: str = "subjectID",
    age_source_col: str = "Final_Age",
    batch_source_col: str = "Site",
    age_col: str = "age",
    iqm_variance: float = 95,
    p_thr: float = 0.05,
    max_qcs: float = float("inf"),
    apply_pca: bool = True,
    outfilename: str = "iqm_harmonised.csv",
    summary_csv: str = "qc_selection_summary.csv",
    additive_detail_csv: str = "qc_selection_additive_details.csv",
    multiplicative_detail_csv: str = "qc_selection_multiplicative_details.csv",
    selected_additive_qcs_csv: str = "selected_additive_qcs_by_volume.csv",
    selected_multiplicative_qcs_csv: str = "selected_multiplicative_qcs_by_volume.csv",
    allow_ols_fallback: bool = True,
    optimizer_order: Sequence[str] = ("lbfgs", "powell", "cg", "nm"),
    maxiter: int = 2000,
    verbose_model_fits: bool = False,
    enable_multiplicative: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    IQM harmonisation pipeline.

    Parameters
    ----------
    preserve_covars
        Covariates that QC components are not allowed to encode.
        These are screened with p > p_thr in the QC-selection models.
    adjust_covars
        Covariates included in the models as adjustment variables.
        They are not explicitly screened as preserve variables.
    reverse_guard_covars
        Optional subset of preserve_covars to screen with reverse-response guards.
        If None, defaults to numeric preserve covariates only.
    categorical_covars
        Covariates treated as categorical in formulas via C(...).
    apply_pca
        If True, z-score QC metrics and run PCA, then use retained PCs.
        If False, use the raw QC variables directly after z-scoring them.

    Returns
    -------
    data_out
        Data with harmonised_<IDP> columns added.
    qc_selection
        Summary dict of selected QCs and correction decisions.
    add_detail_df
        Additive selection detail table.
    mult_detail_df
        Multiplicative selection detail table.
    summary_df
        Per-volume summary table.


    Detailed Arguments
    ----------
    data
        Main dataframe containing IDPs, QC metrics, subject IDs, and covariates.

    idp_list
        List of imaging-derived phenotypes (IDPs) / response variables to harmonise.

    qc_list
        List of QC metric columns used as candidate harmonisation regressors.

    preserve_covars
        Biological covariates that QC variables should NOT encode.
        QCs are rejected if they significantly explain these covariates.

    adjust_covars
        Covariates included for statistical adjustment only.

    reverse_guard_covars
        Optional subset of preserve_covars used in reverse-response tests.
        If None, numeric preserve covariates are used automatically.

    categorical_covars
        Covariates treated as categorical in regression formulas.

    batch_col
        Batch/site variable used as the harmonisation target.

    subject_col
        Subject identifier used for random intercepts in MixedLM.

    age_source_col
        Source column used to generate age_col if age_col is absent.

    batch_source_col
        Source column used to generate batch_col if batch_col is absent.

    age_col
        Standardized working age column.

    iqm_variance
        Percentage variance threshold for retaining PCA QC components.

    p_thr
        Statistical threshold for QC selection and multiplicative testing.

    max_qcs
        Maximum number of QCs allowed per volume.

    apply_pca
        If True:
            z-score QC metrics -> PCA -> retain PCs.
        If False:
            use z-scored raw QC variables directly.

    allow_ols_fallback
        If MixedLM fails to converge, allow fallback to OLS.

    optimizer_order
        Order of optimizers attempted for MixedLM fitting.

    maxiter
        Maximum optimizer iterations.

    verbose_model_fits
        Print detailed model fitting diagnostics.

    enable_multiplicative
        If True: Multiplicative correction is applied. Note that this is computationally more intensive 
        in the current Python implementation because it relies on repeated mixed-effects model fitting.

    Returns
    -------
    data_out
        Dataframe with harmonised_<IDP> columns added.

    qc_selection
        Dictionary summarising selected QCs and correction decisions.

    add_detail_df
        Additive QC selection diagnostics.

    mult_detail_df
        Multiplicative QC selection diagnostics.

    summary_df
        Per-volume summary table.


    Example run
    -----------

    df = pd.read_csv("test_data/alldata.csv")
    idp_list = pd.read_csv("test_data/IDP_list.csv", header=None).iloc[:, 0].dropna().astype(str).str.strip().tolist()
    iqm_list = pd.read_csv("test_data/IQM_list.csv", header=None).iloc[:, 0].dropna().astype(str).str.strip().tolist()


    data_out, qc_selection, add_detail_df, mult_detail_df, summary_df = iqm_harmonise(
        data=df,
        idp_list=idp_list,
        qc_list=iqm_list,
        preserve_covars=("age",),
        adjust_covars=("timepoint",),
        reverse_guard_covars=("age",),
        categorical_covars=("timepoint", "scan_session"),
        batch_col="scan_session",
        subject_col="subject",
        age_source_col="age",
        batch_source_col="scan_session",
        age_col="age",
        iqm_variance=95,
        p_thr=0.05,
        apply_pca=True,
        verbose_model_fits=True,
        enable_multiplicative: bool = False
    )    
        
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def ordered_unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    preserve_covars = ordered_unique(list(preserve_covars))
    adjust_covars = ordered_unique(list(adjust_covars))
    categorical_covars = ordered_unique(list(categorical_covars))

    if batch_col not in categorical_covars:
        categorical_covars.append(batch_col)

    categorical_covars_set = set(categorical_covars)

    if reverse_guard_covars is None:
        reverse_guard_covars = [c for c in preserve_covars if c not in categorical_covars_set]
    else:
        reverse_guard_covars = ordered_unique(list(reverse_guard_covars))
        reverse_guard_covars = [c for c in reverse_guard_covars if c in preserve_covars]
        reverse_guard_covars = [c for c in reverse_guard_covars if c not in categorical_covars_set]

    def term_expr(var: str) -> str:
        return f"C({var})" if var in categorical_covars_set else var

    def rhs_expr(covars: Sequence[str]) -> str:
        covars = ordered_unique([c for c in covars if c is not None and c != ""])
        if len(covars) == 0:
            return "1"
        return " + ".join(term_expr(c) for c in covars)

    def strip_random_effect(formula: str) -> str:
        out = re.sub(r"\s*\+\s*\(\s*1\s*\|\s*[\w]+\s*\)\s*", " ", formula)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def complete_case(df_in: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        cols = list(dict.fromkeys(cols))
        return df_in.loc[:, cols].dropna().copy()

    def fixed_params(result):
        return getattr(result, "fe_params", result.params)
    
    def build_bio_cache(data_in: pd.DataFrame, volname: str):
        """
        Cache the biology-only fit and proxy once per IDP.
        Used by the multiplicative stage to avoid recomputing the same model twice.
        """
        bio_df = complete_case(
            data_in,
            [subject_col] + list(preserve_covars) + list(adjust_covars) + [volname],
        )
        bio_formula = f"{volname} ~ {rhs_expr(list(preserve_covars) + list(adjust_covars))} + (1|{subject_col})"
        bio_fit, _, _ = fit_mixed_then_ols(bio_df, bio_formula, subject_col)

        res = np.asarray(bio_fit.resid, dtype=float)

        resid_proxy = pd.Series(
            np.log(np.maximum(res ** 2, np.finfo(float).eps)),
            index=bio_df.index,
            name="resid_proxy",
        )

        return {
            "bio_df": bio_df,
            "bio_fit": bio_fit,
            "resid": res,
            "resid_proxy": resid_proxy,
            "bio_formula": bio_formula,
        }

    def _fit_mixed_formula(df_in: pd.DataFrame, formula: str, group_col: str, method: str):
        fixed_formula = strip_random_effect(formula)
        model = smf.mixedlm(fixed_formula, data=df_in, groups=df_in[group_col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)
        return res

    def fit_mixed_then_ols(df_in: pd.DataFrame, formula: str, group_col: str):
        """
        Try multiple MixedLM optimizers first.
        If all fail and allow_ols_fallback=True, fall back to OLS.
        Returns result, family, used_method.
        """
        fixed_formula = strip_random_effect(formula)

        if verbose_model_fits:
            print("\n--------------------------------------------------")
            print("Requested formula:")
            print(" ", formula)
            print("Fixed-effects formula used:")
            print(" ", fixed_formula)
            print("Random effects:")
            print(f"  (1 | {group_col})")
            print("Trying MixedLM optimizers:", list(optimizer_order))

        for method in optimizer_order:
            try:
                if verbose_model_fits:
                    print(f"  Trying optimizer: {method}")
                res = _fit_mixed_formula(df_in, formula, group_col, method)
                if getattr(res, "converged", True):
                    if verbose_model_fits:
                        print("  SUCCESS")
                        print("  Fit family: MixedLM")
                        print("  Optimizer:", method)
                        print("  Converged:", getattr(res, "converged", True))
                        print("--------------------------------------------------")
                    return res, "mixed", method
            except Exception as exc:
                if verbose_model_fits:
                    print(f"  FAILED ({method})")
                    print("   Reason:", exc)

        if not allow_ols_fallback:
            raise RuntimeError(f"MixedLM failed for formula: {fixed_formula}")

        if verbose_model_fits:
            print("\n  All MixedLM optimizers failed.")
            print("  Falling back to OLS.")
            print("  WARNING: subject random effect removed.")
            print("--------------------------------------------------")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols_res = smf.ols(fixed_formula, data=df_in).fit()
        return ols_res, "ols", "ols"

    def fit_pair_same_family(df_in: pd.DataFrame, full_formula: str, reduced_formula: str, group_col: str):
        """
        Fit a nested pair using the same family and same optimizer when possible.
        First tries MixedLM for both, then OLS for both.
        Returns full_fit, reduced_fit, family, method_full, method_reduced.
        """
        full_fixed = strip_random_effect(full_formula)
        red_fixed = strip_random_effect(reduced_formula)

        for method in optimizer_order:
            try:
                full_fit = _fit_mixed_formula(df_in, full_formula, group_col, method)
                red_fit = _fit_mixed_formula(df_in, reduced_formula, group_col, method)
                if getattr(full_fit, "converged", True) and getattr(red_fit, "converged", True):
                    return full_fit, red_fit, "mixed", method, method
            except Exception:
                pass

        if not allow_ols_fallback:
            raise RuntimeError(
                "MixedLM failed for nested comparison:\n"
                f"FULL: {full_fixed}\nRED : {red_fixed}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_ols = smf.ols(full_fixed, data=df_in).fit()
            red_ols = smf.ols(red_fixed, data=df_in).fit()
        return full_ols, red_ols, "ols", "ols", "ols"

    def lrt_pvalue(full_fit, reduced_fit, family: str) -> float:
        lr_stat = 2.0 * (full_fit.llf - reduced_fit.llf)
        lr_stat = max(float(lr_stat), 0.0)

        if family == "mixed":
            df_full = int(getattr(full_fit, "k_fe", len(full_fit.params)))
            df_red = int(getattr(reduced_fit, "k_fe", len(reduced_fit.params)))
        else:
            df_full = int(getattr(full_fit, "df_model", len(full_fit.params) - 1))
            df_red = int(getattr(reduced_fit, "df_model", len(reduced_fit.params) - 1))

        df_diff = max(df_full - df_red, 1)
        return float(chi2.sf(lr_stat, df_diff))

    def evaluate_qc_stage(
        df_stage: pd.DataFrame,
        response: str,
        qcname: str,
        proxy_term: str,
        stage_name: str,
    ):
        """
        Evaluate one QC feature for additive or multiplicative selection.
        Returns (passed_bool, row_dict).
        """
        fixed_terms = ordered_unique(list(preserve_covars) + list(adjust_covars) + [batch_col, proxy_term])

        row = {
            "response": response,
            "volume": response,   # keep a "volume" field for CSV compatibility
            "qc": qcname,
            "stage": stage_name,
        }

        passed = True
        first_family = None
        first_method = None

        # Preserve covariates: QC should not explain them
        for cov in preserve_covars:
            full_formula = f"{qcname} ~ {rhs_expr(fixed_terms)} + (1|{subject_col})"
            red_formula = f"{qcname} ~ {rhs_expr([t for t in fixed_terms if t != cov])} + (1|{subject_col})"
            full_fit, red_fit, fam, method_full, method_red = fit_pair_same_family(df_stage, full_formula, red_formula, subject_col)
            p_cov = lrt_pvalue(full_fit, red_fit, fam)
            row[f"p_preserve_{cov}"] = p_cov
            passed = passed and (p_cov > p_thr)
            if first_family is None:
                first_family = fam
                first_method = method_full

        # batch must be significant
        full_formula = f"{qcname} ~ {rhs_expr(fixed_terms)} + (1|{subject_col})"
        red_formula = f"{qcname} ~ {rhs_expr([t for t in fixed_terms if t != batch_col])} + (1|{subject_col})"
        full_fit, red_fit, fam, method_full, method_red = fit_pair_same_family(df_stage, full_formula, red_formula, subject_col)
        p_batch = lrt_pvalue(full_fit, red_fit, fam)
        row["p_batch"] = p_batch
        passed = passed and (p_batch < p_thr)
        if first_family is None:
            first_family = fam
            first_method = method_full

        # proxy term must not drive QC
        full_formula = f"{qcname} ~ {rhs_expr(fixed_terms)} + (1|{subject_col})"
        red_formula = f"{qcname} ~ {rhs_expr([t for t in fixed_terms if t != proxy_term])} + (1|{subject_col})"
        full_fit, red_fit, fam, method_full, method_red = fit_pair_same_family(df_stage, full_formula, red_formula, subject_col)
        p_proxy = lrt_pvalue(full_fit, red_fit, fam)
        row["p_proxy"] = p_proxy
        passed = passed and (p_proxy > p_thr)
        if first_family is None:
            first_family = fam
            first_method = method_full

        # reverse guards for numeric preserve covariates only
        for cov in reverse_guard_covars:
            reverse_terms = ordered_unique([qcname] + list(preserve_covars) + list(adjust_covars) + [batch_col, proxy_term])
            full_formula = f"{cov} ~ {rhs_expr(reverse_terms)} + (1|{subject_col})"
            red_formula = f"{cov} ~ {rhs_expr([t for t in reverse_terms if t != qcname])} + (1|{subject_col})"
            full_fit, red_fit, fam, method_full, method_red = fit_pair_same_family(df_stage, full_formula, red_formula, subject_col)
            p_rev = lrt_pvalue(full_fit, red_fit, fam)
            row[f"p_reverse_{cov}"] = p_rev
            passed = passed and (p_rev > p_thr)
            if first_family is None:
                first_family = fam
                first_method = method_full

        row["passed"] = int(passed)
        row["fit_family"] = first_family
        row["fit_method"] = first_method
        return passed, row

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    data = data.copy()
    data.columns = data.columns.astype(str).str.strip()

    print("=== IQM Harmonization (Flexible Version) ===")
    print(f"P-value threshold: {p_thr:.4f}")
    print(f"Max QCs per volume: {max_qcs}")
    print(f"PCA variance threshold: {iqm_variance:.1f}%")
    print(f"Apply PCA: {apply_pca}")
    print(f"Preserve covars: {list(preserve_covars)}")
    print(f"Adjust covars: {list(adjust_covars)}")
    print(f"Categorical covars: {list(categorical_covars)}")
    print(f"Reverse-guard covars: {list(reverse_guard_covars)}")

    # prepare age / batch columns
    if age_col not in data.columns:
        if age_source_col not in data.columns:
            raise ValueError(f"Missing '{age_col}' and source column '{age_source_col}'")
        data[age_col] = zscore(data[age_source_col].astype(float))

    if batch_col not in data.columns:
        if batch_source_col not in data.columns:
            raise ValueError(f"Missing '{batch_col}' and source column '{batch_source_col}'")
        data[batch_col] = data[batch_source_col]

    # required columns
    for c in [subject_col] + list(adjust_covars) + list(preserve_covars):
        if c not in data.columns:
            raise ValueError(f"Missing required covariate column: {c}")

    missing_idps = [c for c in idp_list if c not in data.columns]
    missing_qcs = [c for c in qc_list if c not in data.columns]
    if missing_idps:
        raise ValueError(f"Missing IDP columns: {missing_idps}")
    if missing_qcs:
        raise ValueError(f"Missing QC columns: {missing_qcs}")

    # categorical casting
    data[subject_col] = data[subject_col].astype("category")
    for c in categorical_covars:
        if c in data.columns:
            data[c] = data[c].astype("category")

    # ------------------------------------------------------------------
    # STEP 1: QC basis (PCA or raw standardized QCs)
    # ------------------------------------------------------------------
    print("\nStep 1: Preparing QC metrics...")

    qc_df = data.loc[:, list(qc_list)].copy()
    if qc_df.isna().any().any():
        raise ValueError("QC columns contain missing values. Please clean/impute before running.")

    qc_matrix = qc_df.to_numpy(dtype=float)

    # remove zero-variance QCs
    stds = np.std(qc_matrix, axis=0, ddof=1)
    zero_var_mask = stds == 0
    removed = [q for q, keep in zip(qc_list, ~zero_var_mask) if not keep]
    if removed:
        print(f"  Removing {len(removed)} zero-variance QCs")
        print(" ", removed[:20])

    qc_matrix = qc_matrix[:, ~zero_var_mask]
    qc_list_clean = [q for q, keep in zip(qc_list, ~zero_var_mask) if keep]

    if qc_matrix.shape[1] == 0:
        raise ValueError("No QC variables left after removing zero-variance columns.")

    qc_z = StandardScaler().fit_transform(qc_matrix)

    if apply_pca:
        pca = PCA()
        scores = pca.fit_transform(qc_z)
        explained = pca.explained_variance_ratio_ * 100.0
        cumvar = np.cumsum(explained)
        num_qc_features = int(np.argmax(cumvar >= iqm_variance) + 1)

        qc_feature_names = [f"QC{i+1}" for i in range(num_qc_features)]
        for i, name in enumerate(qc_feature_names):
            data[name] = scores[:, i]

        qc_basis = "PCA"
        print(f"  PCA input shape: {qc_z.shape}")
        print(f"  Retaining PCs: {num_qc_features} ({cumvar[num_qc_features - 1]:.2f}%)")
    else:
        num_qc_features = qc_z.shape[1]
        qc_feature_names = list(qc_list_clean)
        for i, name in enumerate(qc_feature_names):
            data[name] = qc_z[:, i]

        qc_basis = "RAW"
        cumvar = None
        print(f"  Using raw QC variables directly: {num_qc_features} features")

        # ------------------------------------------------------------------
    # STEP 1b: cache biology-only fits for multiplicative stage
    # ------------------------------------------------------------------
    print("\nStep 1b: Caching biology-only fits for multiplicative stage...")
    bio_cache: Dict[str, Dict[str, Any]] = {}
    for volname in idp_list:
        bio_cache[volname] = build_bio_cache(data, volname)
    print("  Biology-only cache built.")

    # ------------------------------------------------------------------
    # STEP 2: ADDITIVE QC SELECTION
    # ------------------------------------------------------------------
    print("\nStep 2: Selecting QCs for ADDITIVE correction...")
    print("  Criteria: batch-driven, not preserve-covariate-driven")

    good_pcs_add = np.zeros((num_qc_features, len(idp_list)), dtype=int)
    all_out_add = []
    additive_detail_rows = []

    for v_idx, volname in enumerate(idp_list):
        print(f"\n=== Additive selection: {volname} ===")
        rows = []

        for qc_i in range(num_qc_features):
            qcname = qc_feature_names[qc_i]

            df_stage = complete_case(
                data,
                [subject_col] + list(preserve_covars) + list(adjust_covars) + [batch_col, volname, qcname],
            )

            try:
                passed, row = evaluate_qc_stage(df_stage, volname, qcname, proxy_term=volname, stage_name="additive")
                good_pcs_add[qc_i, v_idx] = int(passed)
            except Exception as exc:
                print(f"  Additive QC check failed for {qcname}: {exc}")
                row = {
                    "response": volname,
                    "volume": volname,
                    "qc": qcname,
                    "stage": "additive",
                    "p_batch": np.nan,
                    "p_proxy": np.nan,
                    "passed": 0,
                    "fit_family": "failed",
                    "fit_method": "failed",
                }
                for cov in preserve_covars:
                    row[f"p_preserve_{cov}"] = np.nan
                for cov in reverse_guard_covars:
                    row[f"p_reverse_{cov}"] = np.nan

            rows.append(row)
            additive_detail_rows.append(row)

        all_out_add.append([volname, rows])

    n_add = good_pcs_add.sum(axis=0)
    print(f"  QCs selected: min={n_add.min()}, max={n_add.max()}, mean={n_add.mean():.1f}")

    # ------------------------------------------------------------------
    # STEP 3: MULTIPLICATIVE QC SELECTION
    # ------------------------------------------------------------------
    if enable_multiplicative:

        print("\nStep 3: Selecting QCs for MULTIPLICATIVE correction...")
        print(
            "  NOTE: Multiplicative correction is computationally more intensive "
            "in the current Python implementation because it relies on repeated "
            "mixed-effects model fitting."
        )
        print("\nStep 3: Selecting QCs for MULTIPLICATIVE correction...")
        print("  KEY: Using cached biology-only residual proxy")
        print("  Criteria: batch-driven, NOT strongly variance-driven")

        good_pcs_mult = np.zeros((num_qc_features, len(idp_list)), dtype=int)
        multiplicative_detail_rows = []

        for v_idx, volname in enumerate(idp_list):
            print(f"\n=== Multiplicative selection: {volname} ===")
            rows = []

            resid_proxy = bio_cache[volname]["resid_proxy"]

            for qc_i in range(num_qc_features):
                qcname = qc_feature_names[qc_i]

                df_stage = data.loc[:, [subject_col] + list(preserve_covars) + list(adjust_covars) + [batch_col, qcname]].copy()
                df_stage = df_stage.join(resid_proxy, how="inner").dropna().copy()

                try:
                    passed, row = evaluate_qc_stage(
                        df_stage,
                        volname,
                        qcname,
                        proxy_term="resid_proxy",
                        stage_name="multiplicative",
                    )
                    good_pcs_mult[qc_i, v_idx] = int(passed)
                except Exception as exc:
                    print(f"  Multiplicative QC check failed for {qcname}: {exc}")
                    row = {
                        "response": volname,
                        "volume": volname,
                        "qc": qcname,
                        "stage": "multiplicative",
                        "p_batch": np.nan,
                        "p_proxy": np.nan,
                        "passed": 0,
                        "fit_family": "failed",
                        "fit_method": "failed",
                    }
                    for cov in preserve_covars:
                        row[f"p_preserve_{cov}"] = np.nan
                    for cov in reverse_guard_covars:
                        row[f"p_reverse_{cov}"] = np.nan

                rows.append(row)
                multiplicative_detail_rows.append(row)

        n_mult = good_pcs_mult.sum(axis=0)
        print(f"  QCs selected: min={n_mult.min()}, max={n_mult.max()}, mean={n_mult.mean():.1f}")
    else:
        print("\nStep 3: MULTIPLICATIVE correction disabled.")
        good_pcs_mult = np.zeros((num_qc_features, len(idp_list)), dtype=int)
        multiplicative_detail_rows = []

    # ------------------------------------------------------------------
    # STEP 4: APPLY CORRECTIONS
    # ------------------------------------------------------------------
    print("\nStep 4: Applying corrections...")

    qc_selection: Dict[str, Any] = {
        "volumes": list(idp_list),
        "qc_basis": qc_basis,
        "apply_pca": apply_pca,
        "additive_qcs": [],
        "multiplicative_qcs": [],
        "additive_count": [],
        "multiplicative_count": [],
        "multiplicative_model_pval": [],
        "multiplicative_applied": [],
        "p_threshold": p_thr,
        "max_qcs": max_qcs,
        "preserve_covars": list(preserve_covars),
        "adjust_covars": list(adjust_covars),
        "categorical_covars": list(categorical_covars),
        "reverse_guard_covars": list(reverse_guard_covars),
        "qc_feature_names": list(qc_feature_names),
        "pca_num_components": num_qc_features if apply_pca else None,
        "pca_variance_explained": float(cumvar[num_qc_features - 1]) if apply_pca else None,
    }

    y_harmonised = np.full((len(data), len(idp_list)), np.nan, dtype=float)

    for v_idx, volname in enumerate(idp_list):
        print(f"\n=== Volume: {volname} ===")

        # ----------------------------
        # ADDITIVE CORRECTION
        # ----------------------------
        qc_mask_add = list(np.where(good_pcs_add[:, v_idx] == 1)[0] + 1)
        if not math.isinf(max_qcs) and len(qc_mask_add) > int(max_qcs):
            qc_mask_add = qc_mask_add[: int(max_qcs)]

        if len(qc_mask_add) == 0:
            print("  Additive correction: skipped (no QCs passed)")
            tmp_no_add = data[volname].astype(float).copy()
            qc_selection["additive_qcs"].append([])
            qc_selection["additive_count"].append(0)
        else:
            qc_vars_add = [qc_feature_names[i - 1] for i in qc_mask_add]
            print(f"  Additive correction: applying {len(qc_vars_add)} QC(s) -> {qc_vars_add}")

            dfm = complete_case(
                data,
                [subject_col] + list(preserve_covars) + list(adjust_covars) + [volname] + qc_vars_add,
            )
            formula_add = f"{volname} ~ {rhs_expr(list(preserve_covars) + list(adjust_covars) + qc_vars_add)} + (1|{subject_col})"

            fit_add, family_add, method_add = fit_mixed_then_ols(dfm, formula_add, subject_col)
            beta = fixed_params(fit_add)
            beta_qc = beta.reindex(qc_vars_add).fillna(0.0).to_numpy(dtype=float)
            add_effect = dfm.loc[:, qc_vars_add].to_numpy(dtype=float) @ beta_qc

            tmp_no_add = data[volname].astype(float).copy()
            tmp_no_add.loc[dfm.index] = dfm[volname].to_numpy(dtype=float) - add_effect

            qc_selection["additive_qcs"].append(qc_vars_add)
            qc_selection["additive_count"].append(len(qc_mask_add))

        tmp_no_add = tmp_no_add.clip(lower=1e-6)

        # ----------------------------
        # MULTIPLICATIVE CORRECTION
        # ----------------------------
        if enable_multiplicative:
            print("  Multiplicative selection: using cached biology-only proxy")

            resid_proxy = bio_cache[volname]["resid_proxy"]

            qc_mask_mult = list(np.where(good_pcs_mult[:, v_idx] == 1)[0] + 1)
            if not math.isinf(max_qcs) and len(qc_mask_mult) > int(max_qcs):
                qc_mask_mult = qc_mask_mult[: int(max_qcs)]

            if len(qc_mask_mult) == 0:
                print("  Multiplicative correction: skipped (no QCs passed)")
                y_h = tmp_no_add.copy()
                qc_selection["multiplicative_qcs"].append([])
                qc_selection["multiplicative_count"].append(0)
                qc_selection["multiplicative_model_pval"].append(np.nan)
                qc_selection["multiplicative_applied"].append(False)
            else:
                qc_vars_mult = [qc_feature_names[i - 1] for i in qc_mask_mult]
                print(f"  Multiplicative correction: applying {len(qc_vars_mult)} QC(s) -> {qc_vars_mult}")

                dfm = data.loc[:, [subject_col] + list(preserve_covars) + list(adjust_covars) + qc_vars_mult].copy()
                dfm["tmp_noAdditive"] = tmp_no_add
                dfm = dfm.join(resid_proxy, how="inner").dropna().copy()
                dfm["log_tmp_noAdditive"] = np.log(dfm["tmp_noAdditive"].astype(float))

                formula_red = f"log_tmp_noAdditive ~ {rhs_expr(list(preserve_covars) + list(adjust_covars))} + (1|{subject_col})"
                formula_full = f"log_tmp_noAdditive ~ {rhs_expr(list(preserve_covars) + list(adjust_covars) + qc_vars_mult)} + (1|{subject_col})"

                fit_full, fit_red, family_mult, method_full, method_red = fit_pair_same_family(dfm, formula_full, formula_red, subject_col)
                model_pval = lrt_pvalue(fit_full, fit_red, family_mult)
                qc_selection["multiplicative_model_pval"].append(model_pval)

                print(f"  Multiplicative model comparison p-value: {model_pval:.6g}")

                if model_pval < p_thr:
                    beta = fixed_params(fit_full)
                    beta_qc = beta.reindex(qc_vars_mult).fillna(0.0).to_numpy(dtype=float)
                    lp = dfm.loc[:, qc_vars_mult].to_numpy(dtype=float) @ beta_qc
                    lp_centered = lp - np.nanmean(lp)
                    multiplicative_effect = np.exp(lp_centered)

                    y_h = tmp_no_add.copy()
                    y_h.loc[dfm.index] = dfm["tmp_noAdditive"].to_numpy(dtype=float) / multiplicative_effect

                    qc_selection["multiplicative_qcs"].append(qc_vars_mult)
                    qc_selection["multiplicative_count"].append(len(qc_mask_mult))
                    qc_selection["multiplicative_applied"].append(True)
                    print("  Multiplicative correction: applied")
                else:
                    y_h = tmp_no_add.copy()
                    qc_selection["multiplicative_qcs"].append([])
                    qc_selection["multiplicative_count"].append(0)
                    qc_selection["multiplicative_applied"].append(False)
                    print("  Multiplicative correction: skipped (model p >= threshold)")
        else:
             y_h = tmp_no_add.copy()
             qc_selection["multiplicative_qcs"].append([])
             qc_selection["multiplicative_count"].append(0)
             qc_selection["multiplicative_model_pval"].append(np.nan)
             qc_selection["multiplicative_applied"].append(False)
             print("  Multiplicative correction skipped.")
    
        data[f"harmonised_{volname}"] = y_h.to_numpy(dtype=float)
        y_harmonised[:, v_idx] = y_h.to_numpy(dtype=float)


    # ------------------------------------------------------------------
    # STEP 5: SAVE AND SUMMARISE
    # ------------------------------------------------------------------
    print("\nStep 5: Saving results...")

    data.to_csv(outfilename, index=False)
    print("  Data saved:", outfilename)

    summary_df = pd.DataFrame({
        "volume": qc_selection["volumes"],
        "additive_count": qc_selection["additive_count"],
        "multiplicative_count": qc_selection["multiplicative_count"],
        "multiplicative_applied": qc_selection["multiplicative_applied"],
        "multiplicative_model_pval": qc_selection["multiplicative_model_pval"],
    })
    summary_df.to_csv(summary_csv, index=False)
    print("  Summary saved:", summary_csv)

    add_detail_df = pd.DataFrame(additive_detail_rows)
    add_detail_df.to_csv(additive_detail_csv, index=False)
    print("  Additive details saved:", additive_detail_csv)

    mult_detail_df = pd.DataFrame(multiplicative_detail_rows)
    mult_detail_df.to_csv(multiplicative_detail_csv, index=False)
    print("  Multiplicative details saved:", multiplicative_detail_csv)

    pd.DataFrame({
        "volume": qc_selection["volumes"],
        "selected_qcs": [";".join(x) for x in qc_selection["additive_qcs"]],
    }).to_csv(selected_additive_qcs_csv, index=False)

    pd.DataFrame({
        "volume": qc_selection["volumes"],
        "selected_qcs": [";".join(x) for x in qc_selection["multiplicative_qcs"]],
    }).to_csv(selected_multiplicative_qcs_csv, index=False)

    print("  Selected QC lists saved.")

    print("\n=== SUMMARY ===")
    for i, vol in enumerate(qc_selection["volumes"]):
        print(
            f"{vol}: "
            f"additive={qc_selection['additive_count'][i]} QC(s), "
            f"multiplicative={qc_selection['multiplicative_count'][i]} QC(s), "
            f"multiplicative_applied={qc_selection['multiplicative_applied'][i]}"
        )

    print("\nDone!")
    return data, qc_selection, add_detail_df, mult_detail_df, summary_df