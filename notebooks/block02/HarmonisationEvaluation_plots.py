import numpy as np
from functools import wraps
from typing import Any, Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import pandas as pd


def build_style_registry(subjects,
                         idps,
                         subject_palette="tab10",
                         idp_palette="Set2",
                         subject_markers=None,
                         idp_markers=None):
    """
    Returns:
      subject_style: dict {subject: (color, marker)}
      idp_style: dict {idp: (color, marker)}
    """

    if subject_markers is None:
        subject_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h']
    if idp_markers is None:
        idp_markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*']

    subj_colors = sns.color_palette(subject_palette, len(subjects))
    idp_colors = sns.color_palette(idp_palette, len(idps))

    subject_style = {
        s: (subj_colors[i % len(subj_colors)],
            subject_markers[i % len(subject_markers)])
        for i, s in enumerate(subjects)
    }

    idp_style = {
        i_name: (idp_colors[i % len(idp_colors)],
                 idp_markers[i % len(idp_markers)])
        for i, i_name in enumerate(idps)
    }

    return subject_style, idp_style

def plot_WithinSubjVar(
    df,
    subject_col='subject',
    idp_cols=None,
    subject_style=None,
    idp_style=None,
    limit_subjects=10,
    limit_idps_for_legend=10,
    figsize=(14,6),
    point_size=60,
    jitter=0.08,
    savepath=None,
    rep=None,
    show: bool = False):
    if idp_cols is None:
        idp_cols = [c for c in df.columns if c != subject_col]
    subjects = df[subject_col].tolist()
    idps = [c for c in df.columns if c != subject_col]
    subject_style, idp_style = build_style_registry(
        subjects,
        idps,
        subject_palette="tab10",
        idp_palette="Set2"
)

    subjects = df[subject_col].unique().tolist()
    idps = list(idp_cols)

    if subject_style is None or idp_style is None:
        raise ValueError("Provide subject_style and idp_style from build_style_registry()")

    n_subjects = len(subjects)
    n_idps = len(idps)

    long = df.melt(id_vars=subject_col, value_vars=idps,
                   var_name='IDP', value_name='value')

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.3)

    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # -------- A: Per-IDP boxplots + subject points --------
    sns.boxplot(x='IDP', y='value', data=long, ax=axA, boxprops={'alpha':0.6})
    axA.set_title("Per-IDP distribution (subjects)")
    axA.set_xlabel("")
    axA.set_ylabel("WSV (%)")
    axA.set_xticklabels(axA.get_xticklabels(),
                    rotation=45 if len(idps) > 4 else 0,
                    ha='right' if len(idps) > 4 else 'center')


    show_subject_legend = n_subjects <= limit_subjects
    idp_x = {idp: i for i, idp in enumerate(idps)}

    for subj in subjects:
        row = df[df[subject_col] == subj]
        xs, ys = [], []
        for idp in idps:
            xs.append(idp_x[idp] + np.random.uniform(-jitter, jitter))
            ys.append(row[idp].values[0])

        if show_subject_legend:
            color, marker = subject_style[subj]
            axA.scatter(xs, ys, s=point_size, marker=marker,
                        color=color, edgecolor='k', label=subj, zorder=3)
        else:
            axA.scatter(xs, ys, s=point_size*0.7, marker='o',
                        color='gray', edgecolor='k', alpha=0.8)

    if show_subject_legend:
        axA.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc='upper left')

    # -------- B: Mean across subjects per IDP --------
    idp_means = df[idps].mean(axis=0)
    sns.boxplot(x=idp_means.values, ax=axB, orient='h', boxprops={'alpha':0.6})
    axB.set_title("Per-IDP mean (across subjects)")
    axB.set_yticks([])

    show_idp_legend = n_idps <= limit_idps_for_legend
    for idp, mean_val in idp_means.items():
        if show_idp_legend:
            color, marker = idp_style[idp]
            axB.scatter(mean_val, 0, s=90, marker=marker,
                        color=color, edgecolor='k', label=idp)
        else:
            axB.scatter(mean_val, 0, s=70, marker='o',
                        color='gray', edgecolor='k')

    if show_idp_legend:
        axB.legend(title="IDP", bbox_to_anchor=(1.02, 1), loc='upper left')

    # -------- C: Mean across IDPs per subject --------
    subj_means = df.set_index(subject_col)[idps].mean(axis=1)
    sns.boxplot(x=subj_means.values, ax=axC, orient='h', boxprops={'alpha':0.6})
    axC.set_title("Per-subject mean (across IDPs)")
    axC.set_yticks([])

    show_subject_legend = n_subjects <= limit_subjects
    for subj, mean_val in subj_means.items():
        if show_subject_legend:
            color, marker = subject_style[subj]
            axC.scatter(mean_val, 0, s=90, marker=marker,
                        color=color, edgecolor='k', label=subj)
        else:
            axC.scatter(mean_val, 0, s=70, marker='o',
                        color='gray', edgecolor='k')

    # if show_subject_legend:
    #     axC.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.suptitle("Within subject variability", fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if rep is not None:
        rep.log_plot(fig, "Within subject variability")
        plt.close(fig)
        return None, None  # or return a small marker that it was logged
    if show:
        plt.show()
    return None

def plot_SubjectOrder(df,
                      idp_col='IDP',
                      time_a_col='TimeA',
                      time_b_col='TimeB',
                      rho_col='SpearmanRho',
                      p_col='pValue',
                      times_order=None,
                      significance=0.05,
                      ncols=2,
                      figsize_per_plot=(4,4),
                      cmap="cividis",
                      fmt=".2f",
                      center=0,
                      vmax_abs=None,
                      limit_idps=None,
                      sample_method='first',   # 'first' or 'random'
                      random_state=None,
                      rep=None,
                      show: bool = False,
                      combine_method: str = "stouffer",   # 'stouffer' or 'fisher'
                      p_correction: str = "fdr_bh"        # 'fdr_bh', 'bonferroni', or None
                      ):
    """
    Extended version of your function that *combines* p-values across IDPs (for each time-pair)
    and across time-pairs (for each IDP) using either Stouffer (signed) or Fisher, and optionally
    applies multiple-testing correction (BH or bonferroni).

    Notes:
      - combine_method='stouffer' uses the sign from rho (mean rho across IDPs for that cell)
        to create signed z-scores from two-sided p-values.
      - combine_method='fisher' uses scipy.stats.combine_pvalues(method='fisher') and ignores sign.
      - p_correction operates separately for the time×time summary matrix and for per-IDP combined p's.
      - For best statistical rigor with permutation-based tests, combining at the permutation-level
        (i.e. combining test stats per permutation and building an empirical null) is preferable.
    """
 

    # --- additional imports for combination ---
    try:
        from scipy.stats import combine_pvalues, norm
        from scipy.stats import chi2
    except Exception as e:
        raise ImportError("This function requires scipy. Please install scipy (pip install scipy).") from e

    # --- simple BH implementation and Bonferroni ---
    def bh_adjust(pvals):
        """Benjamini-Hochberg FDR correction. Returns adjusted p-values (same shape as input)."""
        p = np.asarray(pvals, dtype=float)
        flat = p.flatten()
        nanmask = np.isnan(flat)
        idx = np.where(~nanmask)[0]
        if idx.size == 0:
            return p  # nothing to do
        pv = flat[~nanmask]
        order = np.argsort(pv)
        ranked = np.empty_like(order)
        ranked[order] = np.arange(pv.size) + 1  # ranks 1..m
        m = pv.size
        adj = np.empty_like(pv)
        # BH adjusted p-values (step-up)
        adj_vals = pv * m / ranked
        # enforce monotonicity (cumulative minimum from largest to smallest)
        adj_vals_sorted = np.empty_like(adj_vals)
        adj_vals_sorted[order] = adj_vals
        # cumulative minimum from the end
        cummin = np.minimum.accumulate(adj_vals_sorted[::-1])[::-1]
        adj[order] = np.minimum(cummin, 1.0)
        flat_adj = flat.copy()
        flat_adj[~nanmask] = adj
        return flat_adj.reshape(p.shape)

    def bonferroni_adjust(pvals):
        p = np.asarray(pvals, dtype=float)
        flat = p.flatten()
        nanmask = np.isnan(flat)
        idx = np.where(~nanmask)[0]
        if idx.size == 0:
            return p
        pv = flat[~nanmask]
        m = pv.size
        adj = np.minimum(pv * m, 1.0)
        flat_adj = flat.copy()
        flat_adj[~nanmask] = adj
        return flat_adj.reshape(p.shape)

    def apply_correction(p_matrix, method):
        if method is None:
            return p_matrix
        if method == 'fdr_bh':
            return bh_adjust(p_matrix)
        elif method == 'bonferroni':
            return bonferroni_adjust(p_matrix)
        else:
            raise ValueError("p_correction must be 'fdr_bh', 'bonferroni', or None")

    # ---------- Validate and list IDPs ----------
    all_idps = sorted(df[idp_col].unique())
    if len(all_idps) == 0:
        raise ValueError("No IDPs found in dataframe.")

    # choose idps_to_plot according to limit_idps and sample_method
    if limit_idps is None:
        idps_to_plot = all_idps.copy()
    else:
        if not (isinstance(limit_idps, int) and limit_idps >= 1):
            raise ValueError("limit_idps must be None or a positive integer.")
        limit = min(limit_idps, len(all_idps))
        if sample_method == 'first':
            idps_to_plot = all_idps[:limit]
        elif sample_method == 'random':
            rng = random.Random(random_state)
            idps_to_plot = rng.sample(all_idps, limit)
        else:
            raise ValueError("sample_method must be 'first' or 'random'.")

    n_idp_plot = len(idps_to_plot)

    # ---------- Determine time ordering ----------
    if times_order is None:
        times = sorted(set(df[time_a_col].unique()) | set(df[time_b_col].unique()))
    else:
        times = list(times_order)
    n_times = len(times)
    if n_times == 0:
        raise ValueError("No time points found.")

    # ---------- Build matrices for ALL IDPs (used for summaries) ----------
    rho_mats_all = {}
    p_mats_all = {}
    all_rho_values = []
    for idp in all_idps:
        sub = df[df[idp_col] == idp].copy()
        rho = sub.pivot(index=time_a_col, columns=time_b_col, values=rho_col)
        pmat = sub.pivot(index=time_a_col, columns=time_b_col, values=p_col)
        rho = rho.reindex(index=times, columns=times)
        pmat = pmat.reindex(index=times, columns=times)
        # symmetrize if needed (use transpose to fill missing)
        rho = rho.combine_first(rho.T)
        pmat = pmat.combine_first(pmat.T)
        rho = rho.combine_first(rho.T)
        pmat = pmat.combine_first(pmat.T)

        rho_arr = rho.to_numpy(dtype=float, copy=True)
        pmat_arr = pmat.to_numpy(dtype=float, copy=True)

        np.fill_diagonal(rho_arr, np.nan)
        np.fill_diagonal(pmat_arr, np.nan)

        rho_mats_all[idp] = pd.DataFrame(rho_arr, index=times, columns=times)
        p_mats_all[idp] = pd.DataFrame(pmat_arr, index=times, columns=times)

        all_rho_values.extend(rho.values.flatten()[~np.isnan(rho.values.flatten())])

    if len(all_rho_values) == 0:
        raise ValueError("No numeric SpearmanRho values found.")

    # ---------- Color scale ----------
    if vmax_abs is None:
        vmax_abs = max(abs(np.nanmin(all_rho_values)), abs(np.nanmax(all_rho_values)))
    vmin, vmax = -vmax_abs, vmax_abs

    # ---------- Summary calculations (use ALL IDPs) ----------
    stacked = np.stack([rho_mats_all[idp].values for idp in all_idps], axis=0)    # shape (n_idps, n_times, n_times)
    stacked_p = np.stack([p_mats_all[idp].values for idp in all_idps], axis=0)

    # Helper: ensure p in (0,1]; replace exact zeros by tiny value to avoid -inf/log issues
    tiny = 1e-300
    sp = stacked_p.copy()
    sp[np.isnan(sp)] = np.nan  # leave nans
    sp[sp == 0] = tiny

    # Combined p-values per time-pair across IDPs
    combined_p_matrix = np.full((n_times, n_times), np.nan)
    combined_rho_matrix = np.nanmean(stacked, axis=0)  # keep mean rho (for sign in Stouffer)
    if combine_method.lower() == 'fisher':
        # use scipy combine_pvalues for Fisher (ignores sign)
        for i in range(n_times):
            for j in range(n_times):
                pv = sp[:, i, j]
                pv = pv[~np.isnan(pv)]
                if pv.size == 0:
                    combined_p_matrix[i, j] = np.nan
                else:
                    # combine_pvalues returns (stat, p)
                    _, p_comb = combine_pvalues(pv, method='fisher')
                    combined_p_matrix[i, j] = p_comb
    elif combine_method.lower() == 'stouffer':
        # signed Stouffer: convert two-sided p to z, use sign of mean rho across IDPs for that cell
        # z_i = sign_i * norm.ppf(1 - p_i/2)
        for i in range(n_times):
            for j in range(n_times):
                pv = sp[:, i, j]
                pv = pv[~np.isnan(pv)]
                if pv.size == 0:
                    combined_p_matrix[i, j] = np.nan
                else:
                    # determine sign from mean rho across IDPs for that cell
                    rhos = stacked[:, i, j]
                    rhos_nonan = rhos[~np.isnan(rhos)]
                    sign_cell = 0
                    if rhos_nonan.size > 0:
                        mean_rho = np.nanmean(rhos_nonan)
                        sign_cell = np.sign(mean_rho) if not np.isnan(mean_rho) else 1.0
                        if sign_cell == 0:
                            sign_cell = 1.0
                    else:
                        sign_cell = 1.0
                    # convert p to z's, protect from p==0 or p==1
                    p_clip = np.clip(pv, tiny, 1 - 1e-16)
                    zs = norm.ppf(1.0 - p_clip / 2.0)  # two-sided -> two-tailed z magnitude
                    # apply sign
                    signed_zs = sign_cell * zs
                    # combine (equal weights)
                    z_comb = np.sum(signed_zs) / math.sqrt(zs.size)
                    # two-sided combined p:
                    p_comb = 2.0 * (1.0 - norm.cdf(abs(z_comb)))
                    combined_p_matrix[i, j] = float(np.clip(p_comb, tiny, 1.0))
    else:
        raise ValueError("combine_method must be 'stouffer' or 'fisher'")

    # Optional multiple-comparison correction across time×time combined p-matrix
    combined_p_matrix_adj = apply_correction(combined_p_matrix, p_correction)

    # ---------- Per-IDP: combine its off-diagonal p-values into a single p -----
    idp_combined_ps = []
    idp_mean_rhos = []
    for idp in all_idps:
        pmat = p_mats_all[idp].values
        rho = rho_mats_all[idp].values
        mask = ~np.eye(n_times, dtype=bool)  # exclude diagonal
        pv = pmat[mask]
        pv = pv[~np.isnan(pv)]
        rhos = rho[mask]
        rhos = rhos[~np.isnan(rhos)]
        if pv.size == 0:
            idp_combined_ps.append(np.nan)
            idp_mean_rhos.append(np.nan)
            continue
        if combine_method.lower() == 'fisher':
            # fisher combine
            _, p_comb = combine_pvalues(np.clip(pv, tiny, 1.0), method='fisher')
            idp_combined_ps.append(float(p_comb))
        else:  # stouffer signed
            mean_rho = np.nanmean(rho[mask])
            idp_mean_rhos.append(mean_rho)
            sign_cell = np.sign(mean_rho) if not np.isnan(mean_rho) else 1.0
            if sign_cell == 0:
                sign_cell = 1.0
            p_clip = np.clip(pv, tiny, 1.0 - 1e-16)
            zs = norm.ppf(1.0 - p_clip / 2.0)
            signed_zs = sign_cell * zs
            z_comb = np.sum(signed_zs) / math.sqrt(zs.size)
            p_comb = 2.0 * (1.0 - norm.cdf(abs(z_comb)))
            idp_combined_ps.append(float(np.clip(p_comb, tiny, 1.0)))
    # if mean rhos list wasn't filled (Fisher branch), compute idp_mean_rhos now
    if combine_method.lower() == 'fisher':
        idp_mean_rhos = []
        for idp in all_idps:
            rho = rho_mats_all[idp].values
            mask = ~np.eye(n_times, dtype=bool)
            idp_mean_rhos.append(np.nanmean(rho[mask]) if ~np.all(np.isnan(rho[mask])) else np.nan)
    else:
        # ensure lengths correct
        if len(idp_mean_rhos) != len(all_idps):
            # fallback compute
            idp_mean_rhos = []
            for idp in all_idps:
                rho = rho_mats_all[idp].values
                mask = ~np.eye(n_times, dtype=bool)
                idp_mean_rhos.append(np.nanmean(rho[mask]) if ~np.all(np.isnan(rho[mask])) else np.nan)

    idp_combined_ps = np.array(idp_combined_ps, dtype=float)
    idp_combined_ps_adj = idp_combined_ps.copy()
    if p_correction is not None:
        # apply correction across IDPs (treating them as a family)
        idp_combined_ps_adj = apply_correction(idp_combined_ps.reshape(-1, 1), p_correction).reshape(-1)

    # ---------- Prepare mean_rho_matrix (mean across IDPs) ----------
    mean_rho_matrix = np.nanmean(stacked, axis=0)
    np.fill_diagonal(mean_rho_matrix, np.nan)

    # ---------- Figure layout ----------
    nrows = math.ceil(n_idp_plot / ncols) if n_idp_plot > 0 else 0
    fig_w = figsize_per_plot[0] * ncols
    fig_h = max(1, nrows) * figsize_per_plot[1] + 1.6 * figsize_per_plot[1]
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(max(1, nrows) + 2, ncols,
                          height_ratios=[1]*max(1, nrows) + [0.9, 0.9],
                          hspace=0.35, wspace=0.3)

    # ---------- Plot individual IDP heatmaps (ONLY idps_to_plot) ----------
    for idx, idp in enumerate(idps_to_plot):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])
        rho = rho_mats_all[idp]
        pmat = p_mats_all[idp]
        annot = np.full(rho.shape, "", dtype=object)
        for i in range(n_times):
            for j in range(n_times):
                val = rho.iat[i, j]
                pval = pmat.iat[i, j]
                if np.isnan(val):
                    annot[i, j] = ""
                else:
                    star = "*" if (not pd.isna(pval) and pval < significance) else ""
                    annot[i, j] = f"{val:{fmt}}{star}"
        sns.heatmap(rho,
                    ax=ax,
                    annot=annot,
                    fmt="",
                    cmap=cmap,
                    center=center,
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0.35,
                    linecolor="gray",
                    cbar=False,
                    square=False)
        ax.set_title(idp, fontsize=9)
        ax.set_xlabel("")   # explicit: no xlabel
        ax.set_ylabel("")   # explicit: no ylabel
        ax.set_xticklabels(times, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(times, rotation=0, fontsize=7)

    # hide unused axes inside the idp grid
    total_idp_slots = max(1, nrows) * ncols
    if n_idp_plot < total_idp_slots:
        for k in range(n_idp_plot, total_idp_slots):
            r = k // ncols
            c = k % ncols
            ax = fig.add_subplot(gs[r, c])
            ax.axis('off')

    # ---------- Summary 1: mean across ALL IDPs (time x time) ----------
    row_for_summaries = max(0, nrows)
    ax_mean_timepair = fig.add_subplot(gs[row_for_summaries, :])
    annot_mean = np.full(mean_rho_matrix.shape, "", dtype=object)
    # use combined_p_matrix_adj (corrected) to mark significance
    for i in range(n_times):
        for j in range(n_times):
            val = mean_rho_matrix[i, j]
            pval = combined_p_matrix_adj[i, j]
            if np.isnan(val):
                annot_mean[i, j] = ""
            else:
                star = "*" if (not np.isnan(pval) and pval < significance) else ""
                annot_mean[i, j] = f"{val:{fmt}}{star}"
    sns.heatmap(mean_rho_matrix,
                ax=ax_mean_timepair,
                annot=annot_mean,
                fmt="",
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.35,
                linecolor="gray",
                cbar=False,
                square=False)
    ax_mean_timepair.set_title(f"Mean across  {len(all_idps)} IDPs (per time-pair) — combined p: {combine_method}, correction: {p_correction}", fontsize=10)
    ax_mean_timepair.set_xlabel("")
    ax_mean_timepair.set_ylabel("")
    ax_mean_timepair.set_xticklabels(times, rotation=45, ha='right', fontsize=7)
    ax_mean_timepair.set_yticklabels(times, rotation=0, fontsize=7)

    # ---------- Summary 2: per-IDP mean across time-pairs (ALL IDPs) ----------
    ax_idp_mean = fig.add_subplot(gs[row_for_summaries+1, :])
    idp_mean_matrix = np.array(idp_mean_rhos).reshape(-1, 1)
    annot_idp = np.array([f"{v:{fmt}}{'*' if (not np.isnan(p) and p < significance) else ''}"
                          for v, p in zip(idp_mean_rhos, idp_combined_ps_adj)]).reshape(-1, 1)
    sns.heatmap(idp_mean_matrix,
                ax=ax_idp_mean,
                annot=annot_idp,
                fmt="",
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.35,
                linecolor="gray",
                cbar=False,
                yticklabels=all_idps,
                xticklabels=["MeanAcrossTimePairs"],
                square=False)
    ax_idp_mean.set_title(f"Per-IDP mean across time-pairs ({len(all_idps)} IDPs) — combined p: {combine_method}, correction: {p_correction}", fontsize=8)
    ax_idp_mean.set_xlabel("")
    ax_idp_mean.set_ylabel("")
    ax_idp_mean.set_xticklabels([""], rotation=0)

    # ---------- Shared colorbar ----------
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Spearman Rho')

    plt.suptitle(f"Subject order consistency summaries computed from {len(all_idps)} IDPs\n('*' indicates p < {significance} from permutation testing)", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.90, 0.96])

    if rep is not None:
        rep.log_plot(fig, "Subject order consistency")
        plt.close(fig)
        return None, None
    if show:
        plt.show()
    return None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_additive_multiplicative_effects(df, feature_cols=None, batch_col="Site"):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ["Subject", "Timepoint", batch_col]]

    sites = df[batch_col].dropna().unique().tolist()
    n_features = len(feature_cols)

    fig, axes = plt.subplots(n_features, 1, figsize=(10, 4 * n_features), squeeze=False)

    for i, feat in enumerate(feature_cols):
        ax = axes[i, 0]

        data_by_site = [
            df.loc[df[batch_col] == site, feat].dropna().values
            for site in sites
        ]

        ax.boxplot(data_by_site, tick_labels=sites, showmeans=True)
        ax.set_title(f"{feat}: additive shift shows as center change, multiplicative effect as spread change")
        ax.set_xlabel(batch_col)
        ax.set_ylabel(feat)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
    return None