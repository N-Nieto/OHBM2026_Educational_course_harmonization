# nomogram_plotter.py

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output


def get_continuous_covariates(df, batch_col="batch", outcome_col="y"):
    """
    Return numeric columns excluding batch and outcome.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in {batch_col, outcome_col}]


def plot_charts(df, covariate_names, batch_col="batch", outcome_col="y", ncols=3):
    """
    Plot outcome vs selected continuous covariates, colored by batch,
    with an overall regression line and 95% confidence band.

    Parameters
    ----------
    df : pandas.DataFrame
    covariate_names : list[str]
        Covariates to plot.
    batch_col : str
        Column containing batch labels.
    outcome_col : str
        Outcome column.
    ncols : int
        Number of subplot columns.

    Returns
    -------
    matplotlib.figure.Figure
    """

    if isinstance(covariate_names, str):
        covariate_names = [covariate_names]

    covariate_names = [c for c in covariate_names if c in df.columns]
    if len(covariate_names) == 0:
        raise ValueError("No valid covariates were selected.")

    nplots = len(covariate_names)
    nrows = math.ceil(nplots / ncols)

    palette = sns.color_palette("tab10", n_colors=df[batch_col].nunique())

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    axes = axes.ravel()

    for i, cov in enumerate(covariate_names):
        ax = axes[i]

        sns.scatterplot(
            data=df,
            x=cov,
            y=outcome_col,
            hue=batch_col,
            palette=palette,
            alpha=0.6,
            edgecolor=None,
            s=35,
            ax=ax,
            legend=(i == 0),  # show legend only on first subplot
        )

        sns.regplot(
            data=df,
            x=cov,
            y=outcome_col,
            scatter=False,
            ci=95,
            color="black",
            line_kws={"linewidth": 1.8},
            ax=ax,
        )

        ax.set_title(f"Outcome vs {cov}")
        ax.set_xlabel(cov)
        ax.set_ylabel(outcome_col)

        if i != 0:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        else:
            ax.legend(title=batch_col, loc="best")

    # Remove any unused axes
    for j in range(nplots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig


def chart_selector_gui(df, batch_col="batch", outcome_col="y", ncols=3):
    """
    Interactive selector for choosing covariates before plotting.
    """

    covariates = get_continuous_covariates(
        df, batch_col=batch_col, outcome_col=outcome_col
    )

    selector = widgets.SelectMultiple(
        options=covariates,
        value=tuple(covariates[: min(3, len(covariates))]),
        description="Covariates",
        layout=widgets.Layout(width="600px", height="180px"),
    )

    plot_button = widgets.Button(
        description="Plot nomogram", button_style="primary", icon="chart-line"
    )

    output = widgets.Output()

    def on_plot_clicked(_):
        with output:
            clear_output(wait=True)
            selected = list(selector.value)
            if len(selected) == 0:
                print("Select at least one covariate.")
                return

            fig = plot_nomogram(
                df=df,
                covariate_names=selected,
                batch_col=batch_col,
                outcome_col=outcome_col,
                ncols=ncols,
            )
            plt.show(fig)

    plot_button.on_click(on_plot_clicked)

    ui = widgets.VBox(
        [
            widgets.HTML("<h3>Nomogram covariate selector</h3>"),
            selector,
            plot_button,
            output,
        ]
    )

    return ui


def plot_age_percentile_chart(
    df,
    age_col="age",
    value_col="y",
    batch_col="batch",
    percentiles=(5, 25, 50, 75, 95),
    n_bins=25,
    smooth_frac=0.25,
    show_points=True,
    show_batch_coloring=True,
    figsize=(10, 6),
    title=None,
):
    """
    Plot smoothed percentile curves of a value against age.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    age_col : str
        Age variable to use on the x-axis.
    value_col : str
        Outcome / feature to summarize on the y-axis.
    batch_col : str or None
        Optional batch column for point coloring.
    percentiles : tuple[int]
        Percentiles to plot.
    n_bins : int
        Number of age bins used to estimate local percentile curves.
    smooth_frac : float
        LOWESS smoothing fraction applied to the binned percentile estimates.
    show_points : bool
        If True, scatter the raw data faintly.
    show_batch_coloring : bool
        If True, use batch colors for raw points.
    figsize : tuple
        Figure size.
    title : str or None
        Plot title. If None, a default title is used.

    Returns
    -------
    fig, ax
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as e:
        raise ImportError(
            "This function requires statsmodels. Install it with `pip install statsmodels`."
        ) from e

    if age_col not in df.columns:
        raise ValueError(f"'{age_col}' was not found in the dataframe.")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' was not found in the dataframe.")
    if batch_col is not None and batch_col not in df.columns:
        batch_col = None

    d = df[[age_col, value_col] + ([batch_col] if batch_col else [])].copy()
    d = d.dropna(subset=[age_col, value_col])

    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[age_col, value_col])

    if len(d) < 10:
        raise ValueError("Not enough data to plot percentile curves.")

    # Build age bins
    try:
        bins = pd.qcut(d[age_col], q=min(n_bins, d.shape[0]), duplicates="drop")
    except ValueError:
        bins = pd.cut(d[age_col], bins=min(n_bins, max(3, d.shape[0] // 2)))

    d = d.assign(_age_bin=bins)

    # Compute binned percentile estimates
    rows = []
    for bin_id, g in d.groupby("_age_bin", observed=True):
        if g.shape[0] < 3:
            continue
        row = {
            "age_center": float(g[age_col].median()),
            "n": int(g.shape[0]),
        }
        for p in percentiles:
            row[f"p{p}"] = float(np.percentile(g[value_col], p))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("age_center")
    if summary.empty:
        raise ValueError(
            "Could not compute percentile bins. Try fewer bins or more data."
        )

    x_grid = np.linspace(d[age_col].min(), d[age_col].max(), 400)

    def smooth_curve(x, y, frac=smooth_frac):
        """
        LOWESS smoothing on binned percentile estimates, then interpolate to the grid.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if len(x) < 4:
            return np.interp(x_grid, x, y)

        sm = lowess(y, x, frac=frac, return_sorted=True)
        xs, ys = sm[:, 0], sm[:, 1]

        # Ensure unique x values for interpolation
        xs_unique, idx = np.unique(xs, return_index=True)
        ys_unique = ys[idx]

        return np.interp(x_grid, xs_unique, ys_unique)

    curves = {}
    for p in percentiles:
        curves[p] = smooth_curve(summary["age_center"], summary[f"p{p}"])

    fig, ax = plt.subplots(figsize=figsize)

    # Raw points, if requested
    if show_points:
        if show_batch_coloring and batch_col is not None:
            batches = list(pd.unique(d[batch_col]))
            cmap = plt.get_cmap("tab10")
            for i, b in enumerate(batches):
                g = d[d[batch_col] == b]
                ax.scatter(
                    g[age_col],
                    g[value_col],
                    s=12,
                    alpha=0.18,
                    label=str(b),
                    color=cmap(i % 10),
                    edgecolors="none",
                )
        else:
            ax.scatter(
                d[age_col],
                d[value_col],
                s=12,
                alpha=0.15,
                color="0.5",
                edgecolors="none",
            )

    # Shaded percentile bands
    ax.fill_between(
        x_grid,
        curves[25],
        curves[75],
        alpha=0.18,
        label="25th-75th",
    )
    ax.fill_between(
        x_grid,
        curves[5],
        curves[95],
        alpha=0.08,
        label="5th-95th",
    )

    # Main percentile curves
    for p in percentiles:
        linewidth = 2.6 if p == 50 else 1.6
        ax.plot(x_grid, curves[p], linewidth=linewidth, label=f"{p}th percentile")

    ax.set_xlabel(age_col.capitalize())
    ax.set_ylabel(value_col.capitalize() if value_col else "Value")
    ax.set_title(title or f"Smoothed percentile chart of {value_col} by age")

    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Keep the legend readable without clutter
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, ncol=2, loc="best")

    fig.tight_layout()
    return fig, ax

def zscore_reference(a, ref):
    """
    Z-score a using mean/sd from ref, column-wise.
    """
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (a - mu) / sd

def plot_combat_before_after(
    y_before,
    y_after,
    batch,
    feature_names=None,
    feature_idx=0,
    bins=30,
):
    """
    Visualise harmonisation before vs after ComBat.

    Parameters
    ----------
    y_before : array-like, shape (n, p)
    y_after  : array-like, shape (n, p)
    batch    : array-like, shape (n,)
    feature_names : list[str] or None
        Column names for the features.
    feature_idx : int
        Which feature to show in the histogram panel.
    bins : int
        Number of bins for histograms.
    """
    y_before = np.asarray(y_before)
    y_after = np.asarray(y_after)
    batch = np.asarray(batch)

    if y_before.ndim == 1:
        y_before = y_before.reshape(-1, 1)
    if y_after.ndim == 1:
        y_after = y_after.reshape(-1, 1)

    if y_before.shape != y_after.shape:
        raise ValueError("`y_before` and `y_after` must have the same shape.")
    if y_before.shape[0] != batch.shape[0]:
        raise ValueError("`batch` must have the same number of rows as the data.")

    n, p = y_before.shape

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(p)]
    if len(feature_names) != p:
        raise ValueError("`feature_names` must match the number of columns.")

    batches = pd.unique(batch)

    # -----------------------------
    # Panel 1: one-feature histograms
    # -----------------------------
    x_before = y_before[:, feature_idx]
    x_after = y_after[:, feature_idx]

    xmin = min(np.nanmin(x_before), np.nanmin(x_after))
    xmax = max(np.nanmax(x_before), np.nanmax(x_after))
    bin_edges = np.linspace(xmin, xmax, bins + 1)

    fig, axes = plt.subplots(
        2,
        len(batches),
        figsize=(4.2 * len(batches), 7),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    if len(batches) == 1:
        axes = np.array(axes).reshape(2, 1)

    for j, b in enumerate(batches):
        m = batch == b

        axes[0, j].hist(
            x_before[m],
            bins=bin_edges,
            density=True,
            alpha=0.8,
            edgecolor="black",
        )
        axes[0, j].axvline(np.nanmean(x_before[m]), linestyle="--", linewidth=1)
        axes[0, j].set_title(f"{b} before")
        axes[0, j].set_ylabel("Density")

        axes[1, j].hist(
            x_after[m],
            bins=bin_edges,
            density=True,
            alpha=0.8,
            edgecolor="black",
        )
        axes[1, j].axvline(np.nanmean(x_after[m]), linestyle="--", linewidth=1)
        axes[1, j].set_title(f"{b} after")
        axes[1, j].set_ylabel("Density")
        axes[1, j].set_xlabel(feature_names[feature_idx])

    fig.suptitle(
        f"ComBat harmonisation for {feature_names[feature_idx]}",
        y=1.02,
        fontsize=14,
    )
    plt.show()

    # -----------------------------------------
    # Panel 2: batch means across all features
    # Use z-scored values so features are comparable
    # -----------------------------------------
    before_z = zscore_reference(y_before, y_before)
    after_z = zscore_reference(y_after, y_before)

    before_df = pd.DataFrame(before_z, columns=feature_names)
    after_df = pd.DataFrame(after_z, columns=feature_names)
    before_df["batch"] = batch
    after_df["batch"] = batch

    before_means = before_df.groupby("batch")[feature_names].mean().reindex(batches)
    after_means = after_df.groupby("batch")[feature_names].mean().reindex(batches)

    vlim = np.nanmax(np.abs([before_means.to_numpy(), after_means.to_numpy()]))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(1.2 * p + 4, 0.7 * len(batches) + 3),
        constrained_layout=True,
    )

    im0 = axes[0].imshow(before_means, aspect="auto", interpolation="nearest", vmin=-vlim, vmax=vlim)
    axes[0].set_title("Batch mean before")
    axes[0].set_xticks(range(p))
    axes[0].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[0].set_yticks(range(len(batches)))
    axes[0].set_yticklabels(batches)

    im1 = axes[1].imshow(after_means, aspect="auto", interpolation="nearest", vmin=-vlim, vmax=vlim)
    axes[1].set_title("Batch mean after")
    axes[1].set_xticks(range(p))
    axes[1].set_xticklabels(feature_names, rotation=45, ha="right")
    axes[1].set_yticks(range(len(batches)))
    axes[1].set_yticklabels(batches)

    cbar = fig.colorbar(im1, ax=axes, shrink=0.85)
    cbar.set_label("Mean z-score")

    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plot_pca_before_after(
    y_before,
    y_after,
    batch,
    feature_names=None,
    n_components=2,
    standardize=True,
    show_arrows=True,
    figsize=(14, 6),
    title_before="Before ComBat",
    title_after="After ComBat",
):
    """
    PCA visualisation for harmonisation results before and after ComBat.

    Parameters
    ----------
    y_before : array-like, shape (n, p)
        Data before harmonisation.
    y_after : array-like, shape (n, p)
        Data after harmonisation.
    batch : array-like, shape (n,)
        Batch labels.
    feature_names : list[str] or None
        Optional feature names for axis annotations.
    n_components : int
        Number of PCA components to compute. Must be >= 2 for plotting.
    standardize : bool
        If True, standardise features before PCA.
    show_arrows : bool
        If True, draw arrows from before -> after batch centroids.
    figsize : tuple
        Figure size.
    title_before : str
        Left panel title.
    title_after : str
        Right panel title.

    Returns
    -------
    dict
        Contains:
        - "pca": fitted PCA object
        - "scaler": fitted scaler or None
        - "scores_before": PCA scores for before data
        - "scores_after": PCA scores for after data
        - "fig": matplotlib figure
        - "axes": matplotlib axes
    """
    y_before = np.asarray(y_before)
    y_after = np.asarray(y_after)
    batch = np.asarray(batch)

    if y_before.ndim == 1:
        y_before = y_before.reshape(-1, 1)
    if y_after.ndim == 1:
        y_after = y_after.reshape(-1, 1)

    if y_before.shape != y_after.shape:
        raise ValueError("`y_before` and `y_after` must have the same shape.")
    if y_before.shape[0] != batch.shape[0]:
        raise ValueError("`batch` must have the same number of rows as the data.")
    if n_components < 2:
        raise ValueError("`n_components` must be at least 2 for a 2D PCA plot.")

    n, p = y_before.shape

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(p)]
    if len(feature_names) != p:
        raise ValueError("`feature_names` must match the number of columns.")

    # Fit one PCA model on pooled data so before/after are directly comparable.
    pooled = np.vstack([y_before, y_after])

    if standardize:
        scaler = StandardScaler()
        pooled_use = scaler.fit_transform(pooled)
        before_use = scaler.transform(y_before)
        after_use = scaler.transform(y_after)
    else:
        scaler = None
        pooled_use = pooled
        before_use = y_before
        after_use = y_after

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(pooled_use)

    scores_before = pca.transform(before_use)
    scores_after = pca.transform(after_use)

    batches = pd.unique(batch)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    def plot_panel(ax, scores, title):
        for i, b in enumerate(batches):
            m = batch == b
            ax.scatter(
                scores[m, 0],
                scores[m, 1],
                s=20,
                alpha=0.75,
                label=str(b),
            )

            centroid = scores[m, :2].mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                s=120,
                marker="X",
                linewidths=1.5,
            )

        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.axhline(0, linewidth=0.8, alpha=0.4)
        ax.axvline(0, linewidth=0.8, alpha=0.4)

    plot_panel(axes[0], scores_before, title_before)
    plot_panel(axes[1], scores_after, title_after)

    # Add legend once
    axes[0].legend(title="Batch", loc="best")

    # Optional centroid arrows from before -> after
    if show_arrows:
        before_centroids = {
            b: scores_before[batch == b, :2].mean(axis=0) for b in batches
        }
        after_centroids = {
            b: scores_after[batch == b, :2].mean(axis=0) for b in batches
        }

        for b in batches:
            c0 = before_centroids[b]
            c1 = after_centroids[b]
            axes[0].annotate(
                "",
                xy=c1,
                xytext=c0,
                arrowprops=dict(arrowstyle="->", linewidth=1.2),
            )

    fig.suptitle("PCA before and after ComBat harmonisation", fontsize=14)

    return {
        "pca": pca,
        "scaler": scaler,
        "scores_before": scores_before,
        "scores_after": scores_after,
        "fig": fig,
        "axes": axes,
    }